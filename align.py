"""
stfa/align.py
=============
Main pipeline entry point for STFA pairwise slice alignment.

Public API
----------
pairwise_align_stfa(sliceA, sliceB, ...) -> (pi12, ini_n, ini_g, fin_n, fin_g)

Compatible return signature with old INCENT.pairwise_align(return_obj=True).
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from anndata import AnnData
from scipy.spatial.distance import cdist
from typing import Optional, Tuple

from .graph import build_knn_graph, compute_diffusion_signatures, detect_communities
from .costs import (
    compute_M_gene,
    compute_M_celltype,
    compute_M_neighborhood,
    compute_M_topo,
    compute_M_boundary,
    compute_M_shape_context,
    compute_M_anchor,
    compute_M_compact,
    compute_M_region_geom,
    fuse_costs,
)
from .solver import estimate_overlap_fraction, calibrate_rho, solve_ufgw, _norm_dist
from .utils import neighborhood_distribution, jensenshannon_divergence_backend


# ──────────────────────────────────────────────────────────────────────────────
# Internal metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_objectives(
    pi_mat: np.ndarray,
    jsd_neighborhood: np.ndarray,
    cosine_gene: np.ndarray
) -> Tuple[float, float]:
    """Weighted-sum objectives from a transport plan and pre-computed cost matrices."""
    obj_neighbor = float(np.sum(jsd_neighborhood * pi_mat))
    obj_gene_cos = float(np.sum(cosine_gene * pi_mat))
    return obj_neighbor, obj_gene_cos


def _graph_geodesic_dist(
    coords: np.ndarray,
    adj: sp.csr_matrix,
) -> Tuple[np.ndarray, float]:
    """
    Weighted graph geodesic distances using spatial edge lengths.

    Falls back to Euclidean distances if geodesics are ill-defined.
    """
    coords = np.asarray(coords, dtype=np.float64)
    D_euc = _norm_dist(cdist(coords, coords))

    if adj is None:
        return D_euc, 1.0

    adj_csr = sp.csr_matrix(adj)
    if adj_csr.nnz == 0:
        return D_euc, 1.0

    upper = sp.triu(adj_csr, k=1).tocoo()
    if upper.nnz == 0:
        return D_euc, 1.0

    edge_w = np.linalg.norm(coords[upper.row] - coords[upper.col], axis=1)
    W = sp.coo_matrix((edge_w, (upper.row, upper.col)), shape=adj_csr.shape).tocsr()
    W = W + W.T

    D_geo = csgraph.shortest_path(W, directed=False, unweighted=False)
    finite = np.isfinite(D_geo)
    if not np.any(finite):
        return D_euc, 1.0

    # Normalise finite geodesics and use Euclidean fallback across components.
    max_f = float(np.max(D_geo[finite]))
    D_geo = np.asarray(D_geo, dtype=np.float64) / (max_f + 1e-12)
    D_geo = np.where(finite, D_geo, D_euc)
    disconnected_frac = 1.0 - float(np.mean(finite))
    return _norm_dist(D_geo), disconnected_frac


def _build_geometry_matrix(
    coords: np.ndarray,
    adj: sp.csr_matrix,
    geodesic_weight: float,
    geodesic_max_cells: int,
) -> np.ndarray:
    """
    Blend Euclidean and graph-geodesic distances for GW geometry.

    A positive geodesic weight improves intrinsic shape preservation under
    slight nonrigid deformations while keeping rigid-transform compatibility.
    """
    coords = np.asarray(coords, dtype=np.float64)
    D_euc = _norm_dist(cdist(coords, coords))

    w = float(np.clip(geodesic_weight, 0.0, 1.0))
    if w <= 0.0 or coords.shape[0] > int(max(50, geodesic_max_cells)):
        return D_euc

    D_geo, disconnected_frac = _graph_geodesic_dist(coords, adj)

    # If many node pairs are disconnected, rely more on Euclidean geometry.
    w_eff = w * max(0.0, 1.0 - disconnected_frac)
    return ((1.0 - w_eff) * D_euc + w_eff * D_geo).astype(np.float64)


def _community_centroids(
    coords: np.ndarray,
    comm_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique community ids and their centroids."""
    labels = np.asarray(comm_labels)
    unique = np.unique(labels)
    if unique.size == 0:
        return unique, np.zeros((0, 2), dtype=np.float64)
    centroids = np.vstack([coords[labels == c].mean(axis=0) for c in unique])
    return unique, np.asarray(centroids, dtype=np.float64)


def _spatial_community_coupling(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    comm_A: np.ndarray,
    comm_B: np.ndarray,
) -> np.ndarray:
    """
    Build a community coupling from spatial centroids only.

    This is used as a robust fallback/blend when descriptor-only community
    matching is unstable.
    """
    _, cent_A = _community_centroids(np.asarray(coords_A, dtype=np.float64), comm_A)
    _, cent_B = _community_centroids(np.asarray(coords_B, dtype=np.float64), comm_B)

    n_cA = int(cent_A.shape[0])
    n_cB = int(cent_B.shape[0])
    if n_cA == 0 or n_cB == 0:
        return np.zeros((n_cA, n_cB), dtype=np.float64)

    # Canonicalise centroids to reduce global translation/scale sensitivity.
    A0 = cent_A - cent_A.mean(axis=0, keepdims=True)
    B0 = cent_B - cent_B.mean(axis=0, keepdims=True)
    sA = float(np.median(np.linalg.norm(A0, axis=1))) + 1e-12
    sB = float(np.median(np.linalg.norm(B0, axis=1))) + 1e-12
    A0 = A0 / sA
    B0 = B0 / sB

    D = cdist(A0, B0)
    q = float(np.quantile(D, 0.90))
    if np.isfinite(q) and q > 0:
        D = D / (q + 1e-12)

    pA = np.ones(n_cA, dtype=np.float64) / n_cA
    pB = np.ones(n_cB, dtype=np.float64) / n_cB

    try:
        import ot
        pi = ot.emd(pA, pB, D)
    except Exception:
        pi = np.outer(pA, pB)

    return np.asarray(pi, dtype=np.float64)


def _blend_couplings(
    pi_anchor: np.ndarray,
    pi_spatial: np.ndarray,
    spatial_blend: float,
) -> np.ndarray:
    """Blend descriptor and spatial community couplings and renormalize."""
    A = np.asarray(pi_anchor, dtype=np.float64)
    B = np.asarray(pi_spatial, dtype=np.float64)
    if A.shape != B.shape:
        return B if B.size > 0 else A

    w = float(np.clip(spatial_blend, 0.0, 1.0))
    out = (1.0 - w) * A + w * B
    total = float(out.sum())
    if total <= 0:
        fallback = B if float(B.sum()) > 0 else A
        return np.asarray(fallback, dtype=np.float64)
    return out / (total + 1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_align_stfa(
    sliceA: AnnData,
    sliceB: AnnData,
    radius: float = 100.0,
    use_rep: Optional[str] = None,
    gamma: Optional[float] = None,
    n_iter: int = 200,
    eps: float = 0.0,
    k_min: int = 10,
    k_max: int = 30,
    gene_weight: float = 0.8,
    celltype_weight: float = 6.0,
    neighborhood_weight: float = 4.0,
    topology_weight: float = 0.0,
    boundary_weight: float = 0.0,
    anchor_weight: float = 0.35,
    anchor_spatial_blend: float = 0.75,
    compactness_weight: float = 2.0,
    compactness_quantile: float = 0.80,
    compactness_power: float = 1.75,
    strict_celltype_gate: float = 3.0,
    strict_neighborhood_gate: float = 0.9,
    strict_neighborhood_quantile: float = 0.65,
    strict_compactness_gate: float = 0.8,
    strict_compactness_quantile: float = 0.78,
    region_geometry_weight: float = 2.0,
    region_geometry_power: float = 1.25,
    shape_context_weight: float = 2.0,
    shape_context_power: float = 1.25,
    shape_k_neighbors: int = 24,
    geodesic_geometry_weight: float = 0.25,
    geodesic_max_cells: int = 2500,
    rho_min: float = 0.05,
    rho_max: float = 20.0,
    rho_overlap_power: float = 1.5,
    rho_scale: float = 1.60,
    overlap_rotation_samples: int = 24,
    target_mass_fraction: Optional[float] = 0.95,
    rho_retry_factor: float = 1.8,
    rho_retry_rounds: int = 3,
    confidence_power: float = 1.30,
    confidence_rounds: int = 2,
    support_row_ratio: float = 0.001,
    support_col_ratio: float = 0.001,
    support_min_mass: float = 0.0,
    memory_safe_auto: bool = True,
    memory_pair_limit: int = 80_000_000,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Align two MERFISH slices using Unbalanced Fused Gromov-Wasserstein (UFGW)
    with spectral symmetry breaking and community anchoring.

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Must contain:
          - ``.X``                   gene expression (dense or sparse)
          - ``.obsm['spatial']``     (N, 2) 2D coordinates
          - ``.obs['cell_type_annot']`` categorical cell-type labels
    radius    : neighbourhood radius for JSD computation (same units as coords)
    use_rep   : if not None, use ``sliceA.obsm[use_rep]`` for gene cost instead of .X
    gamma     : GW/feature balance ∈ [0,1]. None → auto-set to 0.35.
    n_iter    : max UFGW conditional-gradient iterations
    eps       : entropic regularisation for mm_unbalanced
    k_min/max : adaptive k-NN graph search range
    gene_weight, celltype_weight, neighborhood_weight, topology_weight,
    boundary_weight, anchor_weight : weights for core biological/topological
        costs in the fused linear objective
    anchor_spatial_blend : blend ratio of spatial community coupling in
        geometry-driving pi_comm (0=anchor only, 1=spatial only)
    compactness_weight   : weight of compactness cost in fused linear term
    compactness_quantile : robust scaling quantile for compactness distances
    compactness_power    : >1 sharpens compactness penalties
    strict_celltype_gate : additional unnormalised penalty on cell-type
        mismatched pairs (strongly enforces type-consistent transport)
    strict_neighborhood_gate : additional unnormalised penalty on poor
        neighborhood-compatibility pairs (promotes NSP preservation)
    strict_neighborhood_quantile : row-wise quantile threshold on
        M_neighborhood used to define poor neighborhood pairs
    strict_compactness_gate : additional unnormalised penalty on long-range
        compactness violations
    strict_compactness_quantile : quantile threshold on M_compact used to
        define long-range pairs for strict compactness gating
    region_geometry_weight : weight of bidirectional region-geometry cost
    region_geometry_power  : >1 sharpens region-geometry penalties
    shape_context_weight : weight of local shape-context cost
    shape_context_power  : >1 sharpens local shape-context penalties
    shape_k_neighbors    : neighborhood size for local shape-context descriptor
    geodesic_geometry_weight : blend weight for graph geodesic in GW geometry
    geodesic_max_cells   : skip geodesic all-pairs above this cell count
    rho_min/rho_max      : lower/upper bounds for KL mass penalty calibration
    rho_overlap_power    : overlap-to-rho nonlinearity (>1 tightens high-overlap)
    rho_scale            : global multiplier for calibrated rho
    overlap_rotation_samples : number of global rotations tested for overlap
    target_mass_fraction : optional minimum desired transported mass in [0, 1]
    rho_retry_factor     : multiplicative rho increase per retry if mass is low
    rho_retry_rounds     : number of retries for target_mass_fraction
    confidence_power     : >1 sharpens row/column conditionals (one-to-few)
    confidence_rounds    : alternating row/column sharpening rounds
    support_row_ratio    : prune entries below row-wise max ratio after
        confidence sharpening
    support_col_ratio    : prune entries below column-wise max ratio after
        confidence sharpening
    support_min_mass     : absolute floor pruning after confidence sharpening
    memory_safe_auto     : automatically disable GW term when N_A*N_B is very
        large to prevent out-of-memory crashes
    memory_pair_limit    : pair-count threshold used by memory_safe_auto
    verbose   : print solver progress

    Returns
    -------
    pi12                 : (N_A, N_B) transport plan
    initial_obj_neighbor : float — JSD metric with uniform plan (before)
    initial_obj_gene_cos : float — cosine metric with uniform plan (before)
    final_obj_neighbor   : float — JSD metric with pi12 (after)
    final_obj_gene_cos   : float — cosine metric with pi12 (after)
    """
    t0 = time.time()

    # ── 0. Sync shared genes & cell types ────────────────────────────────────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between the two slices.")
    sA = sliceA[:, shared_genes].copy()
    sB = sliceB[:, shared_genes].copy()

    N_A = sA.n_obs
    N_B = sB.n_obs

    if gamma is None:
        gamma_req = 0.35
    else:
        gamma_req = float(np.clip(gamma, 0.0, 1.0))

    pair_count = int(N_A) * int(N_B)
    disable_gw = bool(
        memory_safe_auto
        and pair_count > int(memory_pair_limit)
        and gamma_req > 0.0
    )

    cell_types_A = np.asarray(sA.obs['cell_type_annot'].values)
    cell_types_B = np.asarray(sB.obs['cell_type_annot'].values)

    coords_A = np.asarray(sA.obsm['spatial'], dtype=np.float64)
    coords_B = np.asarray(sB.obsm['spatial'], dtype=np.float64)

    if verbose:
        print(f"[STFA] Slices: {N_A} × {N_B} cells | {len(shared_genes)} shared genes")

    # ── 1. Build k-NN graphs & diffusion signatures ───────────────────────────
    if verbose:
        print("[STFA] Stage 1: Building graphs & diffusion signatures ...")
    adj_A, fiedler_A, tau_A = build_knn_graph(coords_A, k_min=k_min, k_max=k_max)
    adj_B, fiedler_B, tau_B = build_knn_graph(coords_B, k_min=k_min, k_max=k_max)

    H_A = compute_diffusion_signatures(adj_A, cell_types_A, tau_mix=tau_A)
    H_B = compute_diffusion_signatures(adj_B, cell_types_B, tau_mix=tau_B)

    if verbose:
        print(f"  Fiedler: A={fiedler_A:.4f} (τ={tau_A:.1f}), B={fiedler_B:.4f} (τ={tau_B:.1f})")

    # ── 2. Community anchors ──────────────────────────────────────────────────
    if verbose:
        print("[STFA] Stage 2: Community anchor matching ...")
    comm_A = detect_communities(adj_A)
    comm_B = detect_communities(adj_B)

    M_anchor, pi_comm_anchor = compute_M_anchor(
        comm_A, comm_B, H_A, H_B, cell_types_A, cell_types_B
    )
    pi_comm_spatial = _spatial_community_coupling(coords_A, coords_B, comm_A, comm_B)
    pi_comm_geom = _blend_couplings(
        pi_comm_anchor,
        pi_comm_spatial,
        spatial_blend=anchor_spatial_blend,
    )

    if verbose:
        n_cA = len(np.unique(comm_A))
        n_cB = len(np.unique(comm_B))
        print(f"  Communities: {n_cA} (A) × {n_cB} (B)")

    # ── 3. Assemble fused cost ────────────────────────────────────────────────
    if verbose:
        print("[STFA] Stage 3: Computing fused cost (gene, celltype, neighborhood, topo, boundary, anchor, compactness, region-geometry, shape-context) ...")
    zeros = np.zeros((N_A, N_B), dtype=np.float64)
    M_gene         = compute_M_gene(sA, sB, use_rep=use_rep)
    M_celltype = (
        compute_M_celltype(sA, sB)
        if celltype_weight > 0
        else zeros
    )
    M_neighborhood = (
        compute_M_neighborhood(sA, sB, radius=radius)
        if neighborhood_weight > 0
        else zeros
    )
    M_topo = (
        compute_M_topo(H_A, H_B)
        if topology_weight > 0
        else zeros
    )
    M_boundary = (
        compute_M_boundary(adj_A, adj_B)
        if boundary_weight > 0
        else zeros
    )
    M_shape = (
        compute_M_shape_context(
            coords_A,
            coords_B,
            k_neighbors=shape_k_neighbors,
            shape_power=shape_context_power,
        )
        if shape_context_weight > 0
        else zeros
    )
    M_compact      = compute_M_compact(
        coords_A,
        coords_B,
        comm_A,
        comm_B,
        pi_comm_geom,
        quantile=compactness_quantile,
        distance_power=compactness_power,
    )
    M_region_geom  = compute_M_region_geom(
        coords_A,
        coords_B,
        comm_A,
        comm_B,
        pi_comm_geom,
        shape_power=region_geometry_power,
    )

    M_fused        = fuse_costs(
        M_gene,
        M_celltype,
        M_neighborhood,
        M_topo,
        M_boundary,
        M_anchor,
        M_compact,
        M_region_geom,
        M_shape,
        weights=[
            float(max(0.0, gene_weight)),
            float(max(0.0, celltype_weight)),
            float(max(0.0, neighborhood_weight)),
            float(max(0.0, topology_weight)),
            float(max(0.0, boundary_weight)),
            float(max(0.0, anchor_weight)),
            float(max(0.0, compactness_weight)),
            float(max(0.0, region_geometry_weight)),
            float(max(0.0, shape_context_weight)),
        ],
    )

    # Hard gates are intentionally added AFTER normalized fusion.
    # They provide strict constraints that cannot be diluted by normalization.
    if strict_celltype_gate > 0 and celltype_weight > 0:
        M_fused = M_fused + float(strict_celltype_gate) * M_celltype

    if strict_neighborhood_gate > 0 and neighborhood_weight > 0:
        q_nei = float(np.clip(strict_neighborhood_quantile, 0.50, 0.99))
        cut_n = np.quantile(M_neighborhood, q_nei, axis=1, keepdims=True)
        neigh_hard = (M_neighborhood > cut_n).astype(np.float64)
        M_fused = M_fused + float(strict_neighborhood_gate) * neigh_hard

    if strict_compactness_gate > 0:
        q_gate = float(np.clip(strict_compactness_quantile, 0.50, 0.99))
        cut = np.quantile(M_compact, q_gate, axis=1, keepdims=True)
        compact_hard = (M_compact > cut).astype(np.float64)
        M_fused = M_fused + float(strict_compactness_gate) * compact_hard

    # ── 4. Geometry matrices (optional in memory-safe mode) ─────────────────
    if disable_gw or gamma_req <= 1e-12:
        C_A = None
        C_B = None
        if verbose and disable_gw:
            print(
                "[STFA] Memory-safe mode: disabling GW geometry "
                f"for pair_count={pair_count:,} (> {int(memory_pair_limit):,})."
            )
    else:
        C_A = _build_geometry_matrix(
            coords_A,
            adj_A,
            geodesic_weight=geodesic_geometry_weight,
            geodesic_max_cells=geodesic_max_cells,
        )
        C_B = _build_geometry_matrix(
            coords_B,
            adj_B,
            geodesic_weight=geodesic_geometry_weight,
            geodesic_max_cells=geodesic_max_cells,
        )

    # ── 5. Calibrate rho and gamma ────────────────────────────────────────────
    f_ovlp = estimate_overlap_fraction(
        coords_A,
        coords_B,
        n_rotations=overlap_rotation_samples,
    )
    rho = calibrate_rho(
        f_ovlp,
        rho_min=rho_min,
        rho_max=rho_max,
        overlap_power=rho_overlap_power,
        rho_scale=rho_scale,
    )
    gamma = 0.0 if disable_gw else gamma_req
    p_A = np.ones(N_A) / N_A
    p_B = np.ones(N_B) / N_B

    if verbose:
        print(f"  Overlap≈{f_ovlp:.2%}  ρ={rho:.4f}  γ={gamma:.2f}")

    # ── 6. Initial objective (uniform plan as baseline) ───────────────────────
    pi_uniform = np.ones((N_A, N_B)) / (N_A * N_B)

    # JSD neighbourhood cost
    nd_A = neighborhood_distribution(sA, radius=radius) + 0.01
    nd_B = neighborhood_distribution(sB, radius=radius) + 0.01
    jsd_neighborhood = np.asarray(
        jensenshannon_divergence_backend(nd_A, nd_B),
        dtype=np.float64,
    )
    initial_obj_neighbor, initial_obj_gene_cos = _compute_objectives(
        pi_uniform, jsd_neighborhood, M_gene
    )

    # ── 7. Solve UFGW ─────────────────────────────────────────────────────────
    if verbose:
        print("[STFA] Stage 4: Solving UFGW ...")
    pi12 = solve_ufgw(
        p_A, p_B, M_fused, C_A, C_B,
        rho=rho, gamma=gamma, eps=eps,
        n_iter=n_iter,
        confidence_power=confidence_power,
        confidence_rounds=confidence_rounds,
        support_row_ratio=support_row_ratio,
        support_col_ratio=support_col_ratio,
        support_min_mass=support_min_mass,
        verbose=verbose,
    )

    if target_mass_fraction is not None:
        target = float(np.clip(target_mass_fraction, 0.0, 1.0))
        best_pi = np.asarray(pi12, dtype=np.float64)
        best_mass = float(best_pi.sum())
        rho_cur = float(rho)
        retries = int(max(0, rho_retry_rounds))
        rho_mult = float(max(1.1, rho_retry_factor))

        for rr in range(retries):
            if best_mass >= target or rho_cur >= rho_max - 1e-12:
                break

            rho_cur = min(float(rho_max), rho_cur * rho_mult)
            if verbose:
                print(
                    f"  [STFA] mass retry {rr + 1}/{retries}: "
                    f"ρ={rho_cur:.4f}, current_mass={best_mass:.4f}, target={target:.4f}"
                )

            pi_try = solve_ufgw(
                p_A, p_B, M_fused, C_A, C_B,
                rho=rho_cur, gamma=gamma, eps=eps,
                n_iter=n_iter,
                confidence_power=confidence_power,
                confidence_rounds=confidence_rounds,
                support_row_ratio=support_row_ratio,
                support_col_ratio=support_col_ratio,
                support_min_mass=support_min_mass,
                verbose=False,
            )
            mass_try = float(pi_try.sum())
            if mass_try > best_mass:
                best_mass = mass_try
                best_pi = np.asarray(pi_try, dtype=np.float64)

        pi12 = best_pi
        if verbose:
            print(f"  [STFA] final transported mass: {float(pi12.sum()):.4f}")

    # ── 8. Final objectives ───────────────────────────────────────────────────
    # Normalise pi12 so it sums to 1 (probability plan for metric computation)
    pi12_sum = pi12.sum()
    pi12_norm = pi12 / (pi12_sum + 1e-12)

    final_obj_neighbor, final_obj_gene_cos = _compute_objectives(
        pi12_norm, jsd_neighborhood, M_gene
    )

    if verbose:
        elapsed = time.time() - t0
        print(f"[STFA] Done in {elapsed:.1f}s")
        print(f"  JSD:    {initial_obj_neighbor:.6f} → {final_obj_neighbor:.6f}")
        print(f"  Cosine: {initial_obj_gene_cos:.6f} → {final_obj_gene_cos:.6f}")

    return pi12, initial_obj_neighbor, initial_obj_gene_cos, final_obj_neighbor, final_obj_gene_cos
