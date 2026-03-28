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
from anndata import AnnData
from sklearn.metrics.pairwise import cosine_distances
from typing import Optional, Tuple

from .graph import build_knn_graph, compute_diffusion_signatures, detect_communities
from .costs import compute_M_gene, compute_M_topo, compute_M_boundary, compute_M_anchor, fuse_costs
from .solver import estimate_overlap_fraction, calibrate_rho, solve_ufgw, _norm_dist
from .utils import neighborhood_distribution, jensenshannon_divergence_backend


# ──────────────────────────────────────────────────────────────────────────────
# Internal metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_objectives(
    pi_mat: np.ndarray,
    jsd_neighborhood: np.ndarray,
    cosine_gene: np.ndarray,
) -> Tuple[float, float]:
    """Weighted-sum objectives from a transport plan and pre-computed cost matrices."""
    obj_neighbor = float(np.sum(jsd_neighborhood * pi_mat))
    obj_gene_cos = float(np.sum(cosine_gene * pi_mat))
    return obj_neighbor, obj_gene_cos


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
    eps: float = 0.05,
    k_min: int = 10,
    k_max: int = 30,
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
    gamma     : GW/feature balance ∈ [0,1]. None → auto-set to 0.5.
    n_iter    : max UFGW conditional-gradient iterations
    eps       : entropic regularisation for mm_unbalanced
    k_min/max : adaptive k-NN graph search range
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

    cell_types_A = np.asarray(sA.obs['cell_type_annot'].values)
    cell_types_B = np.asarray(sB.obs['cell_type_annot'].values)

    coords_A = sA.obsm['spatial']
    coords_B = sB.obsm['spatial']

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

    M_anchor, pi_comm = compute_M_anchor(
        comm_A, comm_B, H_A, H_B, cell_types_A, cell_types_B
    )

    if verbose:
        n_cA = len(np.unique(comm_A))
        n_cB = len(np.unique(comm_B))
        print(f"  Communities: {n_cA} (A) × {n_cB} (B)")

    # ── 3. Assemble fused cost ────────────────────────────────────────────────
    if verbose:
        print("[STFA] Stage 3: Computing fused cost (gene, topo, boundary, anchor) ...")
    M_gene     = compute_M_gene(sA, sB, use_rep=use_rep)
    M_topo     = compute_M_topo(H_A, H_B)
    M_boundary = compute_M_boundary(adj_A, adj_B)
    M_fused    = fuse_costs(M_gene, M_topo, M_boundary, M_anchor)

    # ── 4. Geometry matrices ──────────────────────────────────────────────────
    from scipy.spatial.distance import cdist
    C_A = _norm_dist(cdist(coords_A, coords_A))
    C_B = _norm_dist(cdist(coords_B, coords_B))

    # ── 5. Calibrate rho and gamma ────────────────────────────────────────────
    f_ovlp = estimate_overlap_fraction(coords_A, coords_B)
    rho    = calibrate_rho(f_ovlp)
    if gamma is None:
        gamma = 0.5
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
        n_iter=n_iter, verbose=verbose,
    )

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
