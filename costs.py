"""
stfa/costs.py
=============
All pairwise cost matrix computations for the STFA pipeline:
  - M_gene    : cosine distance on (optionally latent) gene expression
  - M_topo    : cosine distance on multi-scale diffusion signatures
  - M_boundary: soft boundary-uncertainty penalty
  - M_anchor  : community-level OT soft-gating cost
  - fuse_costs: unit-Frobenius normalised sum
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from anndata import AnnData
from typing import Optional

from stfa.utils import jensenshannon_divergence_backend, neighborhood_distribution


# ──────────────────────────────────────────────────────────────────────────────
# M_gene  — gene expression cosine distance
# ──────────────────────────────────────────────────────────────────────────────

def compute_M_gene(
    sliceA: AnnData,
    sliceB: AnnData,
    use_rep: Optional[str] = None,
) -> np.ndarray:
    """
    Cosine distance between per-cell gene-expression vectors (or latent
    embeddings if use_rep points to obsm key).

    Returns
    -------
    M : (N_A, N_B) float64 array  in [0, 2]
    """
    def _get(s, rep):
        if rep is None:
            X = s.X
        else:
            X = s.obsm[rep]
        return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)

    A = _get(sliceA, use_rep) + 1e-8   # avoid zero-vector cosine instability
    B = _get(sliceB, use_rep) + 1e-8
    return cosine_distances(A, B).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# M_celltype  — cell-type mismatch penalty
# ──────────────────────────────────────────────────────────────────────────────

def compute_M_celltype(
    sliceA: AnnData,
    sliceB: AnnData,
) -> np.ndarray:
    """
    Binary cost matrix where M_celltype(i,j) = 1 if cell types differ, else 0.

    Returns
    -------
    M : (N_A, N_B) float64 array with values in {0, 1}
    """
    ct_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    ct_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M = (ct_A[:, None] != ct_B[None, :]).astype(np.float64)
    return M


# ──────────────────────────────────────────────────────────────────────────────
# M_neighborhood — local neighborhood distribution divergence
# ──────────────────────────────────────────────────────────────────────────────

def compute_M_neighborhood(
    sliceA: AnnData,
    sliceB: AnnData,
    radius: float = 100.0,
) -> np.ndarray:
    """
    Jensen-Shannon divergence between local neighborhood distributions.

    For each cell, compute the distribution of cell types within a fixed
    radius in spatial coordinates. Then compute the JSD between these
    distributions for all pairs of cells across slices.

    Returns
    -------
    M : (N_A, N_B) float64 array with values in [0, 1]
    """
    dist_A = neighborhood_distribution(sliceA, radius) + 1e-6
    dist_B = neighborhood_distribution(sliceB, radius) + 1e-6
    M = jensenshannon_divergence_backend(dist_A, dist_B)
    return M.astype(np.float64)

# ──────────────────────────────────────────────────────────────────────────────
# M_topo  — topology via multi-scale diffusion signatures
# ──────────────────────────────────────────────────────────────────────────────

def compute_M_topo(
    H_A: np.ndarray,
    H_B: np.ndarray,
) -> np.ndarray:
    """
    Cosine distance between pre-computed diffusion signature matrices.

    Parameters
    ----------
    H_A : (N_A, D) diffusion signatures for slice A
    H_B : (N_B, D) diffusion signatures for slice B

    Returns
    -------
    M : (N_A, N_B) float64 array  in [0, 2]
    """
    return cosine_distances(H_A, H_B).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# M_boundary — smooth geodesic boundary uncertainty
# ──────────────────────────────────────────────────────────────────────────────

def _boundary_weights(adj: sp.csr_matrix) -> np.ndarray:
    """
    Per-cell boundary confidence  u_i ∈ [0.5, 1].

    u_i = 1 − 0.5·exp(−d_i / σ_d)

    where d_i is the geodesic distance from cell i to the graph periphery
    (peripheral = degree < median degree) and σ_d is the 5th percentile
    of all such geodesic distances.

    High u_i → interior cell (far from border) → reliable anchor.
    Low  u_i → peripheral cell → uncertain.
    """
    if adj is None:
        raise ValueError("adj must be a valid sparse adjacency matrix")

    adj_csr = sp.csr_matrix(adj)

    shape = adj_csr.shape
    if shape is None:
        raise ValueError("adjacency matrix must define shape")
    n = int(shape[0])
    deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    median_deg = np.median(deg)
    peripheral = np.where(deg < median_deg)[0]

    if len(peripheral) == 0:
        return np.ones(n, dtype=np.float64)

    dist_mat = csgraph.shortest_path(adj_csr, directed=False, indices=peripheral,
                                     unweighted=True)   # (|peripheral|, N)
    d_to_border = dist_mat.min(axis=0)                  # (N,)
    d_to_border = np.where(np.isinf(d_to_border), d_to_border[~np.isinf(d_to_border)].max() + 1,
                           d_to_border)

    sigma = max(1.0, float(np.percentile(d_to_border, 5)))
    u = 1.0 - 0.5 * np.exp(-d_to_border / sigma)
    return u.astype(np.float64)


def compute_M_boundary(
    adj_A: sp.csr_matrix,
    adj_B: sp.csr_matrix,
) -> np.ndarray:
    """
    M_boundary(i,j) = 1 − u_i · u_j

    Cells near tissue borders have low u → high cost when paired together
    (penalises uncertain border-to-border matches).
    """
    u_A = _boundary_weights(adj_A)    # (N_A,)
    u_B = _boundary_weights(adj_B)    # (N_B,)
    M = 1.0 - np.outer(u_A, u_B)     # (N_A, N_B)
    return M.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# M_compact — coarse rigidly aligned cross-slice spatial compactness
# ──────────────────────────────────────────────────────────────────────────────

def _fit_weighted_rigid_transform(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    coupling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a weighted rigid transform (rotation + translation) from src_pts to
    tgt_pts using coupling weights.

    Returns
    -------
    R      : (2, 2) rotation matrix (det = +1)
    t_src  : (2,) source weighted centroid
    t_tgt  : (2,) target weighted centroid
    """
    mass = float(coupling.sum())
    if mass <= 0:
        return np.eye(2, dtype=np.float64), np.zeros(2), np.zeros(2)

    w_src = coupling.sum(axis=1)
    w_tgt = coupling.sum(axis=0)

    t_src = (w_src[:, None] * src_pts).sum(axis=0) / (w_src.sum() + 1e-12)
    t_tgt = (w_tgt[:, None] * tgt_pts).sum(axis=0) / (w_tgt.sum() + 1e-12)

    src_c = src_pts - t_src
    tgt_c = tgt_pts - t_tgt

    # Weighted Procrustes cross-covariance.
    H = tgt_c.T @ ((coupling / (mass + 1e-12)).T @ src_c)
    U, _, Vt = np.linalg.svd(H, full_matrices=False)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    return R.astype(np.float64), t_src.astype(np.float64), t_tgt.astype(np.float64)


def compute_M_compact(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    comm_A: np.ndarray,
    comm_B: np.ndarray,
    pi_comm: np.ndarray,
    quantile: float = 0.90,
    distance_power: float = 1.25,
) -> np.ndarray:
    """
    Compactness-promoting cross-slice spatial cost.

    1) Build community centroids for both slices.
    2) Estimate a coarse rigid transform using community transport pi_comm.
    3) Transform source coordinates and compute cross-slice distances.
    4) Robustly normalise by a quantile and clip to [0, 1].

    This term discourages mapping a single local source region into multiple
    distant target islands while still allowing unmatched mass through UOT.
    """
    q = float(np.clip(quantile, 0.50, 0.99))
    pwr = float(max(1.0, distance_power))

    unique_A = np.unique(comm_A)
    unique_B = np.unique(comm_B)

    cent_A = np.vstack([coords_A[comm_A == c].mean(axis=0) for c in unique_A])
    cent_B = np.vstack([coords_B[comm_B == c].mean(axis=0) for c in unique_B])

    # Fall back to identity-like behavior if any numerical issue occurs.
    try:
        R, t_src, t_tgt = _fit_weighted_rigid_transform(cent_A, cent_B, pi_comm)
        coords_A_coarse = (R @ (coords_A - t_src).T).T + t_tgt
    except Exception:
        coords_A_coarse = np.asarray(coords_A, dtype=np.float64)

    D = cdist(coords_A_coarse, coords_B)
    scale = float(np.quantile(D, q))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(D.max())
    if scale <= 0:
        return np.zeros_like(D, dtype=np.float64)

    M = np.clip(D / (scale + 1e-12), 0.0, 1.0)
    if pwr > 1.0:
        M = M ** pwr
    return M.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# M_anchor — community-level OT soft-gating
# ──────────────────────────────────────────────────────────────────────────────

def _community_descriptor(
    h: np.ndarray,
    comm_labels: np.ndarray,
    cell_types: np.ndarray,
    n_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each community compute:
      - mean diffusion signature
      - cell-type composition vector
      - [community size fraction, boundary exposure fraction] (2 scalars)

    Returns
    -------
    desc  : (n_comm, D_desc) descriptor matrix per community
    sizes : (n_comm,) number of cells per community
    """
    unique_comms = np.unique(comm_labels)
    n_comm = len(unique_comms)
    comm_idx = {c: i for i, c in enumerate(unique_comms)}

    unique_types = np.unique(cell_types)
    n_types = len(unique_types)
    type_idx = {t: i for i, t in enumerate(unique_types)}

    D_sig = h.shape[1]
    descs = []
    sizes = []

    for c in unique_comms:
        mask = comm_labels == c
        nh = h[mask].mean(axis=0)                          # mean diffusion sig

        # cell-type composition
        ct = np.zeros(n_types)
        for lbl in cell_types[mask]:
            ct[type_idx[lbl]] += 1
        ct /= ct.sum() + 1e-12

        # macroscopic scalars
        size_frac = mask.sum() / n_total
        # boundary exposure: fraction of community cells with low mean sig norm
        norms = np.linalg.norm(h[mask], axis=1)
        bnd_exp = (norms < np.median(norms)).mean()

        desc = np.concatenate([nh, ct, [size_frac, bnd_exp]])
        descs.append(desc)
        sizes.append(mask.sum())

    return np.vstack(descs), np.array(sizes, dtype=float)


def compute_M_anchor(
    comm_A: np.ndarray,
    comm_B: np.ndarray,
    H_A: np.ndarray,
    H_B: np.ndarray,
    cell_types_A: np.ndarray,
    cell_types_B: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a small partial OT problem on community descriptors, then
    build a soft-gating cost for cell-level matching.

    Returns
    -------
    M_anchor       : (N_A, N_B) soft-gating cost matrix
    pi_comm        : (n_comm_A, n_comm_B) community transport plan
    """
    import importlib

    _ot = importlib.import_module("ot")

    desc_A, sizes_A = _community_descriptor(H_A, comm_A, cell_types_A, len(comm_A))
    desc_B, sizes_B = _community_descriptor(H_B, comm_B, cell_types_B, len(comm_B))

    # Normalise descriptor columns to [0,1] for stable OT
    desc_all = np.vstack([desc_A, desc_B])
    col_min = desc_all.min(axis=0)
    col_max = desc_all.max(axis=0)
    col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    desc_A = (desc_A - col_min) / col_range
    desc_B = (desc_B - col_min) / col_range

    # Cost between communities (L2 on normalised descriptors)
    from sklearn.metrics.pairwise import euclidean_distances as ed
    M_comm = ed(desc_A, desc_B)
    M_comm = M_comm / (M_comm.max() + 1e-12)

    # Uniform weights over communities
    n_cA = len(desc_A)
    n_cB = len(desc_B)
    p_cA = np.ones(n_cA) / n_cA
    p_cB = np.ones(n_cB) / n_cB

    try:
        pi_comm = _ot.emd(p_cA, p_cB, M_comm)
    except Exception:
        pi_comm = np.outer(p_cA, p_cB)     # fallback: independent coupling

    # Lift to cell level: M_anchor(i,j) = -log(pi_comm(comm_A[i], comm_B[j]) + ε)
    unique_A = np.unique(comm_A)
    unique_B = np.unique(comm_B)
    cA_to_idx = {c: i for i, c in enumerate(unique_A)}
    cB_to_idx = {c: i for i, c in enumerate(unique_B)}

    idx_A = np.array([cA_to_idx[c] for c in comm_A])   # (N_A,)
    idx_B = np.array([cB_to_idx[c] for c in comm_B])   # (N_B,)

    # Vectorised lookup: M_anchor[i,j] value from pi_comm[idx_A[i], idx_B[j]]
    pi_lifted = pi_comm[np.ix_(idx_A, idx_B)]           # (N_A, N_B)
    M_anchor = -np.log(pi_lifted + epsilon)
    M_anchor = M_anchor.astype(np.float64)

    return M_anchor, pi_comm


# ──────────────────────────────────────────────────────────────────────────────
# Fused cost: normalise and sum
# ──────────────────────────────────────────────────────────────────────────────

def fuse_costs(*matrices: np.ndarray, weights: Optional[list[float]] = None) -> np.ndarray:
    """
    Return the unit-Frobenius-norm–normalised sum of all input matrices.
    By default each term contributes equally regardless of raw scale. Use
    ``weights`` to up/down weight specific terms after normalisation.
    """
    if len(matrices) == 0:
        raise ValueError("fuse_costs requires at least one matrix")

    if weights is None:
        weights_arr = np.ones(len(matrices), dtype=np.float64)
    else:
        if len(weights) != len(matrices):
            raise ValueError("weights must have the same length as matrices")
        weights_arr = np.asarray(weights, dtype=np.float64)

    total = np.zeros_like(matrices[0], dtype=np.float64)
    for w, M in zip(weights_arr, matrices):
        if w <= 0:
            continue
        fro = np.linalg.norm(M, 'fro')
        total += float(w) * (M / (fro + 1e-12))
    return total
