"""
stfa/utils.py
=============
Shared utility functions: JSD backend, neighbourhood distribution,
improved Generalized Procrustes Analysis, slice stacking, and the
three-panel unbalanced-alignment visualiser.

Ported and improved from old/utils.py and old/INCENT.py.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import ot
import torch
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

to_dense_array = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.obsm[rep]


# ──────────────────────────────────────────────────────────────────────────────
# JSD backend  (unchanged from old/utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def _kl_div_backend(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise KL(X‖Y) for matching rows.  Returns 1-D array of length n."""
    nx = ot.backend.get_backend(X, Y)
    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i', X, log_X).reshape(1, -1)
    X_log_Y = nx.einsum('ij,ij->i', X, log_Y).reshape(1, -1)
    D = X_log_X.T - X_log_Y.T
    return nx.to_numpy(D)


def _jsd_1_vs_many(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """JSD distance between a single row X[0] and all rows of Y."""
    assert X.shape[0] == 1
    nx = ot.backend.get_backend(X, Y)
    X = nx.concatenate([X] * Y.shape[0], axis=0)
    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    M = (X + Y) / 2.0
    kl_xm = torch.from_numpy(_kl_div_backend(
        nx.to_numpy(X) if not isinstance(X, np.ndarray) else X,
        nx.to_numpy(M) if not isinstance(M, np.ndarray) else M,
    ))
    kl_ym = torch.from_numpy(_kl_div_backend(
        nx.to_numpy(Y) if not isinstance(Y, np.ndarray) else Y,
        nx.to_numpy(M) if not isinstance(M, np.ndarray) else M,
    ))
    return nx.sqrt((kl_xm + kl_ym) / 2.0).T[0]


def jensenshannon_divergence_backend(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise JSD divergence matrix  (N×M) for probability row-vectors."""
    assert X.shape[1] == Y.shape[1]
    nx = ot.backend.get_backend(X, Y)
    n = X.shape[0]
    js = nx.zeros((n, Y.shape[0]))
    for i in tqdm(range(n), desc="JSD matrix", leave=False):
        js[i, :] = _jsd_1_vs_many(X[i : i + 1], Y)
    if torch.cuda.is_available():
        try:
            return js.numpy()
        except Exception:
            return js
    return js


# ──────────────────────────────────────────────────────────────────────────────
# Neighbourhood distribution  (vectorised for speed vs. old O(N²) loop)
# ──────────────────────────────────────────────────────────────────────────────

def neighborhood_distribution(curr_slice: AnnData, radius: float) -> np.ndarray:
    """
    Compute per-cell cell-type composition within *radius* Euclidean distance.

    Returns
    -------
    ndarray of shape (N_cells, N_cell_types)
    """
    unique_types = np.array(list(curr_slice.obs['cell_type_annot'].unique()))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    labels = curr_slice.obs['cell_type_annot'].values
    coords = curr_slice.obsm['spatial']

    dist = euclidean_distances(coords, coords)            # N×N
    within = dist <= radius                               # bool mask N×N

    out = np.zeros((curr_slice.n_obs, len(unique_types)), dtype=float)
    for j, lbl in enumerate(labels):
        out[:, type_to_idx[lbl]] += within[:, j].astype(float)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Improved Generalized Procrustes Analysis
# ──────────────────────────────────────────────────────────────────────────────

def generalized_procrustes_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    output_params: bool = False,
    matrix: bool = False,
    min_mass_threshold: float = 0.05,
    allow_reflection: bool = False,
) -> tuple:
    """
    Mass-weighted GPA with reflection guard and explicit scale locking.

    Parameters
    ----------
    X, Y              : (N, 2) coordinate arrays — source and target
    pi                : (N_A, N_B) transport plan
    min_mass_threshold: Cells with relative row/col mass below this are
                        down-weighted (handles unmatched cells gracefully).
    allow_reflection  : If False (default), forces det(R)=+1 (rotation only)
                        and emits a warning if a reflection would be better.

    Returns
    -------
    X_aligned, Y_aligned  (always)
    + (theta, tX, tY) or (R, tX, tY) if output_params=True
    """
    assert X.shape[1] == 2 and Y.shape[1] == 2
    pi_mass = pi.sum()
    if pi_mass == 0:
        raise ValueError("Transport plan mass is zero — cannot align.")

    # ── Mass-weighted centroids (unmatched cells have low row/col mass) ──────
    row_mass = pi.sum(axis=1)          # (N_A,)
    col_mass = pi.sum(axis=0)          # (N_B,)
    mean_row = row_mass.mean() + 1e-12
    mean_col = col_mass.mean() + 1e-12

    w_X = np.clip(row_mass / mean_row, min_mass_threshold, None)
    w_Y = np.clip(col_mass / mean_col, min_mass_threshold, None)
    w_X /= w_X.sum()
    w_Y /= w_Y.sum()

    tX = w_X @ X
    tY = w_Y @ Y

    X_c = X - tX
    Y_c = Y - tY

    # ── Scale-locked SVD rotation ────────────────────────────────────────────
    H = Y_c.T @ ((pi / pi_mass).T @ X_c)
    U, S, Vt = np.linalg.svd(H)

    det_sign = np.linalg.det(Vt.T @ U.T)
    is_reflection = det_sign < 0

    if not allow_reflection and is_reflection:
        U[:, -1] *= -1          # flip last column to enforce proper rotation
        warnings.warn(
            "GPA: detected improper rotation (reflection). "
            "This may mean the slice is flipped. "
            "Pass allow_reflection=True to allow reflections.",
            stacklevel=2,
        )

    R = Vt.T @ U.T
    X_aligned = (R @ X_c.T).T + tY
    Y_aligned = Y_c + tY

    if not output_params:
        return X_aligned, Y_aligned

    if matrix:
        return X_aligned, Y_aligned, R, tX, tY
    else:
        M_rot = np.array([[0.0, -1.0], [1.0, 0.0]])
        theta = np.arctan2(np.trace(M_rot @ H), np.trace(H))
        return X_aligned, Y_aligned, theta, tX, tY


# ──────────────────────────────────────────────────────────────────────────────
# Stack slices (same API as old code)
# ──────────────────────────────────────────────────────────────────────────────

def stack_slices_pairwise(
    slices: List[AnnData],
    pis: List[np.ndarray],
    output_params: bool = False,
    matrix: bool = False,
) -> Union[List[AnnData], tuple]:
    assert len(slices) == len(pis) + 1 and len(slices) > 1
    new_coor, thetas, translations = [], [], []

    if output_params:
        S1, S2, theta, tX, tY = generalized_procrustes_analysis(
            slices[0].obsm['spatial'], slices[1].obsm['spatial'],
            pis[0], output_params=True, matrix=matrix,
        )
        thetas.append(theta)
        translations.extend([tX, tY])
    else:
        S1, S2 = generalized_procrustes_analysis(
            slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0]
        )
    new_coor.extend([S1, S2])

    for i in range(1, len(slices) - 1):
        if output_params:
            _, y, theta, tX, tY = generalized_procrustes_analysis(
                new_coor[i], slices[i + 1].obsm['spatial'],
                pis[i], output_params=True, matrix=matrix,
            )
            thetas.append(theta)
            translations.append(tY)
        else:
            _, y = generalized_procrustes_analysis(
                new_coor[i], slices[i + 1].obsm['spatial'], pis[i]
            )
        new_coor.append(y)

    new_slices = []
    for i, s in enumerate(slices):
        sc = s.copy()
        sc.obsm['spatial'] = new_coor[i]
        new_slices.append(sc)

    if output_params:
        return new_slices, thetas, translations
    return new_slices


# ──────────────────────────────────────────────────────────────────────────────
# Three-panel unbalanced alignment visualiser
# ──────────────────────────────────────────────────────────────────────────────

def visualize_alignment_unbalanced(
    sliceA: AnnData,
    sliceB: AnnData,
    pi12: np.ndarray,
    n_arrows: int = 300,
    figsize: tuple = (18, 6),
    title: str = "STFA Unbalanced Alignment",
) -> List[AnnData]:
    """
    Three-panel visualisation for unbalanced alignments.

    Panel 1 — Aligned coordinates coloured by per-cell match confidence.
    Panel 2 — Source cells coloured by local unmatched mass D_local(i).
    Panel 3 — Sampled high-mass transport arrows.

    Returns
    -------
    new_slices : aligned AnnData list (same as stack_slices_pairwise output)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    new_slices = stack_slices_pairwise([sliceA, sliceB], [pi12])
    s1 = new_slices[0].obsm['spatial']
    s2 = new_slices[1].obsm['spatial']

    N_A = pi12.shape[0]
    p_A = np.ones(N_A) / N_A
    row_mass = pi12.sum(axis=1)
    match_conf = np.clip(row_mass / (p_A + 1e-12), 0, 1)
    D_local = np.clip(p_A - row_mass, 0, None)
    D_local_norm = D_local / (D_local.max() + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')

    # Panel 1 — match confidence
    ax = axes[0]
    sc1 = ax.scatter(s1[:, 0], s1[:, 1], s=2, c=match_conf,
                     cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
    ax.scatter(s2[:, 0], s2[:, 1], s=2, c='#60a5fa', alpha=0.4)
    cbar = plt.colorbar(sc1, ax=ax, shrink=0.6)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.set_label('Match confidence', color='white')
    ax.set_title('Alignment (source confidence)', color='white', fontsize=11)
    ax.axis('off')

    # Panel 2 — unmatched mass
    ax = axes[1]
    sc2 = ax.scatter(s1[:, 0], s1[:, 1], s=2, c=D_local_norm,
                     cmap='hot_r', alpha=0.9)
    cbar2 = plt.colorbar(sc2, ax=ax, shrink=0.6)
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.set_label('D_local (unmatched mass)', color='white')
    ax.set_title('Unmatched mass map (source)', color='white', fontsize=11)
    ax.axis('off')

    # Panel 3 — transport arrows
    ax = axes[2]
    ax.scatter(s1[:, 0], s1[:, 1], s=1, c='#f87171', alpha=0.35, label='Source')
    ax.scatter(s2[:, 0], s2[:, 1], s=1, c='#60a5fa', alpha=0.35, label='Target')
    flat_idx = np.argsort(pi12.ravel())[-n_arrows:]
    src_idx, tgt_idx = np.unravel_index(flat_idx, pi12.shape)
    max_pi = pi12.max() + 1e-12
    for si, ti in zip(src_idx, tgt_idx):
        alpha_val = float(pi12[si, ti] / max_pi) * 0.55 + 0.1
        ax.plot([s1[si, 0], s2[ti, 0]], [s1[si, 1], s2[ti, 1]],
                color='#fbbf24', alpha=alpha_val, linewidth=0.4)
    ax.legend(markerscale=5, fontsize=8, facecolor='#16213e',
              labelcolor='white', framealpha=0.6)
    ax.set_title(f'Transport links (top {n_arrows})', color='white', fontsize=11)
    ax.axis('off')

    plt.suptitle(title, color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
    return new_slices


# ──────────────────────────────────────────────────────────────────────────────
# Simple 2-panel visualiser (backward-compatible with old visualize_alignment)
# ──────────────────────────────────────────────────────────────────────────────

def visualize_alignment(sliceA: AnnData, sliceB: AnnData, pi12: np.ndarray) -> List[AnnData]:
    """Simple 2-panel scatter, API-compatible with old visualize_alignment."""
    import matplotlib.pyplot as plt
    new_slices = stack_slices_pairwise([sliceA, sliceB], [pi12])
    plt.figure(figsize=(8, 6))
    plt.scatter(new_slices[0].obsm['spatial'][:, 0],
                new_slices[0].obsm['spatial'][:, 1],
                s=1, alpha=0.5, c='#e41a1c', label='Source')
    plt.scatter(new_slices[1].obsm['spatial'][:, 0],
                new_slices[1].obsm['spatial'][:, 1],
                s=1, alpha=0.5, c='#377eb8', label='Target')
    plt.axis('off')
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.show()
    return new_slices
