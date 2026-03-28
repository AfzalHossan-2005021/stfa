"""
stfa/graph.py
=============
Graph construction, spectral diffusion signatures,
and community detection for the STFA pipeline.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelBinarizer
from anndata import AnnData
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive k-NN graph
# ──────────────────────────────────────────────────────────────────────────────

def build_knn_graph(
    coords: np.ndarray,
    k_min: int = 10,
    k_max: int = 30,
    coverage_thresh: float = 0.95,
) -> Tuple[sp.csr_matrix, float, float]:
    """
    Build the smallest k-NN graph whose dominant connected component
    covers ≥ coverage_thresh fraction of all cells.

    Returns
    -------
    adj       : symmetric binary adjacency (CSR)
    fiedler   : λ₂ of the normalised Laplacian (0 if disconnected)
    tau_mix   : 1 / fiedler  (spectral heuristic mixing timescale)
    """
    n = coords.shape[0]
    best_adj = None
    best_k = k_min

    for k in range(k_min, k_max + 1):
        k_eff = min(k, n - 1)
        A = kneighbors_graph(coords, n_neighbors=k_eff, mode='connectivity',
                             include_self=False)
        A = (A + A.T)                      # symmetrise
        A.data[:] = 1.0                    # binary

        # Connected component sizes
        n_comp, labels = sp.csgraph.connected_components(A, directed=False)
        largest = np.bincount(labels).max()
        if largest / n >= coverage_thresh:
            best_adj = A
            best_k = k
            break
        best_adj = A   # keep last even if threshold not met

    # Normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}
    deg = np.asarray(best_adj.sum(axis=1)).ravel()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L_norm = sp.eye(n, format='csr') - D_inv_sqrt @ best_adj @ D_inv_sqrt

    # Fiedler value  (2nd smallest eigenvalue of L_norm)
    try:
        # k=2 gives the two smallest eigenvalues; tol for speed
        vals = spla.eigsh(L_norm, k=2, which='SM', tol=1e-3,
                          return_eigenvectors=False)
        vals = np.sort(np.abs(vals))      # guarantee ascending
        fiedler = float(vals[1]) if len(vals) > 1 else 1e-3
    except Exception:
        fiedler = 1e-3                    # fallback

    fiedler = max(fiedler, 1e-6)
    tau_mix = 1.0 / fiedler
    return best_adj.tocsr(), fiedler, tau_mix


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scale diffusion signatures on categorical cell types
# ──────────────────────────────────────────────────────────────────────────────

def compute_diffusion_signatures(
    adj: sp.csr_matrix,
    cell_types: np.ndarray,
    tau_mix: float,
    n_power_iter: int = 50,
) -> np.ndarray:
    """
    Compute multi-scale heat-diffusion signatures  h_i(t) = P^t X
    at three timescales: t ∈ {1, τ/4, τ}.

    Features are based on CATEGORICAL cell-type one-hot vectors only
    (not gene expression) to prevent circular coupling with M_gene.

    Parameters
    ----------
    adj        : (N, N) symmetric adjacency (CSR)
    cell_types : (N,) array of cell-type string labels
    tau_mix    : spectral mixing timescale (= 1/λ₂)
    n_power_iter : max steps for each timescale via repeated multiplication

    Returns
    -------
    H : (N, K·3) concatenated diffusion features
        where K = number of unique cell types
    """
    n = adj.shape[0]

    # One-hot encode cell types
    lb = LabelBinarizer()
    X = lb.fit_transform(cell_types).astype(float)   # (N, K)

    # Row-normalised transition matrix  P = D^{-1} A
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    P = sp.diags(deg_inv) @ adj                       # (N, N) sparse

    timescales = [
        max(1, int(round(1.0))),
        max(1, int(round(tau_mix / 4.0))),
        max(1, int(round(tau_mix))),
    ]
    # Cap to avoid O(N·t) blowup
    timescales = [min(t, n_power_iter) for t in timescales]

    features = []
    Xt = X.copy()
    prev_t = 0
    for t in sorted(set(timescales)):
        steps = t - prev_t
        for _ in range(steps):
            Xt = P @ Xt
        prev_t = t
        features.append(Xt.copy())

    H = np.hstack(features)         # (N, K·3)
    # L2-normalise each row for cosine-distance compatibility
    norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
    return H / norms


# ──────────────────────────────────────────────────────────────────────────────
# Community detection  (Leiden preferred, spectral fallback)
# ──────────────────────────────────────────────────────────────────────────────

def detect_communities(
    adj: sp.csr_matrix,
    target_n_communities: int | None = None,
    resolution: float = 1.0,
) -> np.ndarray:
    """
    Detect graph communities.  Tries Leiden (via leidenalg) first,
    falls back to Spectral Clustering if leidenalg is not installed.

    Returns
    -------
    labels : (N,) integer community label array
    """
    n = adj.shape[0]
    k = target_n_communities or max(5, n // 200)

    # ── Try Leiden ─────────────────────────────────────────────────────────
    try:
        import igraph as ig
        import leidenalg

        rows, cols = adj.nonzero()
        edges = list(zip(rows.tolist(), cols.tolist()))
        # keep upper triangle only (undirected)
        edges = [(u, v) for u, v in edges if u < v]
        g = ig.Graph(n=n, edges=edges)
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            n_iterations=10,
            seed=42,
        )
        return np.array(partition.membership)

    except ImportError:
        pass   # fall through to spectral

    # ── Spectral clustering fallback ────────────────────────────────────────
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=k, affinity='precomputed',
        random_state=42, n_init=5,
    )
    sc.fit(adj.toarray())
    return sc.labels_
