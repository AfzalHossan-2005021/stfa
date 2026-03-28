"""
stfa/smoke_test.py
==================
Quick end-to-end smoke test on synthetic MERFISH-like AnnData objects.

Run with:
    cd C:\\Users\\afzal\\OneDrive\\Desktop\\Research\\OPUS
    python -m stfa.smoke_test
"""

from __future__ import annotations

import sys
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

# ── Synthetic data generator ──────────────────────────────────────────────────

def make_synthetic_slice(
    n_cells: int = 300,
    n_genes: int = 50,
    n_types: int = 6,
    seed: int = 0,
    rotation_deg: float = 0.0,
    translation: tuple = (0.0, 0.0),
    partial_frac: float = 1.0,    # keep this fraction of cells (partial overlap)
    rng_expr_shift: float = 0.0,  # add noise to expression (temporal shift sim)
) -> AnnData:
    """
    Build a synthetic MERFISH-like AnnData.

    Cells are laid out on a structured 2D grid with Gaussian jitter.
    Each cell type occupies a distinct spatial region (so spatial alignment
    has a ground-truth solution).
    """
    rng = np.random.default_rng(seed)

    # ── Spatial layout: each type forms a blob ───────────────────────────────
    centres = np.array([
        [0, 0], [200, 0], [400, 0],
        [0, 200], [200, 200], [400, 200],
    ], dtype=float)[:n_types]

    cells_per_type = n_cells // n_types
    coords_list, type_list = [], []
    for t_idx, c in enumerate(centres):
        pts = rng.normal(loc=c, scale=40.0, size=(cells_per_type, 2))
        coords_list.append(pts)
        type_list.extend([f"Type_{t_idx}"] * cells_per_type)

    coords = np.vstack(coords_list)
    cell_types = np.array(type_list)

    # ── Apply rotation & translation ─────────────────────────────────────────
    if rotation_deg != 0.0:
        theta = np.deg2rad(rotation_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        coords = (R @ coords.T).T
    coords += np.array(translation)

    # ── Partial overlap: keep a random subset ────────────────────────────────
    if partial_frac < 1.0:
        keep = rng.choice(len(coords), size=int(len(coords) * partial_frac),
                          replace=False)
        coords = coords[keep]
        cell_types = cell_types[keep]

    n_actual = len(coords)

    # ── Gene expression: type-specific mean + noise ───────────────────────────
    unique_types = sorted(set(cell_types))
    type_means = {t: rng.uniform(0, 3, size=n_genes) for t in unique_types}
    expr = np.vstack([
        rng.poisson(type_means[t] + rng_expr_shift) for t in cell_types
    ]).astype(np.float32)

    adata = AnnData(
        X=sp.csr_matrix(expr),
        obs={"cell_type_annot": cell_types},
    )
    adata.obsm["spatial"] = coords.astype(np.float64)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    return adata


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test(n_cells: int = 300, verbose: bool = True):
    print("=" * 60)
    print("  STFA Smoke Test")
    print("=" * 60)

    # Create sliceA (full) and sliceB (rotated 35°, partial overlap 80%)
    sliceA = make_synthetic_slice(n_cells=n_cells, seed=42)
    sliceB = make_synthetic_slice(
        n_cells=int(n_cells * 0.8), seed=42,
        rotation_deg=35.0,
        translation=(50.0, -30.0),
        partial_frac=0.8,
    )

    print(f"  sliceA: {sliceA.n_obs} cells  |  sliceB: {sliceB.n_obs} cells")
    print(f"  Shared genes: {len(sliceA.var_names.intersection(sliceB.var_names))}")

    from stfa import pairwise_align_stfa, get_perf_metrics, get_perf_metrics_enhanced
    from stfa import visualize_alignment_unbalanced, stack_slices_pairwise

    # ── Run alignment ────────────────────────────────────────────────────────
    pi12, ini_n, ini_g, fin_n, fin_g = pairwise_align_stfa(
        sliceA, sliceB,
        radius=80.0,
        n_iter=50,        # short for smoke test
        verbose=verbose,
    )

    print(f"\n  pi12 shape: {pi12.shape}")
    assert pi12.shape == (sliceA.n_obs, sliceB.n_obs), \
        f"Expected ({sliceA.n_obs}, {sliceB.n_obs}), got {pi12.shape}"
    assert pi12.min() >= -1e-8, "Transport plan has negative entries"
    assert fin_n <= ini_n + 1e-6, \
        f"JSD did not improve: {ini_n:.6f} → {fin_n:.6f}"
    assert fin_g <= ini_g + 1e-6, \
        f"Cosine did not improve: {ini_g:.6f} → {fin_g:.6f}"

    # ── Stack & evaluate ─────────────────────────────────────────────────────
    new_slices = stack_slices_pairwise([sliceA, sliceB], [pi12])
    get_perf_metrics(new_slices, pi12, ini_n, ini_g, fin_n, fin_g, "STFA")
    metrics = get_perf_metrics_enhanced(new_slices, pi12, ini_n, ini_g, fin_n, fin_g, "STFA")

    print("\n  ✓ All assertions passed.")
    print("=" * 60)
    return pi12, new_slices, metrics


if __name__ == "__main__":
    run_smoke_test(verbose="--quiet" not in sys.argv)
