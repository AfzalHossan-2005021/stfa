"""
stfa/evaluate.py
================
Performance metrics for spatial transcriptomics alignment.

Provides:
  - cell_type_matching_metric()    (original, unchanged for compatibility)
  - get_perf_metrics()             (original, unchanged for compatibility)
  - get_perf_metrics_enhanced()    (extended with 6 biologically motivated metrics)
"""

from __future__ import annotations

import warnings
import numpy as np
from anndata import AnnData
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Original metrics (API-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def cell_type_matching_metric(sliceA: AnnData, sliceB: AnnData, pi_mat: np.ndarray) -> float:
    """
    Expected fraction (%) of transport mass that lands on the correct cell type.
    Probabilistic: works for soft/unbalanced transport plans.
    """
    expected_matches = 0.0
    for i in range(sliceA.n_obs):
        src_type = sliceA.obs.iloc[i]['cell_type_annot']
        match_mask = (sliceB.obs['cell_type_annot'] == src_type).values
        expected_matches += float(np.sum(pi_mat[i, match_mask]))
    total_mass = pi_mat.sum()
    if total_mass > 0:
        return (expected_matches / total_mass) * 100.0
    return 0.0


def get_perf_metrics(
    new_slices: List[AnnData],
    pi12: np.ndarray,
    neighbor_initial_obj: float,
    initial_obj_gene_cos: float,
    neighbor_final_obj: float,
    obj_gene_cos: float,
    method_name: str,
) -> None:
    """Original three-metric report. API-compatible with old code."""
    neighborhood_improvement = (
        (neighbor_initial_obj - neighbor_final_obj) / (neighbor_initial_obj + 1e-12) * 100
    )
    gene_expr_improvement = (
        (initial_obj_gene_cos - obj_gene_cos) / (initial_obj_gene_cos + 1e-12) * 100
    )
    expected_matches = cell_type_matching_metric(new_slices[0], new_slices[1], pi12)

    print(f"\n{'─'*55}")
    print(f"  Method: {method_name}")
    print(f"{'─'*55}")
    print(f"  JSD Neighbourhood")
    print(f"    Before: {neighbor_initial_obj:.7f}  After: {neighbor_final_obj:.7f}  "
          f"Δ: {neighborhood_improvement:+.4f}%")
    print(f"  Cosine Gene Expression")
    print(f"    Before: {initial_obj_gene_cos:.7f}  After: {obj_gene_cos:.7f}  "
          f"Δ: {gene_expr_improvement:+.4f}%")
    print(f"  Cell-type Correspondence: {expected_matches:.4f}%")
    print(f"{'─'*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Enhanced metrics
# ──────────────────────────────────────────────────────────────────────────────

def get_perf_metrics_enhanced(
    new_slices: List[AnnData],
    pi12: np.ndarray,
    neighbor_initial_obj: float,
    initial_obj_gene_cos: float,
    neighbor_final_obj: float,
    obj_gene_cos: float,
    method_name: str,
    k_nn: int = 15,
) -> dict:
    """
    Extended performance report with 6 biologically motivated additions:

    1.  Unmatched mass fraction  D_global      — tissue overlap quality
    2.  Spatial residual RMSE                  — geometric alignment error
    3.  k-NN Neighbourhood Structure Pres. NSP — local topology conservation
    4.  Transport entropy                       — symmetry confidence
    5.  Rare cell-type recall (< 1% freq)      — cross-temporal sensitivity
    6.  Symmetry ambiguity score               — whether top-2 targets compete

    Returns a dict of all metric values (for programmatic use).
    """
    from sklearn.neighbors import NearestNeighbors

    N_A, N_B = pi12.shape
    p_A = np.ones(N_A) / N_A

    # ── Original three ──────────────────────────────────────────────────────
    neighborhood_improvement = (
        (neighbor_initial_obj - neighbor_final_obj) / (neighbor_initial_obj + 1e-12) * 100
    )
    gene_expr_improvement = (
        (initial_obj_gene_cos - obj_gene_cos) / (initial_obj_gene_cos + 1e-12) * 100
    )
    expected_matches = cell_type_matching_metric(new_slices[0], new_slices[1], pi12)

    # ── 1. Unmatched mass fraction ────────────────────────────────────────────
    # Under KL-unbalanced OT the plan does NOT sum to 1/N_A per row.
    # D_global = fraction of source mass not transported anywhere.
    row_mass = pi12.sum(axis=1)                  # (N_A,)
    D_local  = np.clip(p_A - row_mass, 0, None)  # (N_A,) per-cell destroyed mass
    D_global = float(D_local.sum())               # fraction ∈ [0, 1]

    # ── 2. Spatial residual RMSE ──────────────────────────────────────────────
    s1 = new_slices[0].obsm['spatial']
    s2 = new_slices[1].obsm['spatial']
    pi_row_safe = row_mass[:, None] + 1e-12
    # Weighted average target position for each source cell
    src_mapped = (pi12 / pi_row_safe) @ s2          # (N_A, 2)
    residuals  = np.linalg.norm(s1 - src_mapped, axis=1)
    # Weight RMSE by how much mass each cell actually transported
    match_weight = row_mass * N_A                    # ~1 for matched, ~0 for unmatched
    spatial_rmse = float(
        np.sqrt(np.average(residuals ** 2, weights=match_weight + 1e-12))
    )

    # ── 3. k-NN Neighbourhood Structure Preservation (NSP) ───────────────────
    nn_s1 = NearestNeighbors(n_neighbors=k_nn).fit(s1)
    nn_s2 = NearestNeighbors(n_neighbors=k_nn).fit(s2)
    _, idx_s1 = nn_s1.kneighbors(s1)
    _, idx_s2 = nn_s2.kneighbors(s2)
    best_match = np.argmax(pi12, axis=1)          # greedy 1-1 for NSP

    nsp_scores = []
    for i in range(N_A):
        j = best_match[i]
        mapped_nbrs = set(best_match[idx_s1[i]])
        true_nbrs   = set(idx_s2[j])
        jaccard = len(mapped_nbrs & true_nbrs) / (len(mapped_nbrs | true_nbrs) + 1e-12)
        nsp_scores.append(jaccard)
    nsp = float(np.mean(nsp_scores) * 100)

    # ── 4. Transport entropy ─────────────────────────────────────────────────
    pi_norm    = pi12 / (pi12.sum() + 1e-12)
    entropy    = float(-np.sum(pi_norm * np.log(pi_norm + 1e-12)))
    max_entropy = float(np.log(N_A * N_B))
    entropy_pct = entropy / max_entropy * 100

    # ── 5. Rare cell-type recall ──────────────────────────────────────────────
    all_types  = new_slices[0].obs['cell_type_annot']
    type_freq  = all_types.value_counts(normalize=True)
    rare_types = type_freq[type_freq < 0.01].index.tolist()

    if rare_types:
        rare_mask = new_slices[0].obs['cell_type_annot'].isin(rare_types).values
        pi_rare   = pi12[rare_mask, :]
        rare_recall = cell_type_matching_metric(
            new_slices[0][rare_mask], new_slices[1], pi_rare
        )
        rare_str = f"{rare_recall:.4f}%"
    else:
        rare_recall = float('nan')
        rare_str = "N/A (no cell types < 1%)"

    # ── 6. Symmetry ambiguity score ───────────────────────────────────────────
    sorted_rows = np.sort(pi12, axis=1)          # ascending
    top1  = sorted_rows[:, -1] + 1e-12
    top2  = sorted_rows[:, -2] + 1e-12
    sym_ambig = float(np.mean(top2 / top1))      # 1.0 = fully ambiguous

    # ── Report ────────────────────────────────────────────────────────────────
    sep = '═' * 60
    print(f"\n{sep}")
    print(f"  {method_name} — Enhanced Performance Report")
    print(sep)
    # Original
    print(f"  JSD Neighbourhood     "
          f"Before: {neighbor_initial_obj:.6f}  "
          f"After: {neighbor_final_obj:.6f}  "
          f"Δ: {neighborhood_improvement:+.3f}%")
    print(f"  Cosine Gene Expr      "
          f"Before: {initial_obj_gene_cos:.6f}  "
          f"After: {obj_gene_cos:.6f}  "
          f"Δ: {gene_expr_improvement:+.3f}%")
    print(f"  Cell-Type Match       {expected_matches:.4f}%")
    # New
    print(f"{'─'*60}")
    print(f"  Unmatched Mass (D_global)      {D_global*100:.2f}% of source")
    print(f"  Spatial RMSE                   {spatial_rmse:.4f} (coord units)")
    print(f"  k-NN NSP (k={k_nn})             {nsp:.4f}%")
    print(f"  Transport Entropy              {entropy_pct:.2f}% of max "
          f"(low = confident)")
    print(f"  Rare Cell-Type Recall          {rare_str}")
    print(f"  Symmetry Ambiguity Score       {sym_ambig:.4f} "
          f"(1.0 = fully ambiguous)")
    print(f"{sep}\n")

    return {
        'jsd_before':         neighbor_initial_obj,
        'jsd_after':          neighbor_final_obj,
        'jsd_improvement_pct': neighborhood_improvement,
        'cosine_before':      initial_obj_gene_cos,
        'cosine_after':       obj_gene_cos,
        'cosine_improvement_pct': gene_expr_improvement,
        'cell_type_match_pct':    expected_matches,
        'unmatched_mass_pct': D_global * 100,
        'spatial_rmse':       spatial_rmse,
        'nsp_pct':            nsp,
        'entropy_pct':        entropy_pct,
        'rare_cell_recall_pct': rare_recall,
        'symmetry_ambiguity': sym_ambig,
    }
