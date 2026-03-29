"""
stfa/real_data_sweep.py
=======================
Constrained parameter sweep on real AnnData slices.

This utility searches around the strict STFA profile and reports settings that
best satisfy target constraints, with primary focus on:
  - high cell-type correspondence
  - strong JSD neighborhood improvement
  - low unmatched mass

Run example:
    python -m stfa.real_data_sweep \
        --slice-a path/to/sliceA.h5ad \
        --slice-b path/to/sliceB.h5ad \
        --radius 100 \
        --max-trials 36
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import time
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np

from stfa import (
    get_perf_metrics_enhanced,
    pairwise_align_stfa,
    stack_slices_pairwise,
)


def _validate_slice_fields(slice_obj: ad.AnnData, name: str) -> None:
    if "spatial" not in slice_obj.obsm:
        raise ValueError(f"{name} is missing obsm['spatial']")
    if "cell_type_annot" not in slice_obj.obs:
        raise ValueError(f"{name} is missing obs['cell_type_annot']")


def _base_config() -> Dict[str, Any]:
    return {
        "celltype_weight": 6.0,
        "neighborhood_weight": 3.0,
        "topology_weight": 0.0,
        "boundary_weight": 0.0,
        "anchor_weight": 0.25,
        "anchor_spatial_blend": 0.75,
        "compactness_weight": 2.0,
        "compactness_quantile": 0.80,
        "compactness_power": 1.75,
        "strict_celltype_gate": 3.0,
        "strict_compactness_gate": 1.0,
        "strict_compactness_quantile": 0.75,
        "region_geometry_weight": 2.0,
        "region_geometry_power": 1.25,
        "shape_context_weight": 2.0,
        "shape_context_power": 1.25,
        "geodesic_geometry_weight": 0.20,
        "rho_scale": 0.80,
        "target_mass_fraction": 0.90,
        "rho_retry_factor": 1.8,
        "rho_retry_rounds": 3,
        "confidence_power": 1.20,
        "gamma": 0.30,
    }


def _build_trial_configs(max_trials: int, seed: int) -> List[Dict[str, Any]]:
    base = _base_config()
    search_axes = {
        "strict_celltype_gate": [2.5, 3.0, 4.0],
        "strict_compactness_gate": [0.75, 1.0, 1.25],
        "neighborhood_weight": [2.5, 3.0, 4.0],
        "rho_scale": [0.70, 0.80, 1.00],
    }

    keys = list(search_axes.keys())
    all_cfgs: List[Dict[str, Any]] = []
    for values in itertools.product(*(search_axes[k] for k in keys)):
        cfg = dict(base)
        cfg.update({k: v for k, v in zip(keys, values)})
        all_cfgs.append(cfg)

    if max_trials <= 0 or max_trials >= len(all_cfgs):
        return all_cfgs

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_cfgs), size=max_trials, replace=False)
    idx = np.sort(idx)
    return [all_cfgs[i] for i in idx]


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _evaluate_trial(
    sA: ad.AnnData,
    sB: ad.AnnData,
    radius: float,
    n_iter: int,
    cfg: Dict[str, Any],
    trial_id: int,
    suppress_metric_report: bool,
) -> Dict[str, object]:
    t0 = time.time()
    try:
        pi12, in_n, in_g, fn_n, fg_g = pairwise_align_stfa(
            sA,
            sB,
            radius=radius,
            n_iter=n_iter,
            verbose=False,
            **cfg,
        )
        new_slices_any = stack_slices_pairwise([sA, sB], [pi12])
        if isinstance(new_slices_any, tuple):
            new_slices = new_slices_any[0]
        else:
            new_slices = new_slices_any
        if suppress_metric_report:
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = get_perf_metrics_enhanced(
                    new_slices,
                    pi12,
                    in_n,
                    in_g,
                    fn_n,
                    fg_g,
                    method_name=f"sweep_trial_{trial_id}",
                )
        else:
            metrics = get_perf_metrics_enhanced(
                new_slices,
                pi12,
                in_n,
                in_g,
                fn_n,
                fg_g,
                method_name=f"sweep_trial_{trial_id}",
            )

        out = {
            "trial_id": trial_id,
            "status": "ok",
            "runtime_sec": round(time.time() - t0, 3),
            "cell_type_match_pct": _as_float(metrics.get("cell_type_match_pct")),
            "jsd_improvement_pct": _as_float(metrics.get("jsd_improvement_pct")),
            "unmatched_mass_pct": _as_float(metrics.get("unmatched_mass_pct")),
            "mapped_geom_corr": _as_float(metrics.get("mapped_geom_corr")),
            "mapped_geom_stress": _as_float(metrics.get("mapped_geom_stress")),
            "symmetry_ambiguity": _as_float(metrics.get("symmetry_ambiguity")),
        }
        out.update(cfg)
        return out
    except Exception as exc:
        out = {
            "trial_id": trial_id,
            "status": "error",
            "runtime_sec": round(time.time() - t0, 3),
            "error": str(exc),
        }
        out.update(cfg)
        return out


def _is_feasible(
    row: Dict[str, object],
    min_celltype: float,
    min_jsd: float,
) -> bool:
    if row.get("status") != "ok":
        return False
    ct = _as_float(row.get("cell_type_match_pct"))
    jsd = _as_float(row.get("jsd_improvement_pct"))
    return (ct >= min_celltype) and (jsd >= min_jsd)


def _rank_ok_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    return sorted(
        ok,
        key=lambda r: (
            -_as_float(r.get("jsd_improvement_pct")),
            -_as_float(r.get("cell_type_match_pct")),
            _as_float(r.get("unmatched_mass_pct")),
            _as_float(r.get("mapped_geom_stress")),
        ),
    )


def _write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_row(prefix: str, row: Dict[str, object]) -> None:
    print(
        f"{prefix} "
        f"trial={row.get('trial_id')} "
        f"ct={_as_float(row.get('cell_type_match_pct')):.2f}% "
        f"jsd={_as_float(row.get('jsd_improvement_pct')):.2f}% "
        f"unmatched={_as_float(row.get('unmatched_mass_pct')):.2f}% "
        f"geomR={_as_float(row.get('mapped_geom_corr')):.3f} "
        f"stress={_as_float(row.get('mapped_geom_stress')):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run constrained STFA parameter sweep on real AnnData slices."
    )
    parser.add_argument("--slice-a", required=True, help="Path to source .h5ad")
    parser.add_argument("--slice-b", required=True, help="Path to target .h5ad")
    parser.add_argument("--radius", type=float, default=100.0, help="Neighborhood radius")
    parser.add_argument("--n-iter", type=int, default=80, help="Max UFGW iterations")
    parser.add_argument("--max-trials", type=int, default=36, help="Number of sampled trials")
    parser.add_argument("--seed", type=int, default=7, help="Sweep sampling seed")
    parser.add_argument(
        "--min-celltype",
        type=float,
        default=95.0,
        help="Minimum cell-type correspondence target (percent)",
    )
    parser.add_argument(
        "--min-jsd",
        type=float,
        default=50.0,
        help="Minimum JSD improvement target (percent)",
    )
    parser.add_argument(
        "--out-csv",
        default="sweep_results.csv",
        help="Output CSV path for all trials",
    )
    parser.add_argument(
        "--show-metric-report",
        action="store_true",
        help="Print full enhanced metric table for every trial",
    )
    args = parser.parse_args()

    path_a = Path(args.slice_a)
    path_b = Path(args.slice_b)
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError("Both --slice-a and --slice-b must exist.")

    print("Loading slices...")
    sA = ad.read_h5ad(path_a)
    sB = ad.read_h5ad(path_b)
    _validate_slice_fields(sA, "slice-a")
    _validate_slice_fields(sB, "slice-b")
    print(f"Loaded: A={sA.n_obs} cells, B={sB.n_obs} cells")

    cfgs = _build_trial_configs(max_trials=args.max_trials, seed=args.seed)
    print(f"Running {len(cfgs)} trial(s)...")

    rows: List[Dict[str, object]] = []
    for i, cfg in enumerate(cfgs, start=1):
        row = _evaluate_trial(
            sA,
            sB,
            radius=args.radius,
            n_iter=args.n_iter,
            cfg=cfg,
            trial_id=i,
            suppress_metric_report=(not args.show_metric_report),
        )
        rows.append(row)

        if row.get("status") == "ok":
            _print_row("  [ok]", row)
        else:
            print(f"  [error] trial={i} err={row.get('error')}")

    out_csv = Path(args.out_csv)
    _write_csv(rows, out_csv)
    print(f"Saved all trial results to: {out_csv}")

    ranked = _rank_ok_rows(rows)
    feasible = [r for r in ranked if _is_feasible(r, args.min_celltype, args.min_jsd)]

    if feasible:
        best = feasible[0]
        print("\nBest feasible trial (meets targets):")
        _print_row("  ->", best)
    elif ranked:
        print("\nNo trial met all targets. Best near-feasible trials:")
        for row in ranked[:5]:
            _print_row("  ->", row)
    else:
        print("\nNo successful trial. Check input data and logs.")


if __name__ == "__main__":
    main()
