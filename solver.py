"""
stfa/solver.py
==============
Unbalanced Fused Gromov-Wasserstein solver.

Key ideas vs. old balanced FGW:
  - KL-penalised marginals (ρ_A, ρ_B) let unmatched cells "die" naturally
  - ρ is calibrated from convex-hull overlap fraction
  - Uses POT mm_unbalanced inside a simple conditional-gradient GW loop
  - Falls back to balanced emd if ot.unbalanced is unavailable
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Overlap fraction estimation
# ──────────────────────────────────────────────────────────────────────────────

def _hull_area(pts: np.ndarray) -> float:
    """Return convex hull area; 0 for degenerate point clouds."""
    try:
        return ConvexHull(pts).volume   # .volume is area in 2D
    except (QhullError, Exception):
        return 0.0


def estimate_overlap_fraction(
    coordsA: np.ndarray,
    coordsB: np.ndarray,
    n_mc: int = 5000,
    seed: int = 42,
) -> float:
    """
    Monte-Carlo estimate of the convex-hull overlap fraction between two 2D
    point clouds.

    Returns
    -------
    f_ovlp : float in [0, 1]  — fraction of A's hull covered by B's hull
    """
    rng = np.random.default_rng(seed)

    try:
        hull_A = ConvexHull(coordsA)
    except (QhullError, Exception):
        return 0.5    # degenerate: assume 50% overlap

    # Sample random points inside hull_A
    lo = coordsA.min(axis=0)
    hi = coordsA.max(axis=0)
    pts = rng.uniform(lo, hi, size=(n_mc, 2))

    # Keep only those inside hull_A (half-space test)
    A_mat = hull_A.equations[:, :2]
    b_vec = hull_A.equations[:, 2]
    in_A = (pts @ A_mat.T + b_vec <= 1e-10).all(axis=1)
    pts_A = pts[in_A]
    if len(pts_A) == 0:
        return 0.5

    # Test which of those are also inside hull_B
    try:
        hull_B = ConvexHull(coordsB)
    except (QhullError, Exception):
        return 0.5

    B_mat = hull_B.equations[:, :2]
    b_B = hull_B.equations[:, 2]
    in_B = (pts_A @ B_mat.T + b_B <= 1e-10).all(axis=1)

    return float(in_B.sum()) / len(pts_A)


# ──────────────────────────────────────────────────────────────────────────────
# ρ calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_rho(f_ovlp: float, rho_min: float = 0.01, rho_max: float = 1.0) -> float:
    """
    KL penalty parameter ρ from maximum-entropy estimate given known overlap.

    ρ = −f · log(f)   (peaked at f=1/e ≈ 0.37, zero at f=0 and f=1)

    When f≈1 (full overlap) ρ is small → nearly balanced (like PASTE).
    When f≈0 (barely any overlap) ρ is small → very unbalanced (destroy all mass).
    """
    f = float(np.clip(f_ovlp, 1e-6, 1 - 1e-6))
    rho = -f * np.log(f)
    return float(np.clip(rho, rho_min, rho_max))


# ──────────────────────────────────────────────────────────────────────────────
# Normalise distance matrices
# ──────────────────────────────────────────────────────────────────────────────

def _norm_dist(D: np.ndarray) -> np.ndarray:
    m = D.max()
    return D / (m + 1e-12) if m > 0 else D


# ──────────────────────────────────────────────────────────────────────────────
# Unbalanced FGW solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_ufgw(
    p_A: np.ndarray,
    p_B: np.ndarray,
    M_fused: np.ndarray,
    C_A: np.ndarray,
    C_B: np.ndarray,
    rho: float,
    gamma: float = 0.5,
    eps: float = 0.05,
    n_iter: int = 200,
    tol: float = 1e-7,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve the Unbalanced Fused Gromov-Wasserstein problem via a
    conditional-gradient (Frank-Wolfe) loop.

    Objective:
        min_{π≥0}  (1-γ)⟨π, M_fused⟩
                  + γ · GW(C_A, C_B, π)
                  + ρ · KL(π1 ‖ p_A) + ρ · KL(πᵀ1 ‖ p_B)

    At each iteration:
      1. Linearise the GW term  →  form linear cost M_lin
      2. Solve a small *unbalanced* OT with KL penalties (mm_unbalanced)
      3. Armijo line-search

    Parameters
    ----------
    p_A, p_B  : marginal priors (uniform 1/N)
    M_fused   : (N_A, N_B) fused linear cost
    C_A, C_B  : (N_A, N_A), (N_B, N_B) spatial distance matrices (normalised)
    rho       : KL penalty for both marginals
    gamma     : balance between feature cost (0) and geometry (1)
    eps       : entropic regularisation for mm_unbalanced (smoother gradients)
    n_iter    : max CG iterations
    tol       : convergence threshold on |Δcost|/|cost|

    Returns
    -------
    pi : (N_A, N_B) transport plan
    """
    import ot

    N_A, N_B = M_fused.shape

    # ── Initialise with outer product (independent coupling) ─────────────────
    pi = np.outer(p_A, p_B)

    # ── Helper: GW gradient ──────────────────────────────────────────────────
    def _gw_grad(pi_cur: np.ndarray) -> np.ndarray:
        """∂GW/∂π = 4 (C_A · π · C_B)  for square loss."""
        return 4.0 * (C_A @ pi_cur @ C_B)

    # ── Helper: scalar GW cost ────────────────────────────────────────────────
    def _gw_cost(pi_cur: np.ndarray) -> float:
        T1 = np.einsum('ij,ij->', C_A @ pi_cur @ C_B, pi_cur)
        return float(4.0 * T1)

    # ── Helper: unbalanced OT step ────────────────────────────────────────────
    def _ot_step(M_lin: np.ndarray) -> np.ndarray:
        M_pos = M_lin - M_lin.min()   # shift non-negative for mm_unbalanced
        try:
            pi_new = ot.unbalanced.mm_unbalanced(
                p_A, p_B, M_pos,
                reg_m=(rho, rho),
                div='kl',
                numItermax=500,
                stopThr=1e-8,
            )
        except Exception:
            # Fallback: balanced emd (balanced OT)
            try:
                pi_new = ot.emd(p_A, p_B, M_pos)
            except Exception:
                pi_new = np.outer(p_A, p_B)
        return np.asarray(pi_new, dtype=np.float64)

    # ── Helper: total cost ────────────────────────────────────────────────────
    def _total_cost(pi_cur: np.ndarray) -> float:
        linear = (1.0 - gamma) * float(np.sum(M_fused * pi_cur))
        gw = gamma * _gw_cost(pi_cur)
        row = pi_cur.sum(axis=1)
        col = pi_cur.sum(axis=0)
        kl_A = float(np.sum(row * np.log((row + 1e-12) / (p_A + 1e-12)) - row + p_A))
        kl_B = float(np.sum(col * np.log((col + 1e-12) / (p_B + 1e-12)) - col + p_B))
        return linear + gw + rho * kl_A + rho * kl_B

    # ── Conditional gradient loop ─────────────────────────────────────────────
    prev_cost = _total_cost(pi)
    for it in range(n_iter):
        # Linearised total cost gradient at current pi
        M_lin = (1.0 - gamma) * M_fused + gamma * _gw_grad(pi)

        # Frank-Wolfe direction
        pi_fw = _ot_step(M_lin)

        # Armijo line search  α ∈ [0, 1]
        delta = pi_fw - pi
        best_alpha = 1.0
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05]:
            pi_test = pi + alpha * delta
            pi_test = np.clip(pi_test, 0, None)
            c_test = _total_cost(pi_test)
            if c_test <= prev_cost:
                best_alpha = alpha
                break

        pi = np.clip(pi + best_alpha * delta, 0, None)
        cur_cost = _total_cost(pi)

        rel = abs(cur_cost - prev_cost) / (abs(prev_cost) + 1e-12)
        if verbose and it % 20 == 0:
            print(f"  [UFGW] iter {it:4d}  cost={cur_cost:.6e}  Δrel={rel:.2e}")
        if rel < tol:
            if verbose:
                print(f"  [UFGW] converged at iter {it}")
            break
        prev_cost = cur_cost

    return pi
