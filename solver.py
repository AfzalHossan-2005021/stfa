"""
stfa/solver.py
==============
Unbalanced Fused Gromov-Wasserstein solver.

Key ideas vs. old balanced FGW:
  - KL-penalised marginals (ρ_A, ρ_B) let unmatched cells "die" naturally
    - ρ is calibrated from transform-invariant convex-hull overlap fraction
  - Uses POT mm_unbalanced inside a simple conditional-gradient GW loop
  - Falls back to balanced emd if ot.unbalanced is unavailable
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from typing import Tuple


def _canonicalise_coords(coords: np.ndarray) -> np.ndarray:
    """
    Canonicalise coordinates for transform-invariant overlap estimation.

    Removes translation and global scale; rotation is handled separately by
    scanning candidate angles in estimate_overlap_fraction.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    centered = pts - pts.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    scale = float(np.median(norms))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.mean(norms) + 1e-12)
    return centered / (scale + 1e-12)


def estimate_overlap_fraction(
    coordsA: np.ndarray,
    coordsB: np.ndarray,
    n_mc: int = 5000,
    seed: int = 42,
    n_rotations: int = 24,
) -> float:
    """
    Transform-invariant Monte-Carlo estimate of convex-hull overlap fraction.

    Both slices are first canonicalised for translation and global scale.
    Then we scan a small set of global rotations for slice B and keep the
    best overlap value. This avoids severely underestimating overlap when two
    slices differ mostly by whole-slice translation/rotation.

    Returns
    -------
    f_ovlp : float in [0, 1]  — fraction of A's hull covered by B's hull
    """
    rng = np.random.default_rng(seed)

    A = _canonicalise_coords(coordsA)
    B = _canonicalise_coords(coordsB)
    if A.shape[0] < 3 or B.shape[0] < 3:
        return 0.5

    try:
        hull_A = ConvexHull(A)
    except (QhullError, Exception):
        return 0.5    # degenerate: assume 50% overlap

    # Sample random points inside hull_A
    lo = A.min(axis=0)
    hi = A.max(axis=0)
    pts = rng.uniform(lo, hi, size=(n_mc, 2))

    # Keep only those inside hull_A (half-space test)
    A_mat = hull_A.equations[:, :2]
    b_vec = hull_A.equations[:, 2]
    in_A = (pts @ A_mat.T + b_vec <= 1e-10).all(axis=1)
    pts_A = pts[in_A]
    if len(pts_A) == 0:
        return 0.5

    # Test overlap across candidate global rotations for B.
    n_rot = int(max(1, n_rotations))
    thetas = np.linspace(0.0, 2.0 * np.pi, num=n_rot, endpoint=False)

    best = 0.0
    for th in thetas:
        c = float(np.cos(th))
        s = float(np.sin(th))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        B_rot = (R @ B.T).T

        try:
            hull_B = ConvexHull(B_rot)
        except (QhullError, Exception):
            continue

        B_mat = hull_B.equations[:, :2]
        b_B = hull_B.equations[:, 2]
        in_B = (pts_A @ B_mat.T + b_B <= 1e-10).all(axis=1)
        frac = float(in_B.sum()) / len(pts_A)
        if frac > best:
            best = frac

    if best <= 0.0:
        return 0.5
    return float(np.clip(best, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# ρ calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_rho(
    f_ovlp: float,
    rho_min: float = 0.05,
    rho_max: float = 20.0,
    overlap_power: float = 1.5,
    rho_scale: float = 1.0,
) -> float:
    """
    Calibrate KL mass-penalty ρ from estimated overlap.

    We use a monotone increasing schedule:
      - high overlap -> larger ρ -> closer to balanced mass transport
      - low overlap  -> smaller ρ -> allows more unmatched mass

    The scale factor rho_scale lets users globally tighten/relax mass matching
    without changing overlap estimation.
    """
    f = float(np.clip(f_ovlp, 0.0, 1.0))
    p = float(max(0.5, overlap_power))
    rho = float(rho_min + (rho_max - rho_min) * (f ** p))
    rho = float(max(0.0, rho_scale)) * rho
    return float(np.clip(rho, rho_min, rho_max))


# ──────────────────────────────────────────────────────────────────────────────
# Normalise distance matrices
# ──────────────────────────────────────────────────────────────────────────────

def _norm_dist(D: np.ndarray) -> np.ndarray:
    m = D.max()
    return D / (m + 1e-12) if m > 0 else D


def _bidir_power_sharpen(
    pi: np.ndarray,
    power: float = 1.0,
    rounds: int = 1,
) -> np.ndarray:
    """
    Sharpen transport conditionals in both directions.

    A mild power transform (power > 1) reduces diffuse many-to-many links while
    preserving current row/column mass profiles at each alternating step.
    """
    pwr = float(power)
    n_rounds = int(rounds)
    if pwr <= 1.0 or n_rounds <= 0:
        return np.asarray(pi, dtype=np.float64)

    out = np.clip(np.asarray(pi, dtype=np.float64), 0.0, None)

    for _ in range(n_rounds):
        # Row-wise sharpening: one source -> fewer targets.
        row_mass = out.sum(axis=1, keepdims=True)
        nz_row = row_mass[:, 0] > 0
        if np.any(nz_row):
            row_sharp = np.power(out[nz_row, :], pwr)
            row_sharp /= row_sharp.sum(axis=1, keepdims=True) + 1e-12
            out[nz_row, :] = row_sharp * row_mass[nz_row, :]

        # Column-wise sharpening: one target <- fewer sources.
        col_mass = out.sum(axis=0, keepdims=True)
        nz_col = col_mass[0, :] > 0
        if np.any(nz_col):
            col_sharp = np.power(out[:, nz_col], pwr)
            col_sharp /= col_sharp.sum(axis=0, keepdims=True) + 1e-12
            out[:, nz_col] = col_sharp * col_mass[:, nz_col]

    return out


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
    confidence_power: float = 1.0,
    confidence_rounds: int = 1,
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
    confidence_power  : >1 sharpens couplings toward one-to-few correspondences
    confidence_rounds : alternating row/column sharpening rounds per OT step

    Returns
    -------
    pi : (N_A, N_B) transport plan
    """
    import ot

    N_A, N_B = M_fused.shape

    # Pre-compute for square-loss GW terms.
    C_A_sq = C_A * C_A
    C_B_sq = C_B * C_B

    # ── Initialise with outer product (independent coupling) ─────────────────
    pi = np.outer(p_A, p_B)

    # ── Helper: GW gradient ──────────────────────────────────────────────────
    def _gw_grad(pi_cur: np.ndarray) -> np.ndarray:
        """
        Gradient of square-loss GW with variable (unbalanced) marginals.

        GW(π) = <C_A^2, r r^T> + <C_B^2, c c^T> - 2 <C_A π C_B, π>
        where r = π1 and c = π^T1.
        """
        row = pi_cur.sum(axis=1)  # r
        col = pi_cur.sum(axis=0)  # c

        term_a = 2.0 * (C_A_sq @ row)[:, None]
        term_b = 2.0 * (C_B_sq @ col)[None, :]
        cross = 4.0 * (C_A @ pi_cur @ C_B)
        return term_a + term_b - cross

    # ── Helper: scalar GW cost ────────────────────────────────────────────────
    def _gw_cost(pi_cur: np.ndarray) -> float:
        row = pi_cur.sum(axis=1)
        col = pi_cur.sum(axis=0)
        term_a = float(row @ (C_A_sq @ row))
        term_b = float(col @ (C_B_sq @ col))
        cross = float(np.einsum('ij,ij->', C_A @ pi_cur @ C_B, pi_cur))
        return term_a + term_b - 2.0 * cross

    # ── Helper: unbalanced OT step ────────────────────────────────────────────
    def _ot_step(M_lin: np.ndarray) -> np.ndarray:
        M_pos = M_lin - M_lin.min()   # shift non-negative for mm_unbalanced
        try:
            pi_new = ot.unbalanced.mm_unbalanced(
                p_A, p_B, M_pos,
                reg_m=(rho, rho),
                reg=eps,
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
        pi_new = np.asarray(pi_new, dtype=np.float64)
        return _bidir_power_sharpen(
            pi_new,
            power=confidence_power,
            rounds=confidence_rounds,
        )

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

    return _bidir_power_sharpen(
        pi,
        power=confidence_power,
        rounds=confidence_rounds,
    )
