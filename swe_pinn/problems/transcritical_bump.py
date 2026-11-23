"""! @file swe_pinn/problems/transcritical_bump.py
@brief Steady transcritical shallow-water flow over a bump with a stationary hydraulic jump.

Standard frictionless benchmark (SWASHES / classical well-balanced SWE test):
  x in [0, 25] m
  z(x) = 0.2 - 0.05 (x - 10)^2 for 8 < x < 12, otherwise 0
  upstream unit discharge q = 0.18 m^2/s
  downstream depth h = 0.33 m

The steady analytical reference is obtained from constant discharge and the
specific-energy relation on each smooth branch.  The stationary jump location
is determined from the Rankine--Hugoniot momentum condition.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
import tensorflow as tf

from swe_pinn.physics.common import DTYPE, Domain1D, Problem, unit_interval


def build_transcritical_bump_problem(p):
    G = tf.constant(p.g, dtype=DTYPE)
    Q_IN = tf.constant(p.q_in, dtype=DTYPE)
    H_DOWN = tf.constant(p.h_down, dtype=DTYPE)
    domain = Domain1D(p.x_min, p.x_max, p.t_min, p.t_max)

    def bed_tf(x):
        inside = tf.logical_and(x > tf.cast(p.bump_x1, DTYPE), x < tf.cast(p.bump_x2, DTYPE))
        bump = tf.cast(p.bump_height, DTYPE) - tf.cast(p.bump_coeff, DTYPE) * tf.square(
            x - tf.cast(p.bump_center, DTYPE)
        )
        return tf.where(inside, bump, tf.zeros_like(x))

    def bed_slope_tf(x):
        inside = tf.logical_and(x > tf.cast(p.bump_x1, DTYPE), x < tf.cast(p.bump_x2, DTYPE))
        slope = -2.0 * tf.cast(p.bump_coeff, DTYPE) * (
            x - tf.cast(p.bump_center, DTYPE)
        )
        return tf.where(inside, slope, tf.zeros_like(x))

    def output_fn(model, x, t, constraints: str):
        # This is a steady benchmark.  Keep the common (x,t) solver interface,
        # but evaluate the network at a fixed dummy time so the learned field is
        # a function of x only.
        t0 = tf.zeros_like(t)
        t_u = unit_interval(t0, p.t_min, p.t_max)
        x_u = unit_interval(x, p.x_min, p.x_max)
        raw = model(tf.concat([t_u, x_u], axis=1))
        h_raw = raw[:, 0:1]
        u_raw = raw[:, 1:2]

        if str(constraints).strip().lower() == "hard":
            xs = (x - p.x_min) / (p.x_max - p.x_min + 1.0e-12)

            # Mixed physical boundary conditions for the standard benchmark:
            #   q(0) = q_in,  h(L) = h_down.
            # First force the downstream depth exactly, then construct u so the
            # upstream discharge h*u is exactly q_in.
            h = H_DOWN + (1.0 - xs) * h_raw
            u = Q_IN / h + xs * u_raw
            return h, u

        return h_raw, u_raw

    # Soft enforcement uses the actual mixed boundary conditions rather than
    # prescribing both h and u at each end.
    x_left = tf.constant([[p.x_min]], dtype=DTYPE)
    x_right = tf.constant([[p.x_max]], dtype=DTYPE)
    t_zero = tf.constant([[0.0]], dtype=DTYPE)
    soft_w_q = tf.cast(float(getattr(p, "soft_w_q_in", 50.0)), DTYPE)
    soft_w_h = tf.cast(float(getattr(p, "soft_w_h_down", 50.0)), DTYPE)

    def soft_constraint_loss_fn(model):
        h_l, u_l = output_fn(model, x_left, t_zero, "soft")
        h_r, _ = output_fn(model, x_right, t_zero, "soft")
        q_l = h_l * u_l

        loss_q = tf.reduce_mean(tf.square(q_l - Q_IN))
        loss_h = tf.reduce_mean(tf.square(h_r - H_DOWN))
        weighted = soft_w_q * loss_q + soft_w_h * loss_h

        zero = tf.constant(0.0, dtype=DTYPE)
        return weighted, {
            "ic": zero,
            "bc": weighted,
            "bc_q_in": loss_q,
            "bc_h_down": loss_h,
        }

    def residual_fn(model, x, t, constraints: str):
        # Steady 1-D SWE with bathymetry and no friction:
        #   (hu)_x = 0
        #   (hu^2 + 1/2 g h^2)_x + g h z_x = 0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            h, u = output_fn(model, x, t, constraints)
            q = h * u
            flux = q * u + 0.5 * G * h * h

        q_x = tape.gradient(q, x)
        flux_x = tape.gradient(flux, x)
        del tape

        z_x = bed_slope_tf(x)
        return q_x, flux_x + G * h * z_x

    # --- Analytical steady reference -------------------------------------------------
    g = float(p.g)
    q0 = float(p.q_in)
    hL = float(p.h_down)
    zM = float(p.bump_height)
    xC = float(p.bump_center)
    hc = (q0 * q0 / g) ** (1.0 / 3.0)

    # Upstream transcritical branch passes through critical flow at the bump crest.
    E_up = zM + 1.5 * hc

    # Downstream subcritical branch is fixed by h(L)=h_down and z(L)=0.
    E_down = hL + q0 * q0 / (2.0 * g * hL * hL)

    def bed_np(x):
        xx = np.asarray(x, dtype=float)
        return np.where(
            (xx > p.bump_x1) & (xx < p.bump_x2),
            p.bump_height - p.bump_coeff * (xx - p.bump_center) ** 2,
            0.0,
        )

    def depth_root(E, z, branch):
        """Positive root of h + q^2/(2gh^2) + z = E."""
        min_head = z + 1.5 * hc
        if E < min_head - 1.0e-11:
            raise ValueError(
                f"No steady depth root: E={E:.12g} is below minimum head "
                f"{min_head:.12g} at z={z:.12g}."
            )
        if abs(E - min_head) <= 1.0e-10:
            return hc

        def f(h):
            return h + q0 * q0 / (2.0 * g * h * h) + z - E

        if branch == "supercritical":
            return brentq(f, 1.0e-10, hc * (1.0 - 1.0e-10), maxiter=200)
        if branch == "subcritical":
            lo = hc * (1.0 + 1.0e-10)
            hi = max(1.0, E - z + 1.0)
            while f(hi) <= 0.0:
                hi *= 2.0
            return brentq(f, lo, hi, maxiter=200)
        raise ValueError(f"Unknown branch {branch!r}")

    def momentum(h):
        return q0 * q0 / h + 0.5 * g * h * h

    def jump_residual(x):
        z = float(bed_np(np.array([x]))[0])
        h1 = depth_root(E_up, z, "supercritical")
        h2 = depth_root(E_down, z, "subcritical")
        return momentum(h1) - momentum(h2)

    # The downstream branch exists only after the bump has dropped enough for
    # E_down >= z + 3/2 h_c.  Search from there to the downstream boundary.
    z_allow = E_down - 1.5 * hc
    if z_allow < zM and p.bump_coeff > 0.0:
        x_branch = p.bump_center + np.sqrt(max((zM - z_allow) / p.bump_coeff, 0.0))
    else:
        x_branch = xC
    x_lo = min(max(float(x_branch) + 1.0e-8, xC + 1.0e-8), p.x_max - 1.0e-6)

    # Find a sign-changing bracket robustly instead of hard-coding shock position.
    scan = np.linspace(x_lo, p.x_max, 1000)
    vals = []
    valid_x = []
    for xx in scan:
        try:
            vals.append(jump_residual(float(xx)))
            valid_x.append(float(xx))
        except ValueError:
            pass
    x_shock = None
    for xa, xb, fa, fb in zip(valid_x[:-1], valid_x[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            x_shock = xa
            break
        if fa * fb < 0.0:
            x_shock = brentq(jump_residual, xa, xb, maxiter=200)
            break
    if x_shock is None:
        raise RuntimeError("Could not locate the stationary hydraulic jump for bump benchmark.")

    def exact_solution(x, t):
        x_np = np.asarray(x, dtype=float).reshape(-1)
        h = np.empty_like(x_np)

        for i, xx in enumerate(x_np):
            z = float(bed_np(np.array([xx]))[0])
            if abs(xx - xC) <= 1.0e-10:
                hh = hc
            elif xx < xC:
                hh = depth_root(E_up, z, "subcritical")
            elif xx < x_shock:
                hh = depth_root(E_up, z, "supercritical")
            else:
                hh = depth_root(E_down, z, "subcritical")
            h[i] = hh

        u = q0 / h
        return (
            tf.convert_to_tensor(h.reshape(-1, 1), dtype=DTYPE),
            tf.convert_to_tensor(u.reshape(-1, 1), dtype=DTYPE),
        )

    # Expose benchmark metadata for diagnostics/analysis without changing the
    # shared Problem dataclass API.
    problem = Problem(
        name="transcritical_bump",
        domain=domain,
        residual_fn=residual_fn,
        output_fn=output_fn,
        exact_solution=exact_solution,
        to_dimensional=None,
        soft_constraint_loss_fn=soft_constraint_loss_fn,
    )
    problem.bed_elevation = bed_tf
    problem.critical_depth = hc
    problem.x_shock = float(x_shock)
    problem.q_exact = q0
    return problem
