"""! @file swe_pinn/problems/dam_break.py
@brief Scaled wet-bed dam-break benchmark matching Desktop/2_dam_break
@brief (Generate_Data_scaled.py / dambreak_scaled_*_optuna.py).

Physical setup from the working scaled case:
  g = 9.81, h_left = 0.5, h_right = 0.25, u0 = 0,
  x in [0, 1], t in [0, 0.2], dam at x = 0.5,
  sharp step IC (x <= x_mid -> h_left).

Soft IC/BC data match the reference Nd points and loss:
  11 IC + 10 left BC + 10 right BC, weight 50 on h and u data MSE,
  as in ``loss = L1 + L2 + 50*L3 + 50*L4``.

Analytic evaluation uses the Delestre / Int. J. Numer. Meth. Fluids
formulae from DamBreakAnalytic.py (equivalent to the Stoker wet-bed
Riemann solution for this setup).

@book{stoker1957water,
  author    = {Joseph J. Stoker},
  title     = {Water Waves: The Mathematical Theory with Applications},
  publisher = {Interscience Publishers},
  address   = {New York},
  year      = {1957}
}
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve
import tensorflow as tf

from swe_pinn.physics.common import DTYPE, Domain1D, Problem, unit_interval


def build_dam_break_problem(p):
    G = tf.constant(p.g, dtype=DTYPE)
    domain = Domain1D(p.x_min, p.x_max, p.t_min, p.t_max)

    def h_init(x):
        """Sharp wet-bed step used by 2_dam_break Generate_Data_scaled.initial_h."""
        if float(getattr(p, "transition_width", 0.0)) <= 0.0:
            return tf.where(
                x <= tf.cast(p.x_mid, DTYPE),
                tf.cast(p.h_left, DTYPE),
                tf.cast(p.h_right, DTYPE),
            )

        return (
            0.5 * (p.h_left + p.h_right)
            + 0.5
            * (p.h_left - p.h_right)
            * tf.tanh((p.x_mid - x) / p.transition_width)
        )

    def output_fn(model, x, t, constraints: str):
        # Network sees both coordinates on [0, 1]. Column order stays [t, x].
        # Residuals and the hard ansatz below still use physical x and t.
        t_u = unit_interval(t, p.t_min, p.t_max)
        x_u = unit_interval(x, p.x_min, p.x_max)
        raw = model(tf.concat([t_u, x_u], axis=1))

        eta = raw[:, 0:1]
        u = raw[:, 1:2]

        if str(constraints).strip().lower() == "hard":
            x_scaled = (x - p.x_min) / (p.x_max - p.x_min + 1.0e-12)
            t_scaled = (t - p.t_min) / (p.t_max - p.t_min + 1.0e-12)
            vanish = t_scaled * x_scaled * (1.0 - x_scaled)
            eta = h_init(x) + vanish * eta
            u = tf.cast(p.u0, DTYPE) + vanish * u

        return eta, u

    # Initial samples are packed toward the dam. Uniform samples put the
    # jump between two points and let the flat water dominate the mean.
    n_ic = int(getattr(p, "soft_n_ic", 41))
    n_bc = int(getattr(p, "soft_n_bc", 10))
    gamma = 1.3
    u_ic = np.linspace(-1.0, 1.0, n_ic, dtype=np.float64)
    mag = np.abs(u_ic) ** gamma
    half_left = float(p.x_mid - p.x_min)
    half_right = float(p.x_max - p.x_mid)
    x_ic_np = np.where(
        u_ic < 0.0,
        p.x_mid - half_left * mag,
        p.x_mid + half_right * mag,
    ).reshape(-1, 1)
    t_ic_np = np.zeros((n_ic, 1), dtype=np.float64)
    t_bc_np = np.linspace(p.t_min, p.t_max, n_bc, dtype=np.float64).reshape(-1, 1)
    x_left_np = np.full((n_bc, 1), p.x_min, dtype=np.float64)
    x_right_np = np.full((n_bc, 1), p.x_max, dtype=np.float64)

    # Combined data tensors in the same order as reference x_d / t_d / h_d / u_d.
    x_d_np = np.concatenate([x_ic_np, x_left_np, x_right_np], axis=0)
    t_d_np = np.concatenate([t_ic_np, t_bc_np, t_bc_np], axis=0)
    h_d_np = np.concatenate(
        [
            np.where(x_ic_np <= p.x_mid, p.h_left, p.h_right),
            np.full((n_bc, 1), p.h_left, dtype=np.float64),
            np.full((n_bc, 1), p.h_right, dtype=np.float64),
        ],
        axis=0,
    )
    u_d_np = np.full_like(h_d_np, float(p.u0))

    x_d_tf = tf.constant(x_d_np, dtype=DTYPE)
    t_d_tf = tf.constant(t_d_np, dtype=DTYPE)
    h_d_tf = tf.constant(h_d_np, dtype=DTYPE)
    u_d_tf = tf.constant(u_d_np, dtype=DTYPE)

    # IC and BC are separate means, so the easy wall points do not dilute the step.
    w_ic_h = tf.cast(float(getattr(p, "soft_w_ic_h", 50.0)), DTYPE)
    w_ic_u = tf.cast(float(getattr(p, "soft_w_ic_u", 50.0)), DTYPE)
    w_bc_h = tf.cast(float(getattr(p, "soft_w_bc_h", 10.0)), DTYPE)
    w_bc_u = tf.cast(float(getattr(p, "soft_w_bc_u", 10.0)), DTYPE)

    def soft_constraint_loss_fn(model):
        """Soft IC and BC penalties. ic + bc is the data term in the objective."""
        n_ic = int(x_ic_np.shape[0])

        h_pred, u_pred = output_fn(model, x_d_tf, t_d_tf, "soft")

        loss_ic_h = tf.reduce_mean(tf.square(h_pred[:n_ic] - h_d_tf[:n_ic]))
        loss_ic_u = tf.reduce_mean(tf.square(u_pred[:n_ic] - u_d_tf[:n_ic]))
        loss_bc_h = tf.reduce_mean(tf.square(h_pred[n_ic:] - h_d_tf[n_ic:]))
        loss_bc_u = tf.reduce_mean(tf.square(u_pred[n_ic:] - u_d_tf[n_ic:]))
        loss_ic = w_ic_h * loss_ic_h + w_ic_u * loss_ic_u
        loss_bc = w_bc_h * loss_bc_h + w_bc_u * loss_bc_u
        weighted = loss_ic + loss_bc
        loss_h = loss_ic_h + loss_bc_h
        loss_u = loss_ic_u + loss_bc_u

        components = {
            "ic": loss_ic,
            "bc": loss_bc,
            "ic_h": loss_ic_h,
            "ic_u": loss_ic_u,
            "bc_h": loss_bc_h,
            "bc_u": loss_bc_u,
            "data_h": loss_h,
            "data_u": loss_u,
        }
        return weighted, components

    def residual_fn(model, x, t, constraints: str):
        # Conservative 1D SWE residuals used by 2_dam_break:
        #   F1 = h_t + (h u)_x
        #   F2 = (h u)_t + (0.5 g h^2 + h u^2)_x
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])

            h, u = output_fn(model, x, t, constraints)

            q = h * u
            flux = q * u + 0.5 * G * h * h

        h_t = tape.gradient(h, t)
        q_t = tape.gradient(q, t)
        q_x = tape.gradient(q, x)
        flux_x = tape.gradient(flux, x)

        del tape

        return h_t + q_x, q_t + flux_x

    def exact_solution(x, t):
        """Delestre wet-bed dam-break solution from 2_dam_break DamBreakAnalytic.py."""

        x_np = np.asarray(x, dtype=float).reshape(-1)
        t_np = np.asarray(t, dtype=float).reshape(-1)

        if t_np.size == 1 and x_np.size > 1:
            t_np = np.full_like(x_np, float(t_np[0]))

        if x_np.size != t_np.size:
            raise ValueError(
                f"exact_solution expected x and t to have compatible sizes, "
                f"got x={x_np.size}, t={t_np.size}"
            )

        h_out = np.zeros_like(x_np)
        u_out = np.zeros_like(x_np)

        hL = float(p.h_left)
        hR = float(p.h_right)
        u0 = float(p.u0)
        g = float(p.g)
        x0 = float(p.x_mid)

        def initial_h(xx):
            return np.where(xx <= x0, hL, hR)

        def analytic_h(xx, tt, xa, xb, xc, cm):
            return (
                np.less(xx, xa) * hL
                + np.greater(xx, xc) * hR
                + np.logical_and(np.less_equal(xa, xx), np.less_equal(xx, xb))
                * (4.0 / (9.0 * g) * (np.sqrt(g * hL) - (xx - x0) / (2.0 * tt)) ** 2)
                + np.logical_and(np.less(xb, xx), np.less_equal(xx, xc)) * (cm**2 / g)
            )

        def analytic_u(xx, tt, xa, xb, xc, cm):
            return (
                np.logical_and(np.less_equal(xa, xx), np.less_equal(xx, xb))
                * (2.0 / 3.0 * ((xx - x0) / tt + np.sqrt(g * hL)))
                + np.logical_and(np.less(xb, xx), np.less_equal(xx, xc))
                * (2.0 * (np.sqrt(g * hL) - cm))
            )

        def solve_one_time(xx, tt):
            if tt <= 0.0:
                return initial_h(xx), np.full_like(xx, u0)

            def find_hm(hm):
                return (
                    -8.0 * g * hR * g * hm * (np.sqrt(g * hL) - np.sqrt(g * hm)) ** 2
                    + (g * hm - g * hR) ** 2 * (g * hm + g * hR)
                )

            hm = float(fsolve(find_hm, 0.5 * (hL + hR))[0])
            cm = np.sqrt(g * hm)
            xa = x0 - tt * np.sqrt(g * hL)
            xb = x0 + tt * (2.0 * np.sqrt(g * hL) - 3.0 * cm)
            xc = x0 + tt * (2.0 * cm**2 * (np.sqrt(g * hL) - cm)) / (cm**2 - g * hR)

            hh = analytic_h(xx, tt, xa, xb, xc, cm)
            uu = analytic_u(xx, tt, xa, xb, xc, cm) + u0

            # Match Generate_Data_scaled nan cleanup near the discontinuity.
            nan_mask = np.isnan(hh)
            if np.any(nan_mask):
                hh[nan_mask] = initial_h(xx[nan_mask])
            uu = np.nan_to_num(uu, nan=u0, posinf=u0, neginf=u0)
            return hh, uu

        for tt in np.unique(t_np):
            mask = np.abs(t_np - tt) < 1.0e-12
            h_out[mask], u_out[mask] = solve_one_time(x_np[mask], float(tt))

        return (
            tf.convert_to_tensor(h_out.reshape(-1, 1), dtype=DTYPE),
            tf.convert_to_tensor(u_out.reshape(-1, 1), dtype=DTYPE),
        )

    return Problem(
        name="dam_break",
        domain=domain,
        residual_fn=residual_fn,
        output_fn=output_fn,
        exact_solution=exact_solution,
        to_dimensional=None,
        soft_constraint_loss_fn=soft_constraint_loss_fn,
    )
