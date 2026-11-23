"""! @file swe_pinn/problems/steady_vortex.py
@brief Isolated vortex in gradient-wind balance, with an exact solution.

Steady rotating shallow water on a flat bed. The azimuthal velocity is a
Gaussian vortex and the free surface satisfies

    v_theta^2 / r + f v_theta = g dh/dr

with no radial flow. Continuity is satisfied identically. There is no
manufactured body force. The same balance is the steady core of a tropical
cyclone or an ocean eddy, and it is the vortex produced on a rotating table.

The square is several vortex radii across, so the velocity at the boundary
is a small fraction of the peak. The hard constraint lifts the boundary
traces only. It does not insert the exact interior solution.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from swe_pinn.physics.common import DTYPE, Domain1D, Problem, unit_interval


def build_steady_vortex_problem(p):
    G = tf.constant(p.g, dtype=DTYPE)
    F = tf.constant(p.f, dtype=DTYPE)
    OMEGA = tf.constant(p.omega, dtype=DTYPE)
    SIGMA = tf.constant(p.sigma, dtype=DTYPE)
    H_C = tf.constant(p.h_center, dtype=DTYPE)
    span_x = float(p.x_max - p.x_min)
    span_y = float(p.y_max - p.y_min)
    if span_x <= 0.0 or span_y <= 0.0:
        raise ValueError("steady_vortex domain spans must be positive")
    domain = Domain1D(p.x_min, p.x_max, p.y_min, p.y_max)
    two_sigma_sq = 2.0 * SIGMA * SIGMA
    sigma_sq = SIGMA * SIGMA

    def exact_tf(x, y):
        r2 = tf.square(x) + tf.square(y)
        core = tf.exp(-r2 / two_sigma_sq)
        wide = tf.exp(-r2 / sigma_sq)
        u = -OMEGA * y * core
        v = OMEGA * x * core
        h = H_C + (OMEGA * OMEGA * sigma_sq / (2.0 * G)) * (1.0 - wide)
        h = h + (F * OMEGA * sigma_sq / G) * (1.0 - core)
        return h, u, v

    def exact_numpy(x, y):
        x_np = np.asarray(x, dtype=np.float64).reshape(-1)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        h, u, v = exact_tf(
            tf.constant(x_np.reshape(-1, 1), dtype=DTYPE),
            tf.constant(y_np.reshape(-1, 1), dtype=DTYPE),
        )
        return (
            h.numpy().reshape(-1),
            u.numpy().reshape(-1),
            v.numpy().reshape(-1),
        )

    def _as_columns(h, u, v):
        return (
            tf.convert_to_tensor(h.reshape(-1, 1), dtype=DTYPE),
            tf.convert_to_tensor(u.reshape(-1, 1), dtype=DTYPE),
            tf.convert_to_tensor(v.reshape(-1, 1), dtype=DTYPE),
        )

    def exact_solution(x, y):
        return _as_columns(*exact_numpy(x, y))

    def _coons(south, north, west, east, sw, se, nw, ne, sx, sy):
        return (
            (1.0 - sy) * south
            + sy * north
            + (1.0 - sx) * west
            + sx * east
            - (1.0 - sx) * (1.0 - sy) * sw
            - sx * (1.0 - sy) * se
            - (1.0 - sx) * sy * nw
            - sx * sy * ne
        )

    def output_fn(model, x, y, constraints: str):
        x_u = unit_interval(x, p.x_min, p.x_max)
        y_u = unit_interval(y, p.y_min, p.y_max)
        raw = model(tf.concat([x_u, y_u], axis=1))
        h_raw = raw[:, 0:1]
        u_raw = raw[:, 1:2]
        v_raw = raw[:, 2:3]
        if str(constraints).strip().lower() != "hard":
            return h_raw, u_raw, v_raw

        sx = (x - tf.cast(p.x_min, DTYPE)) / span_x
        sy = (y - tf.cast(p.y_min, DTYPE)) / span_y
        y_s = tf.ones_like(x) * tf.cast(p.y_min, DTYPE)
        y_n = tf.ones_like(x) * tf.cast(p.y_max, DTYPE)
        x_w = tf.ones_like(y) * tf.cast(p.x_min, DTYPE)
        x_e = tf.ones_like(y) * tf.cast(p.x_max, DTYPE)
        h_s, u_s, v_s = exact_tf(x, y_s)
        h_n, u_n, v_n = exact_tf(x, y_n)
        h_w, u_w, v_w = exact_tf(x_w, y)
        h_e, u_e, v_e = exact_tf(x_e, y)
        corner_x = tf.constant(
            [[p.x_min], [p.x_max], [p.x_min], [p.x_max]], dtype=DTYPE
        )
        corner_y = tf.constant(
            [[p.y_min], [p.y_min], [p.y_max], [p.y_max]], dtype=DTYPE
        )
        h_c, u_c, v_c = exact_tf(corner_x, corner_y)
        bubble = sx * (1.0 - sx) * sy * (1.0 - sy)
        h = _coons(h_s, h_n, h_w, h_e, h_c[0:1], h_c[1:2], h_c[2:3], h_c[3:4], sx, sy)
        u = _coons(u_s, u_n, u_w, u_e, u_c[0:1], u_c[1:2], u_c[2:3], u_c[3:4], sx, sy)
        v = _coons(v_s, v_n, v_w, v_e, v_c[0:1], v_c[1:2], v_c[2:3], v_c[3:4], sx, sy)
        return h + bubble * h_raw, u + bubble * u_raw, v + bubble * v_raw

    n_edge = int(getattr(p, "soft_n_edge", 32))
    xs = np.linspace(p.x_min, p.x_max, n_edge, dtype=np.float64)
    ys = np.linspace(p.y_min, p.y_max, n_edge, dtype=np.float64)
    edge_x = np.concatenate([xs, xs, np.full(n_edge, p.x_min), np.full(n_edge, p.x_max)])
    edge_y = np.concatenate([np.full(n_edge, p.y_min), np.full(n_edge, p.y_max), ys, ys])
    h_b, u_b, v_b = exact_numpy(edge_x, edge_y)
    x_b = tf.constant(edge_x.reshape(-1, 1), dtype=DTYPE)
    y_b = tf.constant(edge_y.reshape(-1, 1), dtype=DTYPE)
    h_b_tf = tf.constant(h_b.reshape(-1, 1), dtype=DTYPE)
    u_b_tf = tf.constant(u_b.reshape(-1, 1), dtype=DTYPE)
    v_b_tf = tf.constant(v_b.reshape(-1, 1), dtype=DTYPE)
    w_h = tf.cast(float(getattr(p, "soft_w_h", 50.0)), DTYPE)
    w_u = tf.cast(float(getattr(p, "soft_w_u", 50.0)), DTYPE)
    w_v = tf.cast(float(getattr(p, "soft_w_v", 50.0)), DTYPE)

    def soft_constraint_loss_fn(model):
        h, u, v = output_fn(model, x_b, y_b, "soft")
        loss_h = tf.reduce_mean(tf.square(h - h_b_tf))
        loss_u = tf.reduce_mean(tf.square(u - u_b_tf))
        loss_v = tf.reduce_mean(tf.square(v - v_b_tf))
        weighted = w_h * loss_h + w_u * loss_u + w_v * loss_v
        zero = tf.constant(0.0, dtype=DTYPE)
        return weighted, {"ic": zero, "bc": weighted, "bc_h": loss_h, "bc_u": loss_u, "bc_v": loss_v}

    def residual_fn(model, x, y, constraints: str):
        # Steady rotating shallow water on a flat bed:
        #   (h u)_x + (h v)_y = 0
        #   (h u^2 + g h^2 / 2)_x + (h u v)_y = f h v
        #   (h u v)_x + (h v^2 + g h^2 / 2)_y = -f h u
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            h, u, v = output_fn(model, x, y, constraints)
            hu = h * u
            hv = h * v
            flux_x = hu * u + 0.5 * G * h * h
            flux_y = hv * v + 0.5 * G * h * h
            shear = h * u * v
            cor_x = F * h * v
            cor_y = F * h * u
        mass = tape.gradient(hu, x) + tape.gradient(hv, y)
        mom_x = tape.gradient(flux_x, x) + tape.gradient(shear, y) - cor_x
        mom_y = tape.gradient(shear, x) + tape.gradient(flux_y, y) + cor_y
        del tape
        return mass, mom_x, mom_y

    return Problem(
        name="steady_vortex",
        domain=domain,
        residual_fn=residual_fn,
        output_fn=output_fn,
        exact_solution=exact_solution,
        soft_constraint_loss_fn=soft_constraint_loss_fn,
        n_outputs=3,
    )
