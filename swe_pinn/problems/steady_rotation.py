"""! @file swe_pinn/problems/steady_rotation.py
@brief Steady rotating shallow-water flow with a parabolic free surface.

Exact solution of the steady, frictionless, non-rotating shallow-water
equations on a flat bed:
    u = -omega * y
    v =  omega * x
    h = h_center + omega**2 * (x**2 + y**2) / (2 g)

The trainer's second coordinate is y. There is no time axis.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from swe_pinn.physics.common import DTYPE, Domain1D, Problem, unit_interval


def build_steady_rotation_problem(p):
    G = tf.constant(p.g, dtype=DTYPE)
    OMEGA = tf.constant(p.omega, dtype=DTYPE)
    H_C = tf.constant(p.h_center, dtype=DTYPE)
    # The shared sampler calls the second axis t. Here that axis is y.
    domain = Domain1D(p.x_min, p.x_max, p.y_min, p.y_max)
    span_x = p.x_max - p.x_min
    span_y = p.y_max - p.y_min

    def exact_numpy(x, y):
        x_np = np.asarray(x, dtype=np.float64).reshape(-1)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        h = p.h_center + p.omega ** 2 * (x_np ** 2 + y_np ** 2) / (2.0 * p.g)
        u = -p.omega * y_np
        v = p.omega * x_np
        return h, u, v

    def _as_columns(h, u, v):
        return (
            tf.convert_to_tensor(h.reshape(-1, 1), dtype=DTYPE),
            tf.convert_to_tensor(u.reshape(-1, 1), dtype=DTYPE),
            tf.convert_to_tensor(v.reshape(-1, 1), dtype=DTYPE),
        )

    def exact_solution(x, y):
        return _as_columns(*exact_numpy(x, y))

    def exact_tf(x, y):
        h = H_C + (OMEGA * OMEGA) * (tf.square(x) + tf.square(y)) / (2.0 * G)
        u = -OMEGA * y
        v = OMEGA * x
        return h, u, v

    def output_fn(model, x, y, constraints: str):
        # The square is [-1, 1] x [-1, 1]. The network sees both axes on [0, 1].
        x_u = unit_interval(x, p.x_min, p.x_max)
        y_u = unit_interval(y, p.y_min, p.y_max)
        raw = model(tf.concat([x_u, y_u], axis=1))
        h_raw = raw[:, 0:1]
        u_raw = raw[:, 1:2]
        v_raw = raw[:, 2:3]
        if str(constraints).strip().lower() != "hard":
            return h_raw, u_raw, v_raw

        h_exact, u_exact, v_exact = exact_tf(x, y)
        sx = (x - p.x_min) / span_x
        sy = (y - p.y_min) / span_y
        bubble = sx * (1.0 - sx) * sy * (1.0 - sy)
        return (
            h_exact + bubble * h_raw,
            u_exact + bubble * u_raw,
            v_exact + bubble * v_raw,
        )

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
        # Steady conservative shallow water on a flat frictionless bed:
        #   (h u)_x + (h v)_y = 0
        #   (h u^2 + g h^2 / 2)_x + (h u v)_y = 0
        #   (h u v)_x + (h v^2 + g h^2 / 2)_y = 0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            h, u, v = output_fn(model, x, y, constraints)
            hu = h * u
            hv = h * v
            flux_x = hu * u + 0.5 * G * h * h
            flux_y = hv * v + 0.5 * G * h * h
            shear = h * u * v
        hu_x = tape.gradient(hu, x)
        hv_y = tape.gradient(hv, y)
        flux_x_x = tape.gradient(flux_x, x)
        shear_y = tape.gradient(shear, y)
        shear_x = tape.gradient(shear, x)
        flux_y_y = tape.gradient(flux_y, y)
        del tape
        return hu_x + hv_y, flux_x_x + shear_y, shear_x + flux_y_y

    return Problem(
        name="steady_rotation",
        domain=domain,
        residual_fn=residual_fn,
        output_fn=output_fn,
        exact_solution=exact_solution,
        soft_constraint_loss_fn=soft_constraint_loss_fn,
        n_outputs=3,
    )
