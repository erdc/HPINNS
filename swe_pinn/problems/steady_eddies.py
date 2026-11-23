"""! @file swe_pinn/problems/steady_eddies.py
@brief Lattice of Fourier eddies on a carved bed, with an exact solution.

Steady shallow water in a square basin. The stream function is a degenerate
Fourier pair

    psi = A [sin(m pi x / L) sin(n pi y / L) + sin(n pi x / L) sin(m pi y / L)]

with velocity (psi_y, -psi_x) and depth H0 + gamma * psi. Both mode pairs
share the eigenvalue mu = (m^2 + n^2) pi^2 / L^2, so the vorticity is
proportional to psi. The free surface that balances advection is

    eta = C - |u|^2 / (2 g) - mu psi^2 / (2 g)

and the bed is z = eta - h. Continuity and momentum hold identically. There
is no manufactured body force and no shock. The walls are slip walls: depth
equals H0 and the normal velocity vanishes.

The hard constraint lifts the boundary traces only. It does not insert the
exact interior solution.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from swe_pinn.physics.common import DTYPE, Domain1D, Problem, unit_interval


def build_steady_eddies_problem(p):
    G = tf.constant(p.g, dtype=DTYPE)
    A = tf.constant(p.amplitude, dtype=DTYPE)
    H0 = tf.constant(p.h_mean, dtype=DTYPE)
    GAMMA = tf.constant(p.gamma, dtype=DTYPE)
    C0 = tf.constant(p.bernoulli_c, dtype=DTYPE)
    M = int(p.mode_m)
    N = int(p.mode_n)
    if M < 1 or N < 1:
        raise ValueError("steady_eddies mode_m and mode_n must be >= 1")
    span_x = float(p.x_max - p.x_min)
    span_y = float(p.y_max - p.y_min)
    if span_x <= 0.0 or span_y <= 0.0:
        raise ValueError("steady_eddies domain spans must be positive")
    if abs(span_x - span_y) > 1.0e-12 * max(span_x, span_y):
        raise ValueError(
            "steady_eddies requires a square domain so the mode pair "
            f"({M}, {N}) shares one eigenvalue; got Lx={span_x}, Ly={span_y}"
        )
    L = span_x
    PI_OVER_L = tf.constant(np.pi / L, dtype=DTYPE)
    MU = tf.constant((M * M + N * N) * (np.pi / L) ** 2, dtype=DTYPE)
    domain = Domain1D(p.x_min, p.x_max, p.y_min, p.y_max)

    def _stream(x, y):
        """psi and its first derivatives from the degenerate Fourier pair."""
        sx = (x - tf.cast(p.x_min, DTYPE)) * PI_OVER_L
        sy = (y - tf.cast(p.y_min, DTYPE)) * PI_OVER_L
        s_mx = tf.sin(tf.cast(M, DTYPE) * sx)
        c_mx = tf.cos(tf.cast(M, DTYPE) * sx)
        s_nx = tf.sin(tf.cast(N, DTYPE) * sx)
        c_nx = tf.cos(tf.cast(N, DTYPE) * sx)
        s_my = tf.sin(tf.cast(M, DTYPE) * sy)
        c_my = tf.cos(tf.cast(M, DTYPE) * sy)
        s_ny = tf.sin(tf.cast(N, DTYPE) * sy)
        c_ny = tf.cos(tf.cast(N, DTYPE) * sy)
        psi = A * (s_mx * s_ny + s_nx * s_my)
        psi_x = A * PI_OVER_L * (
            tf.cast(M, DTYPE) * c_mx * s_ny + tf.cast(N, DTYPE) * c_nx * s_my
        )
        psi_y = A * PI_OVER_L * (
            tf.cast(N, DTYPE) * s_mx * c_ny + tf.cast(M, DTYPE) * s_nx * c_my
        )
        return psi, psi_x, psi_y

    def exact_tf(x, y):
        psi, psi_x, psi_y = _stream(x, y)
        u = psi_y
        v = -psi_x
        h = H0 + GAMMA * psi
        return h, u, v

    def bed_tf(x, y):
        """Bed elevation that makes (h, u, v) an exact steady solution."""
        psi, psi_x, psi_y = _stream(x, y)
        u = psi_y
        v = -psi_x
        h = H0 + GAMMA * psi
        speed2 = u * u + v * v
        eta = C0 - speed2 / (2.0 * G) - MU * psi * psi / (2.0 * G)
        return eta - h

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
    # Slip walls: depth equals the mean, and only the normal velocity is fixed.
    # Tangential speed on the wall is left free so the soft penalty does not
    # dominate the Optuna score.
    x_s = tf.constant(xs.reshape(-1, 1), dtype=DTYPE)
    y_s = tf.constant(np.full((n_edge, 1), p.y_min), dtype=DTYPE)
    x_n = tf.constant(xs.reshape(-1, 1), dtype=DTYPE)
    y_n = tf.constant(np.full((n_edge, 1), p.y_max), dtype=DTYPE)
    x_w = tf.constant(np.full((n_edge, 1), p.x_min), dtype=DTYPE)
    y_w = tf.constant(ys.reshape(-1, 1), dtype=DTYPE)
    x_e = tf.constant(np.full((n_edge, 1), p.x_max), dtype=DTYPE)
    y_e = tf.constant(ys.reshape(-1, 1), dtype=DTYPE)
    h_wall = tf.constant(p.h_mean, dtype=DTYPE)
    w_h = tf.cast(float(getattr(p, "soft_w_h", 50.0)), DTYPE)
    w_u = tf.cast(float(getattr(p, "soft_w_u", 50.0)), DTYPE)
    w_v = tf.cast(float(getattr(p, "soft_w_v", 50.0)), DTYPE)

    def soft_constraint_loss_fn(model):
        h_s, u_s, v_s = output_fn(model, x_s, y_s, "soft")
        h_n, u_n, v_n = output_fn(model, x_n, y_n, "soft")
        h_w, u_w, v_w = output_fn(model, x_w, y_w, "soft")
        h_e, u_e, v_e = output_fn(model, x_e, y_e, "soft")
        loss_h = (
            tf.reduce_mean(tf.square(h_s - h_wall))
            + tf.reduce_mean(tf.square(h_n - h_wall))
            + tf.reduce_mean(tf.square(h_w - h_wall))
            + tf.reduce_mean(tf.square(h_e - h_wall))
        ) / 4.0
        # South/north: normal is v. West/east: normal is u.
        loss_n = (
            tf.reduce_mean(tf.square(v_s))
            + tf.reduce_mean(tf.square(v_n))
            + tf.reduce_mean(tf.square(u_w))
            + tf.reduce_mean(tf.square(u_e))
        ) / 4.0
        weighted = w_h * loss_h + 0.5 * (w_u + w_v) * loss_n
        zero = tf.constant(0.0, dtype=DTYPE)
        return weighted, {
            "ic": zero,
            "bc": weighted,
            "bc_h": loss_h,
            "bc_u": loss_n,
            "bc_v": loss_n,
        }

    def residual_fn(model, x, y, constraints: str):
        # Steady shallow water with bathymetry and no friction:
        #   (h u)_x + (h v)_y = 0
        #   (h u^2 + g h^2 / 2)_x + (h u v)_y + g h z_x = 0
        #   (h u v)_x + (h v^2 + g h^2 / 2)_y + g h z_y = 0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            h, u, v = output_fn(model, x, y, constraints)
            z = bed_tf(x, y)
            hu = h * u
            hv = h * v
            flux_x = hu * u + 0.5 * G * h * h
            flux_y = hv * v + 0.5 * G * h * h
            shear = h * u * v
        mass = tape.gradient(hu, x) + tape.gradient(hv, y)
        mom_x = (
            tape.gradient(flux_x, x)
            + tape.gradient(shear, y)
            + G * h * tape.gradient(z, x)
        )
        mom_y = (
            tape.gradient(shear, x)
            + tape.gradient(flux_y, y)
            + G * h * tape.gradient(z, y)
        )
        del tape
        return mass, mom_x, mom_y

    problem = Problem(
        name="steady_eddies",
        domain=domain,
        residual_fn=residual_fn,
        output_fn=output_fn,
        exact_solution=exact_solution,
        soft_constraint_loss_fn=soft_constraint_loss_fn,
        n_outputs=3,
    )
    problem.bed_elevation = bed_tf
    return problem
