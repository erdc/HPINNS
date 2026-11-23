"""! @file swe_pinn/problems/tidal_test.py
@brief Built-in nondimensional linear tidal test with exact solution and optional hard ansatz.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from swe_pinn.physics.common import DTYPE, ONE, Domain1D, Problem, unit_interval

def build_tidal_test_problem(p):
    alpha = p.g * p.H * p.T_tide**2 / p.L**2
    beta = p.r * p.T_tide
    U = p.A * p.L / (p.H * p.T_tide)

    two_pi = 2.0 * np.pi
    kappa_sq = (two_pi**2 - 1j * two_pi * beta) / alpha
    kappa = np.sqrt(kappa_sq)
    denom = np.cos(kappa)
    vel_coeff = -(alpha * kappa) / (1j * two_pi + beta)

    kr, ki = float(np.real(kappa)), float(np.imag(kappa))
    dr, di = float(np.real(denom)), float(np.imag(denom))
    vc_r, vc_i = float(np.real(vel_coeff)), float(np.imag(vel_coeff))
    denom_abs_sq = dr * dr + di * di

    KR = tf.constant(kr, dtype=DTYPE)
    KI = tf.constant(ki, dtype=DTYPE)
    DR = tf.constant(dr, dtype=DTYPE)
    DI = tf.constant(di, dtype=DTYPE)
    VC_R = tf.constant(vc_r, dtype=DTYPE)
    VC_I = tf.constant(vc_i, dtype=DTYPE)
    DEN_ABS = tf.constant(denom_abs_sq, dtype=DTYPE)
    TWO_PI = tf.constant(two_pi, dtype=DTYPE)
    ALPHA = tf.constant(alpha, dtype=DTYPE)
    BETA = tf.constant(beta, dtype=DTYPE)

    domain = Domain1D(0.0, 1.0, 0.0, p.tau_max)

    def exact_solution(xi: tf.Tensor, tau: tf.Tensor):
        sxi = ONE - xi
        a = KR * sxi
        b = KI * sxi

        cos_a = tf.cos(a)
        sin_a = tf.sin(a)
        cosh_b = tf.cosh(b)
        sinh_b = tf.sinh(b)

        num_e_r = cos_a * cosh_b
        num_e_i = -sin_a * sinh_b

        sin_term_r = sin_a * cosh_b
        sin_term_i = cos_a * sinh_b

        e_r = (num_e_r * DR + num_e_i * DI) / DEN_ABS
        e_i = (num_e_i * DR - num_e_r * DI) / DEN_ABS

        vs_r = VC_R * sin_term_r - VC_I * sin_term_i
        vs_i = VC_R * sin_term_i + VC_I * sin_term_r

        v_r = (vs_r * DR + vs_i * DI) / DEN_ABS
        v_i = (vs_i * DR - vs_r * DI) / DEN_ABS

        s = tf.sin(TWO_PI * tau)
        c = tf.cos(TWO_PI * tau)

        eta = e_r * s + e_i * c
        u = v_r * s + v_i * c
        return eta, u

    def output_fn(model, x, t, constraints: str):
        # x is already on [0, 1]. t / tau_max puts time on [0, 1] as well.
        x_u = unit_interval(x, domain.x_min, domain.x_max)
        t_u = unit_interval(t, domain.t_min, domain.t_max)
        raw = model(tf.concat([x_u, t_u], axis=1))
        n_eta = raw[:, 0:1]
        n_u = raw[:, 1:2]
        if str(constraints).strip().lower() == "hard":
            # Match the exact tide on t = 0, x = 0, and x = 1 only. The
            # interior value is that boundary lift plus the network, so the
            # collocation loss is the PDE residual.
            eta_ic, u_ic = exact_solution(x, tf.zeros_like(t))
            eta_left, u_left = exact_solution(tf.zeros_like(x), t)
            eta_right, u_right = exact_solution(tf.ones_like(x), t)
            eta_left0, u_left0 = exact_solution(tf.zeros_like(x), tf.zeros_like(t))
            eta_right0, u_right0 = exact_solution(tf.ones_like(x), tf.zeros_like(t))
            eta_bc = (
                eta_ic
                + (eta_left - eta_left0) * (ONE - x)
                + (eta_right - eta_right0) * x
            )
            u_bc = (
                u_ic
                + (u_left - u_left0) * (ONE - x)
                + (u_right - u_right0) * x
            )
            vanish = t * x * (ONE - x)
            return eta_bc + vanish * n_eta, u_bc + vanish * n_u
        return n_eta, n_u

    def soft_constraint_loss_fn(model):
        """Soft IC/BC penalties against the analytic tidal solution."""
        n_ic = int(getattr(p, "soft_n_ic", 2048))
        n_bc = int(getattr(p, "soft_n_bc", 1024))

        x_ic = tf.random.uniform(
            (n_ic, 1), domain.x_min, domain.x_max, dtype=DTYPE
        )
        t_ic = tf.fill((n_ic, 1), tf.cast(domain.t_min, DTYPE))
        eta_ic_pred, u_ic_pred = output_fn(model, x_ic, t_ic, "soft")
        eta_ic_true, u_ic_true = exact_solution(x_ic, t_ic)

        loss_ic_eta = tf.reduce_mean(tf.square(eta_ic_pred - eta_ic_true))
        loss_ic_u = tf.reduce_mean(tf.square(u_ic_pred - u_ic_true))

        t_bc = tf.random.uniform(
            (n_bc, 1), domain.t_min, domain.t_max, dtype=DTYPE
        )
        x_left = tf.fill((n_bc, 1), tf.cast(domain.x_min, DTYPE))
        x_right = tf.fill((n_bc, 1), tf.cast(domain.x_max, DTYPE))

        eta_left_pred, u_left_pred = output_fn(model, x_left, t_bc, "soft")
        eta_right_pred, u_right_pred = output_fn(model, x_right, t_bc, "soft")
        eta_left_true, u_left_true = exact_solution(x_left, t_bc)
        eta_right_true, u_right_true = exact_solution(x_right, t_bc)

        loss_bc_eta = (
            tf.reduce_mean(tf.square(eta_left_pred - eta_left_true))
            + tf.reduce_mean(tf.square(eta_right_pred - eta_right_true))
        )
        loss_bc_u = (
            tf.reduce_mean(tf.square(u_left_pred - u_left_true))
            + tf.reduce_mean(tf.square(u_right_pred - u_right_true))
        )

        w_ic_eta = tf.cast(getattr(p, "soft_w_ic_eta", 20.0), DTYPE)
        w_ic_u = tf.cast(getattr(p, "soft_w_ic_u", 20.0), DTYPE)
        w_bc_eta = tf.cast(getattr(p, "soft_w_bc_eta", 10.0), DTYPE)
        w_bc_u = tf.cast(getattr(p, "soft_w_bc_u", 10.0), DTYPE)

        weighted = (
            w_ic_eta * loss_ic_eta
            + w_ic_u * loss_ic_u
            + w_bc_eta * loss_bc_eta
            + w_bc_u * loss_bc_u
        )
        loss_ic = w_ic_eta * loss_ic_eta + w_ic_u * loss_ic_u
        loss_bc = w_bc_eta * loss_bc_eta + w_bc_u * loss_bc_u
        components = {
            "ic": loss_ic,
            "bc": loss_bc,
            "ic_eta": loss_ic_eta,
            "ic_u": loss_ic_u,
            "bc_eta": loss_bc_eta,
            "bc_u": loss_bc_u,
        }
        return weighted, components

    def residual_fn(model, x, t, constraints: str):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            eta, u = output_fn(model, x, t, constraints)
        eta_t = tape.gradient(eta, t)
        eta_x = tape.gradient(eta, x)
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        del tape
        return eta_t + u_x, u_t + ALPHA * eta_x + BETA * u

    def to_dimensional(ETA, UHAT, X, T):
        XKM = X * p.L / 1000.0
        TIMEH = T * p.T_tide / 3600.0
        return p.A * ETA, U * UHAT, XKM, TIMEH

    return Problem(
        name="tidal_test",
        domain=domain,
        residual_fn=residual_fn,
        output_fn=output_fn,
        exact_solution=exact_solution,
        to_dimensional=to_dimensional,
        soft_constraint_loss_fn=soft_constraint_loss_fn,
    )
