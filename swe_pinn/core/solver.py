"""! @file swe_pinn/core/solver.py
@brief Main training, selective Optuna search, optional L-BFGS refinement,
@brief evaluation, and parameter reporting.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import time
import numpy as np
import tensorflow as tf
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from scipy.optimize import minimize

from swe_pinn.problems import build_problem
from swe_pinn.physics.common import (
    DTYPE,
    EMBEDDINGS,
    MODEL_FAMILIES,
    PINN,
    sample_interior,
    sample_interior_grid,
    residuals,
    outputs,
    normalized_relative_weights,
)
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.set_floatx("float64")
tf.get_logger().setLevel("ERROR")


@dataclass
class HyperParams:
    width: int
    depth: int
    lr: float
    w_mass: float
    w_mom: float
    log_w_mass: float = 0.0
    log_w_mom: float = 0.0
    model_family: str = "classical"
    n_qubits: int = 0
    q_layers: int = 0
    embedding: str = "once"


def build_model(hp: HyperParams, n_outputs: int = 2):
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)
    model = PINN(
        width=hp.width,
        depth=hp.depth,
        model_family=hp.model_family,
        n_qubits=hp.n_qubits,
        q_layers=hp.q_layers,
        embedding=hp.embedding,
        n_outputs=int(n_outputs),
    )
    model(tf.zeros((1, 2), dtype=DTYPE))
    return model


def _normalize_constraints(constraints) -> str:
    """Normalize constraint mode to lowercase 'hard' or 'soft'."""
    mode = str(constraints).strip().lower()
    if mode not in ("hard", "soft"):
        raise ValueError(
            f"Unknown constraints={constraints!r}. Use 'hard' or 'soft'."
        )
    return mode


def compute_losses(model, problem, n_f, constraints, w_mass, w_mom, *, x=None, t=None):
    """Compute PDE loss and, for soft mode, explicit IC/BC penalties.

    Optional ``x`` / ``t`` reuse fixed collocation points (e.g. L-BFGS).

    Returns
    -------
    total, loss_mass, loss_mom, loss_constraints, loss_ic, loss_bc
    """
    constraints = _normalize_constraints(constraints)
    if x is None or t is None:
        x, t = sample_interior(problem.domain, tf.constant(n_f, dtype=tf.int32))
    raw = residuals(model, problem, x, t, constraints)
    if len(raw) == 2:
        r_mass, r_mom = raw
        loss_mass = tf.reduce_mean(tf.square(r_mass))
        loss_mom = tf.reduce_mean(tf.square(r_mom))
    elif len(raw) == 3:
        r_mass, r_mx, r_my = raw
        loss_mass = tf.reduce_mean(tf.square(r_mass))
        loss_mom = (
            tf.reduce_mean(tf.square(r_mx)) + tf.reduce_mean(tf.square(r_my))
        )
    else:
        raise ValueError(
            f"Problem '{problem.name}' returned {len(raw)} residuals; expected 2 or 3."
        )
    loss_constraints = tf.constant(0.0, dtype=DTYPE)
    loss_ic = tf.constant(0.0, dtype=DTYPE)
    loss_bc = tf.constant(0.0, dtype=DTYPE)

    if constraints == "soft":
        if problem.soft_constraint_loss_fn is None:
            raise ValueError(
                f"Problem '{problem.name}' does not define soft IC/BC losses. "
                "Soft constraints cannot be used safely."
            )
        loss_constraints, components = problem.soft_constraint_loss_fn(model)
        if "ic" in components:
            loss_ic = components["ic"]
        if "bc" in components:
            loss_bc = components["bc"]

    total = (
        tf.cast(w_mass, DTYPE) * loss_mass
        + tf.cast(w_mom, DTYPE) * loss_mom
        + loss_constraints
    )
    return total, loss_mass, loss_mom, loss_constraints, loss_ic, loss_bc


def _format_model_banner(label: str, model, hp, total_params, classical_params, quantum_params) -> str:
    return (
        f"[{label}] "
        f"model_family={model.model_family}, "
        f"width={hp.width}, "
        f"depth={hp.depth}, "
        f"n_qubits={model.n_qubits}, "
        f"q_layers={model.q_layers}, "
        f"embedding={getattr(model, 'embedding', 'once')}, "
        f"classical_params={classical_params}, "
        f"quantum_params={quantum_params}, "
        f"total_params={total_params}"
    )


def _print_epoch_header(stage_label: str, banner: str | None = None):
    print("-" * 98)
    if banner:
        print(banner)
    else:
        print(f"[{stage_label}]")
    print("-" * 98)
    print(
        f"{'epoch':>8}  {'total loss':>14}  {'mass loss':>14}  "
        f"{'mom loss':>14}  {'IC data':>14}  {'BC data':>14}"
    )
    print("-" * 98)


def _maybe_print_epoch_row(
    epoch,
    total,
    mass,
    mom,
    ic_data,
    bc_data,
    *,
    stage_label,
    show_output,
    every,
    header_printed,
    banner=None,
):
    if not show_output:
        return header_printed
    every = max(1, int(every))
    should_print = epoch == 1 or epoch % every == 0
    if not should_print:
        return header_printed
    if not header_printed:
        _print_epoch_header(stage_label, banner=banner)
        header_printed = True
    print(
        f"{epoch:8d}  {total:14.6e}  {mass:14.6e}  {mom:14.6e}  "
        f"{ic_data:14.6e}  {bc_data:14.6e}"
    )
    return header_printed


def train_adam(
    model,
    problem,
    hp,
    epochs,
    constraints,
    *,
    nx,
    nt,
    show_epoch_output=False,
    epoch_output_every=10,
    stage_label="Adam",
    banner=None,
    fixed_collocation=True,
    collocation_grid=True,
    # 2_dam_break uses a constant Adam LR for the full run. Exponential
    # decay freezes this dam-break soft PINN near the ~0.4 loss plateau.
    lr_decay_steps=None,
    lr_decay_rate=None,
    # Hybrid: train as pure classical for this many epochs (no QNode eval),
    # then enable the quantum residual. Escapes the soft ~0.4 plateau first
    # without paying sample-by-sample PennyLane cost every step.
    hybrid_classical_warmup_epochs=0,
):
    if lr_decay_rate is None or float(lr_decay_rate) >= 1.0:
        opt = tf.keras.optimizers.Adam(learning_rate=float(hp.lr))
    else:
        decay_steps = int(lr_decay_steps) if lr_decay_steps is not None else max(epochs // 4, 1)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(hp.lr),
            decay_steps=max(int(decay_steps), 1),
            decay_rate=float(lr_decay_rate),
            staircase=True,
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    hist = {
        "total": [],
        "mass": [],
        "mom": [],
        "constraints": [],
        "ic": [],
        "bc": [],
    }
    epoch_header_printed = False
    # Fixed interior points match 2_dam_break (stable PDE residual each epoch).
    x_fixed = t_fixed = None
    n_f = int(nx) * int(nt)
    if fixed_collocation:
        if collocation_grid:
            x_fixed, t_fixed = sample_interior_grid(problem.domain, int(nx), int(nt))
        else:
            x_fixed, t_fixed = sample_interior(
                problem.domain, tf.constant(n_f, dtype=tf.int32)
            )

    is_hybrid = (
        str(getattr(model, "model_family", "")) == "hybrid_quantum"
        and int(getattr(model, "n_qubits", 0)) > 0
    )
    warmup = int(hybrid_classical_warmup_epochs) if is_hybrid else 0
    warmup = max(0, min(warmup, int(epochs)))
    if is_hybrid and hasattr(model, "quantum_enabled"):
        # Ensure quantum weights exist before warmup so the optimizer is built
        # against the full variable set (Keras 3 rejects newly seen vars later).
        model.quantum_enabled = True
        _ = model(tf.zeros((1, 2), dtype=DTYPE))
        model.quantum_enabled = warmup <= 0
        if warmup > 0:
            print(
                f"[{stage_label}] hybrid classical warmup: "
                f"epochs 1–{warmup} without QNode; "
                f"quantum residual enabled from epoch {warmup + 1}"
            )

    # Build Adam against every trainable variable up front (classical + quantum).
    opt.build(model.trainable_variables)

    for epoch in range(1, epochs + 1):
        if is_hybrid and hasattr(model, "quantum_enabled"):
            model.quantum_enabled = epoch > warmup

        with tf.GradientTape() as tape:
            total, mass, mom, constraint_loss, ic_loss, bc_loss = compute_losses(
                model,
                problem,
                n_f,
                constraints,
                hp.w_mass,
                hp.w_mom,
                x=x_fixed,
                t=t_fixed,
            )
        grads = tape.gradient(total, model.trainable_variables)
        # Keep the full variable list every step. Warmup skips the QNode, so
        # quantum grads are None — replace with zeros so Keras 3 Adam does not
        # rebuild around a changing variable set when quantum turns on.
        gv = [
            (g if g is not None else tf.zeros_like(v), v)
            for g, v in zip(grads, model.trainable_variables)
        ]
        opt.apply_gradients(gv)

        total_f = float(total.numpy())
        mass_f = float(mass.numpy())
        mom_f = float(mom.numpy())
        ic_f = float(ic_loss.numpy())
        bc_f = float(bc_loss.numpy())
        hist["total"].append(total_f)
        hist["mass"].append(mass_f)
        hist["mom"].append(mom_f)
        hist["constraints"].append(float(constraint_loss.numpy()))
        hist["ic"].append(ic_f)
        hist["bc"].append(bc_f)

        epoch_header_printed = _maybe_print_epoch_row(
            epoch,
            total_f,
            mass_f,
            mom_f,
            ic_f,
            bc_f,
            stage_label=stage_label,
            show_output=show_epoch_output,
            every=epoch_output_every,
            header_printed=epoch_header_printed,
            banner=banner,
        )
    return hist


def _pack_variables(variables):
    return np.concatenate(
        [tf.reshape(v, [-1]).numpy() for v in variables], axis=0
    ).astype(np.float64)


def _unpack_variables_1d(vector, variables):
    idx = 0
    for var in variables:
        size = int(np.prod(var.shape))
        new_val = vector[idx: idx + size].reshape(tuple(var.shape))
        var.assign(tf.convert_to_tensor(new_val, dtype=DTYPE))
        idx += size


def train_lbfgs(model, problem, hp, nx, nt, constraints, max_iters, ftol, gtol, maxfun):
    constraints = _normalize_constraints(constraints)
    x_fixed, t_fixed = sample_interior_grid(problem.domain, int(nx), int(nt))
    n_f = int(nx) * int(nt)
    variables = model.trainable_variables
    hist = {"total": [], "mass": [], "mom": [], "constraints": []}

    def loss_and_grad(vector):
        _unpack_variables_1d(vector, variables)
        # Use the same total loss as Adam, including soft IC/BC penalties.
        with tf.GradientTape() as tape:
            total, loss_mass, loss_mom, loss_constraints, _, _ = compute_losses(
                model,
                problem,
                n_f,
                constraints,
                hp.w_mass,
                hp.w_mom,
                x=x_fixed,
                t=t_fixed,
            )

        grads = tape.gradient(total, variables)
        grad_vec = np.concatenate(
            [
                tf.reshape(g if g is not None else tf.zeros_like(v), [-1]).numpy()
                for g, v in zip(grads, variables)
            ],
            axis=0,
        ).astype(np.float64)

        hist["total"].append(float(total.numpy()))
        hist["mass"].append(float(loss_mass.numpy()))
        hist["mom"].append(float(loss_mom.numpy()))
        hist["constraints"].append(float(loss_constraints.numpy()))
        return float(total.numpy()), grad_vec

    x0 = _pack_variables(variables)
    result = minimize(
        fun=loss_and_grad,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iters),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxfun": int(maxfun),
            "maxls": 50,
        },
    )
    _unpack_variables_1d(result.x, variables)

    return {
        "history": hist,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", -1)),
        "nfev": int(getattr(result, "nfev", -1)),
        "used": True,
    }


def _normalized_tune_set(cfg):
    return {str(v).strip() for v in getattr(cfg, "tune_variables", ()) if str(v).strip()}


def _choose_value_or_suggest(trial, name, fixed_value, suggest_fn, tune_set):
    if name in tune_set:
        return suggest_fn()
    return fixed_value


def _trial_param(trial, name, default):
    """Read a completed trial parameter, falling back to user attrs then default."""
    params = getattr(trial, "params", None) or {}
    if name in params:
        return params[name]
    attrs = getattr(trial, "user_attrs", None) or {}
    if name in attrs:
        return attrs[name]
    return default


def _hyperparams_from_completed_trial(cfg, trial):
    """Rebuild HyperParams from a finished Optuna trial's recorded values.

    Uses ``trial.params`` (and user attrs) rather than the current
    ``tune_variables`` set, so reloading a study after changing which
    variables are tuned still recovers the winning architecture.
    """
    model_family = str(
        _trial_param(trial, "model_family", cfg.model_family)
    )
    if model_family == "full_quantum":
        # Pure quantum does not use a classical backbone.
        width = 0
        depth = 0
    else:
        width = int(_trial_param(trial, "width", cfg.width_value))
        depth = int(_trial_param(trial, "depth", cfg.depth_value))
    lr = float(_trial_param(trial, "lr", cfg.lr_value))
    log_w_mass = float(_trial_param(trial, "log_w_mass", cfg.log_w_mass_value))
    log_w_mom = float(_trial_param(trial, "log_w_mom", cfg.log_w_mom_value))

    n_qubits = 0
    q_layers = 0
    if model_family in ("hybrid_quantum", "full_quantum"):
        n_qubits = int(
            _trial_param(trial, "n_qubits", cfg.n_qubits_value)
        )
        q_layers = int(
            _trial_param(trial, "q_layers", cfg.q_layers_value)
        )
        if model_family == "hybrid_quantum" and n_qubits <= 0:
            q_layers = 0
            model_family = "classical"
            n_qubits = 0
        elif model_family == "full_quantum" and n_qubits < 1:
            raise ValueError(
                "Completed full_quantum trial is missing n_qubits >= 1"
            )
        elif model_family == "full_quantum" and q_layers < 1:
            raise ValueError(
                "Completed full_quantum trial is missing q_layers >= 1"
            )
    elif model_family != "classical":
        raise ValueError(
            f"Unknown model_family {model_family!r} on completed trial. "
            f"Use one of {MODEL_FAMILIES}."
        )

    embedding = _embedding_for_family(cfg, trial, model_family)
    w_mass, w_mom = normalized_relative_weights(log_w_mass, log_w_mom)
    return HyperParams(
        width=width,
        depth=depth,
        lr=lr,
        w_mass=w_mass,
        w_mom=w_mom,
        log_w_mass=float(log_w_mass),
        log_w_mom=float(log_w_mom),
        model_family=str(model_family),
        n_qubits=n_qubits,
        q_layers=q_layers,
        embedding=embedding,
    )


def _normalize_embedding(value) -> str:
    name = str(value).strip().lower()
    if name not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding {value!r}. Use one of {EMBEDDINGS}.")
    return name


def _embedding_options(cfg):
    """Circuit layouts a quantum trial may draw."""
    raw = _cfg(cfg, "embedding_options", EMBEDDINGS)
    options = tuple(_normalize_embedding(v) for v in raw)
    if not options:
        raise ValueError("embedding_options must not be empty")
    return options


def _embedding_for_family(cfg, trial, model_family) -> str:
    """Layout recorded on a finished trial.

    A trial that predates this choice used one embedding. A classical trial
    has no circuit, so the layout is ``once``.
    """
    if str(model_family) not in ("hybrid_quantum", "full_quantum"):
        return "once"
    if trial is None:
        return _normalize_embedding(_cfg(cfg, "embedding_value", "once"))
    return _normalize_embedding(_trial_param(trial, "embedding", "once"))


def _cfg(cfg, name, default):
    """Read an optional config value with a backwards-compatible default."""
    return getattr(cfg, name, default)


def _model_family_options(cfg):
    """Categorical families Optuna may pick.

    Defaults to every known family. ``user_input.model_family_options`` can
    drop one (for example to search only classical and hybrid).
    """
    raw = _cfg(cfg, "model_family_options", MODEL_FAMILIES)
    options = tuple(str(v) for v in raw)
    if not options:
        raise ValueError("model_family_options must not be empty")
    unknown = [name for name in options if name not in MODEL_FAMILIES]
    if unknown:
        raise ValueError(
            f"Unknown model_family_options {unknown}. Use names from {MODEL_FAMILIES}."
        )
    return options


def _uses_qnode(model_family, n_qubits) -> bool:
    """True when a forward pass evaluates the PennyLane circuit."""
    family = str(model_family)
    qubits = int(n_qubits)
    if family == "full_quantum":
        return True
    return family == "hybrid_quantum" and qubits > 0


def _is_hybrid(model_family, n_qubits) -> bool:
    return str(model_family) == "hybrid_quantum" and int(n_qubits) > 0


def _family_range(cfg, family, name, default):
    """Return a family-specific range value when configured.

    Example optional names:
        classical_width_min, hybrid_width_min
        classical_depth_max, hybrid_depth_max
    """
    if family == "hybrid_quantum":
        return _cfg(cfg, f"hybrid_{name}", default)
    if family == "classical":
        return _cfg(cfg, f"classical_{name}", default)
    return default


def _trial_hyperparams(cfg, trial=None):
    tune_set = _normalized_tune_set(cfg)

    model_family = str(cfg.model_family)
    if model_family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model_family {model_family!r}. Use one of {MODEL_FAMILIES}."
        )
    if trial is not None:
        family_options = _model_family_options(cfg)
        model_family = str(_choose_value_or_suggest(
            trial,
            "model_family",
            cfg.model_family,
            lambda: trial.suggest_categorical("model_family", list(family_options)),
            tune_set,
        ))

        # Pure quantum has no classical backbone, so width/depth are not sampled.
        if model_family == "full_quantum":
            width = 0
            depth = 0
        else:
            use_family_ranges = bool(_cfg(cfg, "use_family_specific_search_ranges", False))

            if use_family_ranges:
                width_min = int(_family_range(cfg, model_family, "width_min", cfg.width_min))
                width_max = int(_family_range(cfg, model_family, "width_max", cfg.width_max))
                width_step = int(_family_range(cfg, model_family, "width_step", cfg.width_step))
                depth_min = int(_family_range(cfg, model_family, "depth_min", cfg.depth_min))
                depth_max = int(_family_range(cfg, model_family, "depth_max", cfg.depth_max))
            else:
                width_min = int(cfg.width_min)
                width_max = int(cfg.width_max)
                width_step = int(cfg.width_step)
                depth_min = int(cfg.depth_min)
                depth_max = int(cfg.depth_max)

            width = int(_choose_value_or_suggest(
                trial, "width", cfg.width_value,
                lambda: trial.suggest_int("width", width_min, width_max, step=width_step),
                tune_set,
            ))
            depth = int(_choose_value_or_suggest(
                trial, "depth", cfg.depth_value,
                lambda: trial.suggest_int("depth", depth_min, depth_max),
                tune_set,
            ))
        lr = float(_choose_value_or_suggest(
            trial, "lr", cfg.lr_value,
            lambda: trial.suggest_float("lr", cfg.lr_min, cfg.lr_max, log=True),
            tune_set,
        ))
        log_w_mass = float(_choose_value_or_suggest(
            trial, "log_w_mass", cfg.log_w_mass_value,
            lambda: trial.suggest_float("log_w_mass", cfg.log_w_mass_min, cfg.log_w_mass_max),
            tune_set,
        ))
        log_w_mom = float(_choose_value_or_suggest(
            trial, "log_w_mom", cfg.log_w_mom_value,
            lambda: trial.suggest_float("log_w_mom", cfg.log_w_mom_min, cfg.log_w_mom_max),
            tune_set,
        ))
    else:
        if model_family == "full_quantum":
            width = 0
            depth = 0
        else:
            width = int(cfg.width_value)
            depth = int(cfg.depth_value)
        lr = float(cfg.lr_value)
        log_w_mass = float(cfg.log_w_mass_value)
        log_w_mom = float(cfg.log_w_mom_value)

    n_qubits = 0
    q_layers = 0
    if model_family in ("hybrid_quantum", "full_quantum"):
        if trial is not None:
            n_qubits = int(_choose_value_or_suggest(
                trial, "n_qubits", cfg.n_qubits_value,
                lambda: trial.suggest_int("n_qubits", cfg.n_qubits_min, cfg.n_qubits_max),
                tune_set,
            ))

            # n_qubits == 0 is treated as classical for hybrid only (legacy).
            # Prefer n_qubits_min >= 1 so quantum trials actually use a circuit.
            # full_quantum cannot fall back to a classical trunk.
            if model_family == "hybrid_quantum" and n_qubits <= 0:
                q_layers = 0
                model_family = "classical"
                n_qubits = 0
            else:
                if model_family == "full_quantum" and n_qubits < 1:
                    raise ValueError(
                        "full_quantum requires n_qubits >= 1. "
                        "Set n_qubits_min >= 1 (and n_qubits_value >= 1)."
                    )
                q_layers = int(_choose_value_or_suggest(
                    trial, "q_layers", cfg.q_layers_value,
                    lambda: trial.suggest_int("q_layers", cfg.q_layers_min, cfg.q_layers_max),
                    tune_set,
                ))
                if model_family == "full_quantum" and q_layers < 1:
                    raise ValueError(
                        "full_quantum requires q_layers >= 1. "
                        "Set q_layers_min >= 1 (and q_layers_value >= 1)."
                    )
        else:
            n_qubits = int(cfg.n_qubits_value)
            q_layers = int(cfg.q_layers_value)
            if model_family == "hybrid_quantum" and n_qubits <= 0:
                q_layers = 0
                model_family = "classical"
                n_qubits = 0
            elif model_family == "full_quantum" and n_qubits < 1:
                raise ValueError("full_quantum requires n_qubits_value >= 1")
            elif model_family == "full_quantum" and q_layers < 1:
                raise ValueError("full_quantum requires q_layers_value >= 1")

    embedding = "once"
    if model_family in ("hybrid_quantum", "full_quantum"):
        fixed_embedding = _normalize_embedding(_cfg(cfg, "embedding_value", "once"))
        if trial is None:
            embedding = fixed_embedding
        else:
            embedding = _normalize_embedding(_choose_value_or_suggest(
                trial,
                "embedding",
                fixed_embedding,
                lambda: trial.suggest_categorical(
                    "embedding", list(_embedding_options(cfg))
                ),
                tune_set,
            ))

    w_mass, w_mom = normalized_relative_weights(log_w_mass, log_w_mom)
    return HyperParams(
        width=width,
        depth=depth,
        lr=lr,
        w_mass=w_mass,
        w_mom=w_mom,
        log_w_mass=float(log_w_mass),
        log_w_mom=float(log_w_mom),
        model_family=str(model_family),
        n_qubits=n_qubits,
        q_layers=q_layers,
        embedding=embedding,
    )

def _screen(message: str) -> None:
    """Write one line to the terminal, even when stdout is a log file."""
    line = message if message.endswith("\n") else message + "\n"
    try:
        with open("/dev/tty", "a", encoding="utf-8") as tty:
            tty.write(line)
            tty.flush()
    except OSError:
        print(line, end="", flush=True)


def _trial_progress(trial, cap: int) -> str:
    done = sum(1 for t in trial.study.trials if t.state == TrialState.COMPLETE) + 1
    return f"{done}/{int(cap)}"


def _show_embedding_column(cfg) -> bool:
    """Show the layout column when a trial can use either circuit."""
    if "embedding" in _normalized_tune_set(cfg):
        return True
    return str(getattr(cfg, "embedding_value", "once")) != "once"


def _screen_trial_table(
    row: str, *, show_lr: bool, show_log_weights: bool, show_embedding: bool = False
) -> None:
    """Write one trial row. The header is printed once for the whole search."""
    parts = []
    if _claim_trial_table_header():
        header = _trial_table_header(show_lr, show_log_weights, show_embedding)
        rule = "-" * len(header)
        parts.extend((rule, header, rule))
    parts.append(row)
    _screen("\n".join(parts))


def _claim_trial_table_header() -> bool:
    """True for the first finished trial of this launch, across all workers."""
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "trial_table_rows")
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            text = handle.read().strip()
            count = int(text) if text else 0
            count += 1
            handle.seek(0)
            handle.truncate()
            handle.write(str(count))
            handle.flush()
            return count == 1
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _log_weights_are_tuned(cfg) -> bool:
    """Show the log-weight columns only when they can differ from zero.

    A fixed pair of zeros leaves the PDE weights at (1, 1) and is omitted.
    Sampling either log-weight, or fixing either one away from zero, keeps
    both columns.
    """
    tune_set = _normalized_tune_set(cfg)
    if "log_w_mass" in tune_set or "log_w_mom" in tune_set:
        return True
    return (
        float(getattr(cfg, "log_w_mass_value", 0.0)) != 0.0
        or float(getattr(cfg, "log_w_mom_value", 0.0)) != 0.0
    )


def _trial_table_header(show_lr: bool, show_log_weights: bool, show_embedding: bool = False) -> str:
    lr_col = f"  {'lr':>9}" if show_lr else ""
    log_cols = ""
    if show_log_weights:
        log_cols = f"  {'log_w_mass':>10}  {'log_w_mom':>9}"
    embed_col = f"  {'embed':>8}" if show_embedding else ""
    return (
        f"{'trial':>6}  {'family':<14}  {'width':>5}  {'depth':>5}  "
        f"{'qubits':>6}  {'layers':>6}{embed_col}{lr_col}{log_cols}  "
        f"{'classical':>9}  {'quantum':>7}  {'total':>9}  "
        f"{'mass':>12}  {'mom':>12}  {'ic':>12}  {'bc':>12}  "
        f"{'objective':>12}  {'minutes':>8}  {'done':>9}"
    )


def _trial_table_row(
    trial_number,
    family,
    width,
    depth,
    qubits,
    layers,
    classical,
    quantum,
    total,
    objective,
    done,
    *,
    mass,
    mom,
    ic,
    bc,
    lr=None,
    log_w_mass=None,
    log_w_mom=None,
    embedding="-",
    minutes="-",
    show_lr=False,
    show_log_weights=False,
    show_embedding=False,
) -> str:
    lr_col = f"  {lr:>9}" if show_lr else ""
    log_cols = ""
    if show_log_weights:
        log_cols = f"  {log_w_mass:>10}  {log_w_mom:>9}"
    embed_col = f"  {str(embedding):>8}" if show_embedding else ""
    return (
        f"{int(trial_number):6d}  {str(family):<14}  {width:>5}  {depth:>5}  "
        f"{qubits:>6}  {layers:>6}{embed_col}{lr_col}{log_cols}  "
        f"{classical:>9}  {quantum:>7}  {total:>9}  "
        f"{mass:>12}  {mom:>12}  {ic:>12}  {bc:>12}  "
        f"{objective:>12}  {minutes:>8}  {done:>9}"
    )


def _trial_minutes(started: float) -> str:
    """Wall-clock minutes since ``started``, from ``time.perf_counter``."""
    return f"{(time.perf_counter() - started) / 60.0:.1f}"


def objective_factory(cfg, problem):
    def objective(trial):
        started = time.perf_counter()
        try:
            return _run_optuna_trial(cfg, problem, trial, started)
        except optuna.TrialPruned:
            raise
        except Exception:
            import traceback
            print(f"[Trial {trial.number}] FAILED with exception:")
            traceback.print_exc()
            show_lr = "lr" in _normalized_tune_set(cfg)
            show_log_weights = _log_weights_are_tuned(cfg)
            show_embedding = _show_embedding_column(cfg)
            _screen_trial_table(
                _trial_table_row(
                    trial.number,
                    "FAILED",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    _trial_progress(trial, cfg.n_trials),
                    mass="-",
                    mom="-",
                    ic="-",
                    bc="-",
                    lr="-",
                    log_w_mass="-",
                    log_w_mom="-",
                    embedding="-",
                    minutes=_trial_minutes(started),
                    show_lr=show_lr,
                    show_log_weights=show_log_weights,
                    show_embedding=show_embedding,
                ),
                show_lr=show_lr,
                show_log_weights=show_log_weights,
                show_embedding=show_embedding,
            )
            raise

    return objective


def _run_optuna_trial(cfg, problem, trial, started):
    hp = _trial_hyperparams(cfg, trial)
    model = build_model(hp, n_outputs=int(getattr(problem, "n_outputs", 2)))

    total_params_start, classical_params_start, quantum_params_start = _count_trainable_parameters(model)
    param_min = _cfg(cfg, "total_params_min", None)
    param_max = _cfg(cfg, "total_params_max", None)
    bound_reason = None
    if param_min is not None and int(param_min) > 0 and total_params_start < int(param_min):
        bound_reason = f"under the minimum of {int(param_min)}"
    elif param_max is not None and int(param_max) > 0 and total_params_start > int(param_max):
        bound_reason = f"over the maximum of {int(param_max)}"
    if bound_reason is not None:
        trial.set_user_attr("total_params", total_params_start)
        trial.set_user_attr("classical_params", classical_params_start)
        trial.set_user_attr("quantum_params", quantum_params_start)
        trial.set_user_attr("model_family", hp.model_family)
        trial.set_user_attr("n_qubits", hp.n_qubits)
        trial.set_user_attr("q_layers", hp.q_layers)
        trial.set_user_attr("embedding", hp.embedding)
        _screen(
            f"Trial {trial.number} discarded: {total_params_start} parameters "
            f"is {bound_reason} "
            f"({hp.model_family}, width {hp.width}, depth {hp.depth}, "
            f"{hp.n_qubits} qubits, {hp.q_layers} layers, {hp.embedding})."
        )
        raise optuna.TrialPruned(
            f"{total_params_start} parameters is {bound_reason}"
        )
    banner = _format_model_banner(
        f"Trial {trial.number} Start",
        model,
        hp,
        total_params_start,
        classical_params_start,
        quantum_params_start,
    )

    is_hybrid = _is_hybrid(hp.model_family, hp.n_qubits)
    trial_nx = int(cfg.trial_nx)
    trial_nt = int(cfg.trial_nt)
    warmup = 0
    if is_hybrid:
        # Default: classical for 75% of the budget, then enable quantum residual.
        frac = float(getattr(cfg, "hybrid_classical_warmup_fraction", 0.75))
        warmup = int(round(frac * int(cfg.trial_epochs)))
        warmup = max(0, min(warmup, int(cfg.trial_epochs) - 1))

    train_adam(
            model,
            problem,
            hp,
            cfg.trial_epochs,
            cfg.constraints,
            nx=trial_nx,
            nt=trial_nt,
            show_epoch_output=bool(getattr(cfg, "show_trial_epoch_output", False)),
            epoch_output_every=int(getattr(cfg, "trial_epoch_output_every", 20)),
            stage_label=f"Trial {trial.number} Adam",
            banner=banner,
            hybrid_classical_warmup_epochs=warmup,
    )

    # Keep quantum on for validation / figures after a hybrid trial.
    if is_hybrid and hasattr(model, "quantum_enabled"):
        model.quantum_enabled = True

    # Train on trial_nx by trial_nt. Rank the trial on the same loss over the
    # separate score lattice.
    score_nx = int(cfg.score_nx)
    score_nt = int(cfg.score_nt)
    x_score, t_score = sample_interior_grid(problem.domain, score_nx, score_nt)
    val_total_tf, val_mass_tf, val_mom_tf, val_constraints_tf, val_ic_tf, val_bc_tf = compute_losses(
            model,
            problem,
            score_nx * score_nt,
            cfg.constraints,
            hp.w_mass,
            hp.w_mom,
            x=x_score,
            t=t_score,
    )
    val_mass = float(val_mass_tf.numpy())
    val_mom = float(val_mom_tf.numpy())
    val_constraints = float(val_constraints_tf.numpy())
    val_ic = float(val_ic_tf.numpy())
    val_bc = float(val_bc_tf.numpy())

    # The Optuna objective must include soft IC/BC penalties. Otherwise a
    # spatially and temporally constant field can achieve a tiny PDE loss.
    raw_objective = float(val_total_tf.numpy())
    total_params, classical_params, quantum_params = _count_trainable_parameters(model)
    objective_value, param_norm, param_penalty, param_multiplier = (
            _parameter_regularized_objective(raw_objective, total_params, cfg)
    )

    trial.set_user_attr("score_nx", score_nx)
    trial.set_user_attr("score_nt", score_nt)
    trial.set_user_attr("score_n_f", int(x_score.shape[0]))
    trial.set_user_attr("val_mass", val_mass)
    trial.set_user_attr("val_mom", val_mom)
    trial.set_user_attr("val_constraints", val_constraints)
    trial.set_user_attr("val_ic", val_ic)
    trial.set_user_attr("val_bc", val_bc)
    trial.set_user_attr("raw_objective", raw_objective)
    trial.set_user_attr("objective", objective_value)
    trial.set_user_attr("model_family", hp.model_family)
    trial.set_user_attr("n_qubits", hp.n_qubits)
    trial.set_user_attr("q_layers", hp.q_layers)
    trial.set_user_attr("embedding", hp.embedding)
    trial.set_user_attr("total_params", total_params)
    trial.set_user_attr("classical_params", classical_params)
    trial.set_user_attr("quantum_params", quantum_params)
    trial.set_user_attr("param_norm", param_norm)
    trial.set_user_attr("param_penalty", param_penalty)
    trial.set_user_attr("param_multiplier", param_multiplier)
    trial.set_user_attr("parameter_objective_mode", str(_cfg(cfg, "parameter_objective_mode", "loss_multiplier")))
    target_raw_objective = _cfg(cfg, "target_raw_objective", None)
    if target_raw_objective is not None:
            trial.set_user_attr("target_raw_objective", float(target_raw_objective))
            trial.set_user_attr("meets_target_raw_objective", bool(raw_objective <= float(target_raw_objective)))

    print(
            f"[Trial {trial.number} Summary] "
            f"model_family={model.model_family}, "
            f"width={hp.width}, "
            f"depth={hp.depth}, "
            f"n_qubits={model.n_qubits}, "
            f"q_layers={model.q_layers}, "
            f"embedding={hp.embedding}, "
            f"classical_params={classical_params}, "
            f"quantum_params={quantum_params}, "
            f"total_params={total_params}, "
            f"raw_val_loss={raw_objective:.6e}, "
            f"param_norm={param_norm:.6e}, "
            f"param_penalty={param_penalty:.6e}, "
            f"param_multiplier={param_multiplier:.6e}, "
            f"objective={objective_value:.6e}, "
            f"mode={str(_cfg(cfg, 'parameter_objective_mode', 'loss_multiplier'))}"
    )
    show_lr = "lr" in _normalized_tune_set(cfg)
    show_log_weights = _log_weights_are_tuned(cfg)
    show_embedding = _show_embedding_column(cfg)
    table_embedding = hp.embedding if hp.model_family in ("hybrid_quantum", "full_quantum") else "-"
    _screen_trial_table(
        _trial_table_row(
            trial.number,
            hp.model_family,
            int(hp.width),
            int(hp.depth),
            int(hp.n_qubits),
            int(hp.q_layers),
            int(classical_params),
            int(quantum_params),
            int(total_params),
            f"{float(objective_value):.6e}",
            _trial_progress(trial, cfg.n_trials),
            mass=f"{val_mass:.6e}",
            mom=f"{val_mom:.6e}",
            ic=f"{val_ic:.6e}",
            bc=f"{val_bc:.6e}",
            lr=f"{float(hp.lr):.3e}",
            log_w_mass=f"{float(hp.log_w_mass):.3f}",
            log_w_mom=f"{float(hp.log_w_mom):.3f}",
            embedding=table_embedding,
            minutes=_trial_minutes(started),
            show_lr=show_lr,
            show_log_weights=show_log_weights,
            show_embedding=show_embedding,
        ),
        show_lr=show_lr,
        show_log_weights=show_log_weights,
        show_embedding=show_embedding,
    )
    return objective_value


def _stringify_meta(meta_dict):
    out = {}
    for key, value in meta_dict.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _count_trainable_parameters(model):
    total = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    quantum = 0
    for attr in ("quantum_h", "quantum_u", "quantum_v", "quantum"):
        layer = getattr(model, attr, None)
        if layer is None:
            continue
        # Avoid double-counting when quantum aliases quantum_h.
        if attr == "quantum" and getattr(model, "quantum_h", None) is layer:
            continue
        quantum += int(sum(np.prod(v.shape) for v in layer.trainable_variables))
    classical = total - quantum
    return total, classical, quantum


def _parameter_regularized_objective(raw_loss, total_params, cfg):
    """Optuna objective with optional parameter-efficiency modes.

    Modes:
      - "loss_multiplier" (default): raw_loss * (1 + alpha * params/reference)
      - "target_error_min_params": primarily minimize params once raw_loss is at or
        below target_raw_objective. Trials above target receive a large normalized
        error penalty, which directly tests "same error with fewer parameters".
      - "raw_loss": ignore parameter count in the scalar objective but still log it.

    The parameter values are always stored as trial user attributes.
    """
    mode = str(_cfg(cfg, "parameter_objective_mode", "loss_multiplier")).strip()
    optimize_params = bool(_cfg(cfg, "optimize_total_parameters", False))
    alpha = float(_cfg(cfg, "total_params_objective_weight", 0.0))
    reference = float(_cfg(cfg, "total_params_reference", 1.0))
    if reference <= 0.0:
        raise ValueError("total_params_reference must be > 0")

    raw_loss = float(raw_loss)
    param_norm = float(total_params) / reference
    param_multiplier = 1.0
    param_penalty = 0.0

    if (not optimize_params) or mode == "raw_loss":
        return raw_loss, param_norm, param_penalty, param_multiplier

    if mode == "target_error_min_params":
        target = _cfg(cfg, "target_raw_objective", None)
        if target is None:
            raise ValueError(
                "parameter_objective_mode='target_error_min_params' requires "
                "target_raw_objective to be set in user_input.py"
            )
        target = float(target)
        if target <= 0.0:
            raise ValueError("target_raw_objective must be > 0")

        # Zero when at/below target; grows as relative excess error otherwise.
        excess_error = max(0.0, (raw_loss - target) / target)
        error_weight = float(_cfg(cfg, "target_error_constraint_weight", 100.0))

        # Below target, params dominate. Above target, the error violation dominates.
        objective = param_norm + error_weight * excess_error
        param_penalty = alpha * param_norm
        param_multiplier = 1.0 + param_penalty
        return float(objective), param_norm, param_penalty, param_multiplier

    if mode != "loss_multiplier":
        raise ValueError(
            "Unknown parameter_objective_mode. Use 'loss_multiplier', "
            "'target_error_min_params', or 'raw_loss'."
        )

    # alpha == 0 disables the parameter term only for loss_multiplier mode.
    if alpha == 0.0:
        return raw_loss, param_norm, param_penalty, param_multiplier

    param_penalty = alpha * param_norm
    param_multiplier = 1.0 + param_penalty
    objective = raw_loss * param_multiplier
    return float(objective), param_norm, param_penalty, param_multiplier

def _write_best_hyperparameters(filepath, hp, best_trial, model, lbfgs_result=None):
    total_params, classical_params, quantum_params = _count_trainable_parameters(model)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Best hyperparameters\n")
        f.write("=" * 80 + "\n")
        f.write(f"model_family: {hp.model_family}\n")
        f.write(f"width: {hp.width}\n")
        f.write(f"depth: {hp.depth}\n")
        f.write(f"lr: {hp.lr:.16g}\n")
        f.write(f"w_mass: {hp.w_mass:.16g}\n")
        f.write(f"w_mom: {hp.w_mom:.16g}\n")
        if hp.model_family in ("hybrid_quantum", "full_quantum"):
            f.write(f"n_qubits: {hp.n_qubits}\n")
            f.write(f"q_layers: {hp.q_layers}\n")
            f.write(f"embedding: {hp.embedding}\n")

        f.write("\nTrainable parameter counts\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total trainable parameters: {total_params:,}\n")
        f.write(f"Classical trainable parameters: {classical_params:,}\n")
        f.write(f"Quantum trainable parameters: {quantum_params:,}\n")

        if best_trial is not None:
            f.write("\nOptuna best trial\n")
            f.write("-" * 80 + "\n")
            f.write(f"number: {best_trial.number}\n")
            f.write(f"objective: {best_trial.value:.16g}\n")
            val_mass = best_trial.user_attrs.get("val_mass")
            val_mom = best_trial.user_attrs.get("val_mom")
            if val_mass is not None:
                f.write(f"val_mass: {val_mass:.16g}\n")
            if val_mom is not None:
                f.write(f"val_mom: {val_mom:.16g}\n")
            for attr_name in (
                "raw_objective",
                "total_params",
                "classical_params",
                "quantum_params",
                "param_norm",
                "param_penalty",
                "param_multiplier",
            ):
                attr_value = best_trial.user_attrs.get(attr_name)
                if attr_value is not None:
                    if isinstance(attr_value, float):
                        f.write(f"{attr_name}: {attr_value:.16g}\n")
                    else:
                        f.write(f"{attr_name}: {attr_value}\n")

        if lbfgs_result is not None:
            f.write("\nL-BFGS-B\n")
            f.write("-" * 80 + "\n")
            f.write(f"used: {bool(lbfgs_result.get('used', False))}\n")
            f.write(f"success: {lbfgs_result.get('success')}\n")
            f.write(f"status: {lbfgs_result.get('status')}\n")
            f.write(f"message: {lbfgs_result.get('message')}\n")
            f.write(f"nit: {lbfgs_result.get('nit')}\n")
            f.write(f"nfev: {lbfgs_result.get('nfev')}\n")


def save_plot_data(filepath, data_dict, meta_dict):
    meta_safe = _stringify_meta(meta_dict)
    np.savez(filepath, **data_dict, meta_json=np.array(json.dumps(meta_safe)))


def _study_stem(cfg) -> str:
    """Folder name for one study: the database file name without ``.db``."""
    url = str(getattr(cfg, "storage_url", "") or "")
    prefix = "sqlite:///"
    raw = url[len(prefix):] if url.startswith(prefix) else url
    name = os.path.splitext(os.path.basename(raw))[0] if raw else ""
    return name or str(cfg.problem_name)


def _analysis_dir(cfg) -> str:
    """Directory for this study's database, ``data/``, and ``figures/``.

    The folder is ``analysis/<database name>``, the same place
    ``analyze_study.py`` writes its report and figures.
    """
    path = os.path.join("analysis", _study_stem(cfg))
    os.makedirs(path, exist_ok=True)
    return path


def _data_dir(cfg) -> str:
    path = os.path.join(_analysis_dir(cfg), "data")
    os.makedirs(path, exist_ok=True)
    return path


def _data_path(cfg, filename: str) -> str:
    return os.path.join(_data_dir(cfg), filename)


def _save_model_artifacts(model, hp: HyperParams, cfg):
    folder = _data_dir(cfg)
    weights_path = os.path.join(folder, "best_optuna_model.weights.h5")
    config_path = os.path.join(folder, "best_optuna_model_config.json")
    model.save_weights(weights_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(hp), f, indent=2)
    print(f"Saved weights: {weights_path}")
    print(f"Saved config: {config_path}")


def _sqlite_file(url: str):
    """Path of a sqlite:/// URL, or None for any other storage."""
    prefix = "sqlite:///"
    text = str(url)
    if not text.startswith(prefix):
        return None
    return text[len(prefix):] or None


def _sqlite_schema_race(exc: BaseException) -> bool:
    """True when parallel workers collide while creating an empty SQLite study."""
    text = str(exc).lower()
    return (
        "alembic_version" in text
        or "already exists" in text
        or "database is locked" in text
    )


def _resolved_storage_url(cfg) -> str:
    """Place a relative SQLite study in ``analysis/<database name>/``.

    Absolute URLs and non-SQLite storage are left unchanged. Only the file
    name is used, so ``sqlite:///optuna_study.db`` and a URL that already
    points into the analysis folder land on the same file.
    """
    url = str(getattr(cfg, "storage_url", ""))
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix):]
    if not raw or os.path.isabs(raw):
        return url
    dest = os.path.join(_analysis_dir(cfg), os.path.basename(raw))
    return prefix + dest


def _relocate_legacy_sqlite(cfg) -> None:
    """Move a database that is still beside the project or under the problem name.

    The destination is ``analysis/<database name>/<file>``. An existing file
    there is left in place.
    """
    url = str(getattr(cfg, "storage_url", ""))
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return
    raw = url[len(prefix):]
    if not raw or os.path.isabs(raw):
        return
    basename = os.path.basename(raw)
    dest = os.path.join(_analysis_dir(cfg), basename)
    sources = [raw, os.path.join("analysis", str(cfg.problem_name), basename)]
    for src in sources:
        if os.path.abspath(src) == os.path.abspath(dest):
            continue
        if os.path.exists(dest) or not os.path.exists(src):
            continue
        for suffix in ("", "-wal", "-shm"):
            src_side = src + suffix
            if os.path.exists(src_side):
                os.replace(src_side, dest + suffix)
        stale_lock = src + ".lock"
        if os.path.exists(stale_lock):
            os.remove(stale_lock)
        return


@contextmanager
def _sqlite_create_lock(cfg):
    """One worker creates a missing database. The others wait, then open it."""
    path = _sqlite_file(_resolved_storage_url(cfg))
    if path is None:
        yield
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path + ".lock", "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            _relocate_legacy_sqlite(cfg)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _optuna_storage(cfg):
    """SQLite storage with a long lock wait so several workers can share one study."""
    _relocate_legacy_sqlite(cfg)
    return optuna.storages.RDBStorage(
        url=_resolved_storage_url(cfg),
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def _open_study(cfg, sampler):
    """Create or reopen the study. Parallel workers may race on the first create."""
    last_error = None
    for attempt in range(12):
        try:
            with _sqlite_create_lock(cfg):
                storage = _optuna_storage(cfg)
                return optuna.create_study(
                    direction="minimize",
                    storage=storage,
                    study_name=cfg.study_name,
                    load_if_exists=True,
                    sampler=sampler,
                )
        except optuna.exceptions.DuplicatedStudyError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
        except Exception as exc:
            if not _sqlite_schema_race(exc) or attempt == 11:
                raise
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    raise last_error


def _load_best_hyperparams(cfg):
    study = optuna.load_study(
        study_name=cfg.study_name,
        storage=_optuna_storage(cfg),
    )
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"Study {cfg.study_name!r} has no completed trials to train."
        )
    best = study.best_trial
    hp = _hyperparams_from_completed_trial(cfg, best)
    print(
        f"Loaded best trial {best.number} from {cfg.study_name} "
        f"(objective={best.value:.6e})"
    )
    return hp, study


def _resolve_training_hyperparams(cfg, problem, worker_id=None):
    tune_set = _normalized_tune_set(cfg)
    if bool(getattr(cfg, "use_optuna", True)) and len(tune_set) > 0 and int(cfg.n_trials) > 0:
        # group=True helps TPE with conditional params (n_qubits, q_layers, and
        # embedding exist for hybrid_quantum and full_quantum; width/depth do
        # not exist for full_quantum). Startup trials explore each family in
        # model_family_options before TPE starts concentrating on one.
        n_startup = int(_cfg(cfg, "optuna_n_startup_trials", max(10, int(cfg.n_trials))))
        seed = 42 if worker_id is None else 42 + int(worker_id)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
            group=True,
        )
        study = _open_study(cfg, sampler)
        label = "Optuna joint search" if worker_id is None else f"Optuna worker {worker_id}"
        print(
            f"{label}: {', '.join(_model_family_options(cfg))} "
            f"(tune_variables={sorted(tune_set)}, "
            f"n_startup_trials={n_startup}, "
            f"completed_trial_cap={int(cfg.n_trials)})"
        )
        callbacks = [
            MaxTrialsCallback(int(cfg.n_trials), states=(TrialState.COMPLETE,))
        ]
        # optimize() counts every attempt. A draw outside the parameter
        # bounds is discarded and the search continues until n_trials
        # completed trials exist. Each of those was inside the bounds.
        attempt_limit = int(cfg.n_trials)
        param_min = _cfg(cfg, "total_params_min", None)
        param_max = _cfg(cfg, "total_params_max", None)
        has_min = param_min is not None and int(param_min) > 0
        has_max = param_max is not None and int(param_max) > 0
        if has_min or has_max:
            attempt_limit = None
            bounds = []
            if has_min:
                bounds.append(f"at least {int(param_min)}")
            if has_max:
                bounds.append(f"at most {int(param_max)}")
            print(
                f"{label}: running until {int(cfg.n_trials)} completed trials, "
                f"each with {' and '.join(bounds)} parameters"
            )
        study.optimize(
            objective_factory(cfg, problem),
            n_trials=attempt_limit,
            callbacks=callbacks,
            show_progress_bar=False,
        )
        if worker_id is not None:
            return None, study
        best = study.best_trial
        hp = _hyperparams_from_completed_trial(cfg, best)
        print("Best trial params:")
        for key, value in best.params.items():
            print(f"  {key}: {value}")
        print(
            f"Best trial family={best.user_attrs.get('model_family', best.params.get('model_family'))}, "
            f"n_qubits={best.user_attrs.get('n_qubits', best.params.get('n_qubits', 0))}, "
            f"q_layers={best.user_attrs.get('q_layers', best.params.get('q_layers', 0))}, "
            f"embedding={best.user_attrs.get('embedding', best.params.get('embedding', 'once'))}"
        )
        return hp, study

    hp = _trial_hyperparams(cfg, trial=None)
    print("Optuna search skipped. Using fixed values from user_input.py")
    return hp, None


def _save_three_field_comparison(problem, cfg, common_meta, predicted, x_tf, y_tf, X, Y):
    """Archive the final h, u, v fields for a steady (x, y) problem."""
    nx, ny = int(cfg.nx), int(cfg.nt)
    h_m, u_m, v_m = predicted
    h_a, u_a, v_a = problem.exact_solution(x_tf, y_tf)
    packed = (
        (h_m, h_a, r"$h$", "elevation"),
        (u_m, u_a, r"$u$", "velocity_u"),
        (v_m, v_a, r"$v$", "velocity_v"),
    )
    fields = []
    for model_tf, analytic_tf, clabel, stem in packed:
        model_np = model_tf.numpy().reshape(nx, ny)
        analytic_np = analytic_tf.numpy().reshape(nx, ny)
        fields.append((model_np, analytic_np))
        save_plot_data(
            _data_path(cfg, f"{problem.name}_{stem}_compare_data.npz"),
            data_dict={"Y": np.asarray(Y), "X": np.asarray(X), "A": analytic_np, "M": model_np},
            meta_dict={**common_meta, "clabel": clabel},
        )
    save_plot_data(
        _data_path(cfg, f"{problem.name}_error_data.npz"),
        data_dict={
            "Y": np.asarray(Y),
            "X": np.asarray(X),
            "H_ERR": fields[0][0] - fields[0][1],
            "U_ERR": fields[1][0] - fields[1][1],
            "V_ERR": fields[2][0] - fields[2][1],
        },
        meta_dict=common_meta,
    )


def run_solver(cfg, *, worker_id=None, finalize_only=False):
    problem = build_problem(cfg)
    constraints = _normalize_constraints(cfg.constraints)
    if constraints == "soft" and problem.soft_constraint_loss_fn is None:
        raise ValueError(
            f"Problem '{problem.name}' does not define soft IC/BC losses. "
            "Use constraints='hard', or add soft_constraint_loss_fn."
        )
    print(f"Running problem: {problem.name}")
    print(f"Constraints: {constraints}")
    print(f"Model family: {cfg.model_family}")
    print(f"Tune variables: {tuple(_normalized_tune_set(cfg))}")
    print(f"Use L-BFGS-B: {bool(getattr(cfg, 'use_lbfgs', True))}")
    print(f"Show final epoch output: {bool(getattr(cfg, 'show_epoch_output', False))}")
    print(f"Show trial epoch output: {bool(getattr(cfg, 'show_trial_epoch_output', False))}")

    if finalize_only:
        hp, study = _load_best_hyperparams(cfg)
    else:
        hp, study = _resolve_training_hyperparams(cfg, problem, worker_id=worker_id)
        if worker_id is not None:
            print(f"Worker {worker_id} finished. Final training is left to --finalize.")
            return

    model = build_model(hp, n_outputs=int(getattr(problem, "n_outputs", 2)))
    total_params, classical_params, quantum_params = _count_trainable_parameters(model)
    final_banner = _format_model_banner(
        "Final Adam",
        model,
        hp,
        total_params,
        classical_params,
        quantum_params,
    )
    adam_hist = train_adam(
        model,
        problem,
        hp,
        cfg.final_adam_epochs,
        constraints,
        nx=int(cfg.final_adam_nx),
        nt=int(cfg.final_adam_nt),
        show_epoch_output=bool(getattr(cfg, "show_epoch_output", False)),
        epoch_output_every=int(getattr(cfg, "epoch_output_every", 10)),
        stage_label="Final Adam",
        banner=final_banner,
        hybrid_classical_warmup_epochs=(
            int(round(float(getattr(cfg, "hybrid_classical_warmup_fraction", 0.75)) * int(cfg.final_adam_epochs)))
            if _is_hybrid(hp.model_family, hp.n_qubits)
            else 0
        ),
    )
    if bool(getattr(cfg, "use_lbfgs", True)):
        lbfgs = train_lbfgs(
            model,
            problem,
            hp,
            int(cfg.final_lbfgs_nx),
            int(cfg.final_lbfgs_nt),
            constraints,
            cfg.final_lbfgs_max_iters,
            cfg.final_lbfgs_ftol,
            cfg.final_lbfgs_gtol,
            cfg.final_lbfgs_maxfun,
        )
        print("L-BFGS-B:", lbfgs["message"])
    else:
        lbfgs = {
            "history": {"total": [], "mass": [], "mom": [], "constraints": []},
            "success": None,
            "status": None,
            "message": "Skipped by user option (use_lbfgs=False)",
            "nit": 0,
            "nfev": 0,
            "used": False,
        }
        print("L-BFGS-B: skipped (use_lbfgs=False)")

    _save_model_artifacts(model, hp, cfg)
    summary_path = os.path.join(_data_dir(cfg), "best_hyperparameters.txt")
    _write_best_hyperparameters(
        summary_path,
        hp,
        study.best_trial if study is not None else None,
        model,
        lbfgs,
    )
    print(f"Saved summary: {summary_path}")

    common_meta = {
        "problem": problem.name,
        "constraints": constraints,
        "width": hp.width,
        "depth": hp.depth,
        "lr": hp.lr,
        "w_mass": hp.w_mass,
        "w_mom": hp.w_mom,
        "model_family": hp.model_family,
        "n_qubits": hp.n_qubits,
        "q_layers": hp.q_layers,
        "embedding": hp.embedding,
        "n_trials": cfg.n_trials if study is not None else 0,
        "nx": cfg.nx,
        "nt": cfg.nt,
    }

    save_plot_data(
        _data_path(cfg, f"{problem.name}_losses_data.npz"),
        data_dict={
            "adam_total": np.asarray(adam_hist["total"], dtype=np.float64),
            "adam_mass": np.asarray(adam_hist["mass"], dtype=np.float64),
            "adam_mom": np.asarray(adam_hist["mom"], dtype=np.float64),
            "lbfgs_total": np.asarray(lbfgs["history"]["total"], dtype=np.float64),
            "lbfgs_mass": np.asarray(lbfgs["history"]["mass"], dtype=np.float64),
            "lbfgs_mom": np.asarray(lbfgs["history"]["mom"], dtype=np.float64),
        },
        meta_dict={
            **common_meta,
            "lbfgs_success": lbfgs["success"],
            "lbfgs_status": lbfgs["status"],
            "lbfgs_message": lbfgs["message"],
            "lbfgs_nit": lbfgs["nit"],
            "lbfgs_nfev": lbfgs["nfev"],
        },
    )

    nx, nt = cfg.nx, cfg.nt
    xv = np.linspace(problem.domain.x_min, problem.domain.x_max, nx).reshape(-1, 1)
    tv = np.linspace(problem.domain.t_min, problem.domain.t_max, nt).reshape(-1, 1)
    X, T = np.meshgrid(xv.squeeze(), tv.squeeze(), indexing="ij")

    x_tf = tf.convert_to_tensor(X.reshape(-1, 1), dtype=DTYPE)
    t_tf = tf.convert_to_tensor(T.reshape(-1, 1), dtype=DTYPE)

    predicted = outputs(model, problem, x_tf, t_tf, constraints)
    if int(getattr(problem, "n_outputs", 2)) == 3 and problem.exact_solution is not None:
        _save_three_field_comparison(
            problem, cfg, common_meta, predicted, x_tf, t_tf, X, T,
        )
        return

    eta_m, u_m = predicted
    ETA_M = eta_m.numpy().reshape(nx, nt)
    U_M = u_m.numpy().reshape(nx, nt)

    if problem.exact_solution is not None:
        eta_a, u_a = problem.exact_solution(x_tf, t_tf)
        ETA_A = eta_a.numpy().reshape(nx, nt)
        U_A = u_a.numpy().reshape(nx, nt)

        if problem.to_dimensional is not None:
            ETAA_d, UA_d, X_plot, TIME_plot = problem.to_dimensional(ETA_A, U_A, X, T)
            ETAM_d, UM_d, _, _ = problem.to_dimensional(ETA_M, U_M, X, T)
        else:
            ETAA_d, UA_d, X_plot, TIME_plot = ETA_A, U_A, X, T
            ETAM_d, UM_d = ETA_M, U_M

        save_plot_data(
            _data_path(cfg, f"{problem.name}_elevation_compare_data.npz"),
            data_dict={
                "TIME": np.asarray(TIME_plot),
                "X": np.asarray(X_plot),
                "A": np.asarray(ETAA_d),
                "M": np.asarray(ETAM_d),
            },
            meta_dict={
                **common_meta,
                "title_a": "Analytic elevation",
                "title_m": "Model elevation",
                "clabel": r"$\eta$",
            },
        )

        save_plot_data(
            _data_path(cfg, f"{problem.name}_velocity_compare_data.npz"),
            data_dict={
                "TIME": np.asarray(TIME_plot),
                "X": np.asarray(X_plot),
                "A": np.asarray(UA_d),
                "M": np.asarray(UM_d),
            },
            meta_dict={
                **common_meta,
                "title_a": "Analytic velocity",
                "title_m": "Model velocity",
                "clabel": r"$u$",
            },
        )

        save_plot_data(
            _data_path(cfg, f"{problem.name}_error_data.npz"),
            data_dict={
                "TIME": np.asarray(TIME_plot),
                "X": np.asarray(X_plot),
                "ETA_ERR": np.asarray(ETAM_d - ETAA_d),
                "U_ERR": np.asarray(UM_d - UA_d),
            },
            meta_dict=common_meta,
        )

        tau_test = np.linspace(problem.domain.t_min, problem.domain.t_max, 200).reshape(-1, 1)
        tau_tf = tf.convert_to_tensor(tau_test, dtype=DTYPE)
        x0_tf = tf.zeros((len(tau_test), 1), dtype=DTYPE) + tf.cast(problem.domain.x_min, DTYPE)

        _, u_left_m = outputs(model, problem, x0_tf, tau_tf, constraints)
        _, u_left_a = problem.exact_solution(x0_tf, tau_tf)

        save_plot_data(
            _data_path(cfg, f"{problem.name}_profiles_boundary_data.npz"),
            data_dict={
                "X": np.asarray(X_plot),
                "ETA_A": np.asarray(ETAA_d),
                "ETA_M": np.asarray(ETAM_d),
                "tau_test": np.asarray(tau_test),
                "u_left_m": np.asarray(u_left_m.numpy()),
                "u_left_a": np.asarray(u_left_a.numpy()),
            },
            meta_dict=common_meta,
        )
