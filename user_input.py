"""! @file user_input.py
@brief User-editable inputs, Optuna ranges, training settings, and problem selection.

The dataclasses are instantiated at the bottom of this file. The solver reads
``USER_PINN_INPUTS`` for the run, then loads the physical dataclass that matches
``problem_name``.

A name listed in ``tune_variables`` is sampled by Optuna on each trial, inside
the matching ``*_min`` / ``*_max`` range. Every other trainable setting stays
at its ``*_value``. ``model_family`` is the exception: when it is tuned, Optuna
draws from ``model_family_options`` rather than from a numeric range.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class UserPINNInputs:
    # Which shallow-water benchmark to build.
    # "tidal_test": linear tide on a flat friction bed, with an exact solution.
    # "dam_break": scaled wet-bed Stoker dam break.
    # "transcritical_bump": steady frictionless flow over a bump with a hydraulic jump.
    # "steady_rotation": steady (x, y) flow in solid-body rotation with an exact
    #   parabolic free surface. The trainer's second axis is y, not time.
    # "steady_vortex": isolated Gaussian vortex in gradient-wind balance.
    #   The trainer's second axis is y, not time.
    # "steady_eddies": Fourier eddy lattice on a carved bed. The trainer's
    #   second axis is y, not time.
    # The matching physical dataclass below supplies g, the domain, and boundary data.
    problem_name: str = "steady_eddies"

    # How initial and boundary conditions enter the loss.
    # "soft": the network output is free, and a penalty term pulls it onto the
    #   IC/BC data. Required for the bump's mixed boundary conditions.
    # "hard": the output matches the initial and boundary data exactly. The
    #   interior is the network, and the loss there is the PDE residual.
    #   The penalty term is zero.
    constraints: str = "hard"

    # Architecture used when "model_family" is not in tune_variables.
    # "classical": two tanh MLPs, one for depth and one for velocity.
    #   Each MLP has ``depth`` hidden layers of ``width`` units. No circuit.
    # "hybrid_quantum": those same MLPs, plus the circuit correction on each
    #   tower: trunk + decode(circuit(encode(trunk))). The correction is added
    #   in full on every epoch. ``n_qubits == 0`` is stored as classical.
    # "full_quantum": pure quantum. Each field is readout(circuit(encode(x, t))).
    #   There is no hidden MLP and no residual skip. ``width`` and ``depth`` are
    #   ignored. ``n_qubits`` and ``q_layers`` must both be at least 1.
    # Both quantum families build a separate circuit for depth and for velocity.
    # ``embedding`` chooses that circuit. See ``embedding_value``.
    model_family: str = "hybrid_quantum"

    # Families Optuna may draw when "model_family" is in tune_variables.
    # Remove a name to keep it out of the search. The fixed ``model_family``
    # above is used only when model_family is not being tuned.
    # An existing Optuna study freezes this set. Change ``study_name`` before
    # adding or removing a family, or Optuna will reject the new categorical.
    model_family_options: tuple[str, ...] = (
        "classical",
        "hybrid_quantum",
        "full_quantum",
    )

    # SQLite URL of the Optuna study. Trials are appended; a new URL starts a
    # new database. A relative file is stored in analysis/<database name>/.
    storage_url: str = "sqlite:///optuna_steady_eddies_hard_three_family.db"

    # Name of the study inside that database. Reloading the same name continues
    # the same search, including its frozen categorical choices and conditional
    # parameter distributions.
    study_name: str = "steady_eddies_hard_three_family"

    # When False, no study is created and every setting stays at its ``*_value``
    # (or ``model_family``). The same skip happens if ``tune_variables`` is empty
    # or ``n_trials`` is 0.
    use_optuna: bool = True

    # Hidden-layer width of each classical tower, used when "width" is not tuned.
    # Ignored for full_quantum. The 2_dam_break reference search preferred about 22.
    width_value: int = 22

    # Number of hidden tanh layers in each classical tower, used when "depth"
    # is not tuned. Ignored for full_quantum.
    depth_value: int = 6

    # Adam learning rate for every family. "lr" is not sampled. Adam still
    # rescales each weight from the gradient moments. The scalar does not decay.
    lr_value: float = 1.0e-3

    # Log-weight of the mass (continuity) PDE residual, used when "log_w_mass"
    # is not tuned. The training weight is exp(log_w_mass), then both PDE weights
    # are divided by their mean, so equal logs give weights (1, 1).
    # This does not scale the soft IC/BC penalty; that penalty has its own weights
    # on the physical dataclass.
    log_w_mass_value: float = 0.0

    # Log-weight of the momentum PDE residual, used when "log_w_mom" is not tuned.
    # Normalized together with log_w_mass as described above.
    log_w_mom_value: float = 0.0

    # Qubits in each tower's circuit, used when "n_qubits" is not tuned.
    # Sampled only for hybrid_quantum and full_quantum. A hybrid trial that
    # draws 0 is stored as classical. full_quantum requires at least 1.
    # Circuit angles per tower: 3 * q_layers * n_qubits.
    n_qubits_value: int = 2

    # Variational layers in each tower's circuit, used when "q_layers" is not
    # tuned. Each layer is one 3-angle rotation per qubit, then a ring of CNOTs.
    # Sampled only for quantum families, and must be at least 1 whenever the
    # circuit is actually built.
    q_layers_value: int = 2

    # Circuit layout for hybrid_quantum and full_quantum when "embedding" is
    # not tuned. Classical trials ignore it. This study fixes reupload, so
    # every quantum tower uses a linear map and embeds those angles again at
    # the start of every layer. Each angle is w_x x + w_t t.
    # "once" remains available: a tanh map, embedded on the first gate only.
    embedding_value: str = "reupload"

    # Layouts Optuna may draw when "embedding" is in tune_variables. Sampled
    # only after a trial is still hybrid_quantum or full_quantum. This study
    # leaves "embedding" out of tune_variables, so embedding_value is used.
    # An existing study freezes this set. Change ``study_name`` before editing it.
    embedding_options: tuple[str, ...] = (
        "reupload",
    )

    # Names Optuna is allowed to sample. Anything omitted stays at the matching
    # ``*_value``. Supported names:
    #   "model_family", "width", "depth", "lr", "log_w_mass", "log_w_mom",
    #   "n_qubits", "q_layers", "embedding"
    # Conditional sampling:
    #   classical: width, depth, and the log-weights if they are listed.
    #     No n_qubits, q_layers, or embedding. The learning rate stays at lr_value.
    #   hybrid_quantum: those, plus n_qubits and q_layers. The circuit is reupload.
    #   full_quantum: the log-weights, n_qubits, and q_layers. The circuit is reupload.
    #     width and depth are not sampled. The learning rate stays at lr_value.
    tune_variables: tuple[str, ...] = (
        "model_family",
        "width",
        "depth",
        "n_qubits",
        "q_layers",
    )

    # Completed-trial cap for the study. A single ``python run_pinn.py`` stops
    # at this many new trials. ``run_12cores.sh`` shares the same cap across
    # workers, so 12 processes still stop near this many completed trials.
    # The steady-rotation three-family case uses 500 completed trials.
    n_trials: int = 500

    # Above n_trials so redraws of over-cap hybrids do not start TPE before the
    # completed cap is reached. Each draw still picks a family with equal
    # probability, so the three finished counts stay close.
    optuna_n_startup_trials: int = 2000

    # Three-tower overlap for steady_eddies: largest of the three family
    # minima (hybrid width 1, depth 2, 1 qubit, 1 layer = 42) and smallest
    # of the three maxima (classical width 5, depth 7 = 603). A draw outside
    # this window is discarded and does not count toward n_trials.
    total_params_min: int | None = 42
    total_params_max: int | None = 603

    # Adam epochs inside each Optuna trial. After this many epochs the trial
    # is scored on the score_nx by score_nt lattice. This is a search budget,
    # separate from ``final_adam_epochs``.
    trial_epochs: int = 4000

    # x stations in the fixed trial collocation lattice. Adam trains on the
    # tensor product trial_nx by trial_nt, including both endpoints.
    # For steady (x, y) problems the second axis is y. Mode 7 needs more than
    # twenty stations across the square; forty-eight resolves the eddy pair.
    trial_nx: int = 48
    trial_nt: int = 48

    # x stations in the lattice where the Optuna objective is evaluated
    # after training. Twice trial_nx, so the product has four times as many points.
    score_nx: int = 96
    score_nt: int = 96

    # Fraction of Adam epochs, in both the trial and the final stage, during
    # which a hybrid model trains with the circuit skipped. The quantum weights
    # still exist; their residual is added only after the warmup. 0 keeps the
    # circuit on from epoch 1. Values are clamped to the epoch budget.
    # Applied only to hybrid_quantum. full_quantum has no classical trunk to
    # warm up, so this fraction does not change that family.
    # A dam-break hybrid run has used 0.25 here.
    hybrid_classical_warmup_fraction: float = 0

    # Fallback width bounds, used when "width" is tuned and
    # ``use_family_specific_search_ranges`` is False. Inclusive.
    # ``width_step`` is the Optuna stride, so sampled widths are
    # width_min, width_min + width_step, ... up to width_max.
    width_min: int = 1
    width_max: int = 5
    width_step: int = 1

    # Fallback depth bounds, used in the same situation as the width fallback.
    # Depth is sampled at stride 1. Inclusive.
    depth_min: int = 2
    depth_max: int = 7

    # When True, a classical trial uses the ``classical_*`` bounds and a hybrid
    # trial uses the ``hybrid_*`` bounds. Those two boxes are the same trunk.
    # A hybrid whose circuit pushes the total over total_params_max is discarded.
    # full_quantum does not sample width or depth in either case.
    use_family_specific_search_ranges: bool = True

    # Width bounds for classical trials when family-specific ranges are on.
    # Three towers: width 1 depth 2 is 21 parameters; width 5 depth 7 is 603.
    classical_width_min: int = 1
    classical_width_max: int = 5
    classical_width_step: int = 1

    # Depth bounds for classical trials when family-specific ranges are on.
    # Inclusive, stride 1.
    classical_depth_min: int = 2
    classical_depth_max: int = 7

    # Width bounds for hybrid_quantum trials when family-specific ranges are on.
    # Same trunk as classical: width 1 through 5, stride 1.
    hybrid_width_min: int = 1
    hybrid_width_max: int = 5
    hybrid_width_step: int = 1

    # Depth bounds for hybrid_quantum trials when family-specific ranges are on.
    # Same trunk as classical. Inclusive, stride 1.
    hybrid_depth_min: int = 2
    hybrid_depth_max: int = 7

    # Inclusive log-uniform Adam learning-rate range, used when "lr" is tuned.
    # The floor is 1e-4 because a draw near 2e-5 stayed on the soft-loss plateau
    # for the whole trial. Reference classical rates sit near 3e-4 to 7e-4.
    lr_min: float = 1e-4
    lr_max: float = 1e-3

    # Inclusive uniform range for log_w_mass when that name is tuned.
    # See log_w_mass_value for how the log becomes a PDE weight.
    log_w_mass_min: float = -1.0
    log_w_mass_max: float = 1.0

    # Inclusive uniform range for log_w_mom when that name is tuned.
    log_w_mom_min: float = -1.0
    log_w_mom_max: float = 1.0

    # Inclusive qubit range when "n_qubits" is tuned. Shared by hybrid_quantum
    # and full_quantum, so the lower bound should stay >= 1. A hybrid draw of 0
    # would be rewritten as classical; a full_quantum draw below 1 aborts the trial.
    n_qubits_min: int = 1
    n_qubits_max: int = 8

    # Inclusive variational-layer range when "q_layers" is tuned. Shared by both
    # quantum families. Keep the lower bound >= 1.
    q_layers_min: int = 1
    q_layers_max: int = 8

    # Master switch for the parameter-count term in the Optuna objective.
    # False forces every trial to minimize raw validation loss, regardless of
    # ``parameter_objective_mode``. Parameter counts are still logged.
    # True with mode "raw_loss" also minimizes raw loss. The flag changes the
    # objective only for "loss_multiplier" and "target_error_min_params".
    optimize_total_parameters: bool = True

    # How total trainable parameters enter the scalar Optuna objective.
    # Counts are always stored on the trial either way.
    # "raw_loss": objective = validation loss (PDE terms plus the soft penalty).
    # "loss_multiplier":
    #     objective = raw_loss * (1 + total_params_objective_weight
    #                              * total_params / total_params_reference)
    #     A weight of 0 disables the parameter factor and returns raw_loss.
    # "target_error_min_params":
    #     objective = total_params / total_params_reference
    #                 + target_error_constraint_weight * excess
    #     excess is 0 once raw_loss <= target_raw_objective, and is the relative
    #     amount above that target otherwise. Requires target_raw_objective > 0.
    #     total_params_objective_weight is logged in this mode and does not
    #     change the objective.
    parameter_objective_mode: str = "raw_loss"

    # Raw-loss target for mode "target_error_min_params". None is valid for the
    # other modes. Set it from a finished classical baseline on this same problem
    # before using the target mode.
    target_raw_objective: float | None = None

    # Multiplier on the relative excess loss in "target_error_min_params".
    # Larger values make Optuna reject trials that miss the loss target before
    # it rewards a smaller parameter count.
    target_error_constraint_weight: float = 100.0

    # Coefficient named above in the "loss_multiplier" formula. Unused by the
    # objective when the mode is "raw_loss" or "target_error_min_params".
    total_params_objective_weight: float = 1.0

    # Parameter count that counts as "1" in both parameter-aware objectives.
    # A model with this many trainable parameters has param_norm = 1.
    # Must be > 0.
    total_params_reference: float = 100_000.0

    # Same aligned table, printed inside each Optuna trial.
    show_trial_epoch_output: bool = True

    # Trial table row at epoch 1 and every Nth trial epoch.
    # The column header is printed once per trial, before the first row.
    trial_epoch_output_every: int = 1

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Adam epochs for the model rebuilt from the best trial (or from the fixed
    # values, if the search was skipped). This run is not itself an Optuna trial.
    final_adam_epochs: int = 45000

    # x stations in the final Adam collocation lattice.
    final_adam_nx: int = 64

    # y stations in the final Adam collocation lattice.
    final_adam_nt: int = 64

    # Aligned loss table during the final Adam stage. Same five losses as the
    # one-line print: total, mass, momentum, IC, and BC.
    show_epoch_output: bool = True

    # Final-stage table row at epoch 1 and every Nth epoch after that.
    # The column header is printed once, before the first row.
    epoch_output_every: int = 500

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # After the final Adam stage, continue with SciPy L-BFGS-B on the same
    # trainable variables. False keeps the Adam weights and still writes the
    # summary, plots, and archives.
    use_lbfgs: bool = False

    # x stations in the L-BFGS-B collocation lattice.
    final_lbfgs_nx: int = 128

    # y stations in the L-BFGS-B collocation lattice.
    final_lbfgs_nt: int = 128

    # SciPy L-BFGS-B maxiter. Iteration cap, not a function-evaluation cap.
    final_lbfgs_max_iters: int = 250

    # SciPy ftol. Stop when the relative loss decrease falls below this.
    final_lbfgs_ftol: float = 1e-12

    # SciPy gtol. Stop when the projected gradient norm falls below this.
    final_lbfgs_gtol: float = 1e-10

    # SciPy maxfun. Cap on loss/gradient evaluations, separate from maxiter.
    final_lbfgs_maxfun: int = 800

    # x stations in the final comparison figures and npz archives.
    # This is an evaluation mesh, not the training collocation grid. Optuna
    # trials do not write a model-versus-analytic figure; that figure is saved
    # once, after the best trial is trained.
    nx: int = 201

    # y stations in that same final mesh for steady_rotation.
    nt: int = 201

@dataclass
class TidalTestParameters:
    """Linear tidal test. Lengths in meters, time in seconds, g in m/s^2.

    The solver nondimensionalizes x by L and t by T_tide. The network domain
    in that scaling is x in [0, 1] and t in [0, tau_max]. Soft IC/BC weights
    are not fields here; the tidal builder uses its own defaults.
    """

    # Gravitational acceleration.
    g: float = 9.81

    # Undisturbed water depth.
    H: float = 10.0

    # Linear friction coefficient, 1/s. Enters the momentum residual as
    # beta = r * T_tide.
    r: float = 1.0e-4

    # Basin length. Also the scale that converts nondimensional x to meters
    # (and to kilometers on the comparison axes).
    L: float = 100_000.0

    # Tidal free-surface amplitude at the forced end. Also the scale for the
    # nondimensional surface elevation plotted in the comparison figures.
    A: float = 0.01

    # Tidal period. 12.42 hours. Also the scale that converts nondimensional
    # time to seconds (and to hours on the comparison axes).
    T_tide: float = 12.42 * 3600.0

    # Nondimensional end time, in units of T_tide. The training window is
    # tau from 0 to tau_max.
    tau_max: float = 3.0

@dataclass
class DamBreakParameters:
    """Scaled wet-bed dam break matching Desktop/2_dam_break.

    g = 9.81 m/s^2, left depth 0.5 m, right depth 0.25 m, still water,
    x in [0, 1] m, t in [0, 0.2] s, dam at x = 0.5 m.
    Soft mode penalizes the initial step and the two walls separately:
    L_mass + L_mom + 50 * MSE(h_ic) + 50 * MSE(u_ic)
    + 10 * MSE(h_bc) + 10 * MSE(u_bc).
    The 41 initial points are packed toward the dam. Each wall has 10 points.
    """

    # Gravitational acceleration, m/s^2.
    g: float = 9.81

    # Initial water depth to the left of the dam, m.
    h_left: float = 0.5

    # Initial water depth to the right of the dam, m.
    h_right: float = 0.25

    # Initial velocity everywhere, m/s. Also the boundary velocity target in
    # soft mode.
    u0: float = 0.0

    # Dam position, m. Initial depth is h_left for x <= x_mid and h_right otherwise.
    x_mid: float = 0.5

    # Width of an optional tanh smoothing of the initial step, in meters.
    # 0 keeps the sharp step used by the reference case. A positive value
    # replaces the step with a tanh jump of that scale.
    transition_width: float = 0.0

    # Initial-condition samples at t = t_min. Placed on [x_min, x_max] and
    # packed toward x_mid, so the step is sampled more tightly than the flats.
    # An odd count puts a sample on the dam.
    soft_n_ic: int = 41

    # Boundary samples on each wall, uniformly spaced in time. The wall
    # targets are constant, so this count does not need to resolve a front.
    soft_n_bc: int = 10

    # Multiplier on the initial-condition MSE of water depth.
    soft_w_ic_h: float = 50.0

    # Multiplier on the initial-condition MSE of velocity.
    soft_w_ic_u: float = 50.0

    # Multiplier on the wall MSE of water depth. Lower than the initial
    # step weight: the walls are constant and are already fit closely.
    soft_w_bc_h: float = 10.0

    # Multiplier on the wall MSE of velocity.
    soft_w_bc_u: float = 10.0

    # Left end of the spatial domain, m.
    x_min: float = 0.0

    # Right end of the spatial domain, m.
    x_max: float = 1.0

    # Initial time, s.
    t_min: float = 0.0

    # Final time, s.
    t_max: float = 0.2

@dataclass
class TranscriticalBumpParameters:
    """Steady frictionless transcritical flow over a parabolic bump.

    Standard benchmark: x in [0, 25] m, upstream discharge 0.18 m^2/s,
    downstream depth 0.33 m, bump z = 0.2 - 0.05 (x - 10)^2 on 8 < x < 12
    and z = 0 elsewhere. The solver still carries a time coordinate, but the
    network is always evaluated at t = 0, so the learned field is steady.
    """

    # Gravitational acceleration, m/s^2.
    g: float = 9.81

    # Prescribed upstream unit discharge, m^2/s. Soft mode penalizes
    # h(0) * u(0) against this value. Hard mode enforces it exactly.
    q_in: float = 0.18

    # Prescribed downstream water depth, m. Soft mode penalizes h(x_max)
    # against this value. Hard mode enforces it exactly.
    h_down: float = 0.33

    # Crest elevation of the bump, m. z(bump_center) = bump_height.
    bump_height: float = 0.20

    # Parabola coefficient, 1/m. On the bump,
    # z = bump_height - bump_coeff * (x - bump_center)^2.
    bump_coeff: float = 0.05

    # Streamwise location of the crest, m.
    bump_center: float = 10.0

    # Upstream toe of the bump, m. Bed elevation is 0 for x <= bump_x1.
    bump_x1: float = 8.0

    # Downstream toe of the bump, m. Bed elevation is 0 for x >= bump_x2.
    bump_x2: float = 12.0

    # Multiplier on the upstream discharge penalty (h*u - q_in)^2.
    # Used only when constraints = "soft".
    soft_w_q_in: float = 50.0

    # Multiplier on the downstream depth penalty (h - h_down)^2.
    # Used only when constraints = "soft".
    soft_w_h_down: float = 50.0

    # Left end of the channel, m.
    x_min: float = 0.0

    # Right end of the channel, m.
    x_max: float = 25.0

    # Dummy time interval retained so this steady problem can use the shared
    # (x, t) training loop. The bump output ignores t and evaluates at 0.
    t_min: float = 0.0
    t_max: float = 1.0

@dataclass
class SteadyRotationParameters:
    """Steady solid-body rotation on a flat bed, with an exact parabolic surface.

    The square is centered on the origin. Set problem_name to "steady_rotation"
    and point storage_url and study_name at a new study before training it.
    For this problem trial_nx and trial_nt are the x and y station counts.
    """

    g: float = 9.81

    # Rotation rate, 1/s. Velocity is (-omega * y, omega * x).
    omega: float = 1.0

    # Depth at the origin, m. The surface rises with radius, so this is the minimum.
    h_center: float = 1.0

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0

    # Stations on each side of the square for the soft boundary penalty.
    soft_n_edge: int = 32

    # Multipliers on the boundary penalties for h, u, and v.
    soft_w_h: float = 50.0
    soft_w_u: float = 50.0
    soft_w_v: float = 50.0

@dataclass
class SteadyVortexParameters:
    """Isolated vortex in gradient-wind balance on a flat bed.

    The depth is lowest at the origin and the azimuthal speed peaks near
    ``sigma``. Centrifugal and Coriolis contributions to the surface slope
    are equal at the defaults. Set problem_name to "steady_vortex" and point
    storage_url and study_name at a new study before training it.
    """

    g: float = 9.81

    # Coriolis parameter, 1/s. A rotating table at one revolution per few
    # seconds, not the planetary value. Set to 0 for a non-rotating bucket vortex.
    f: float = 2.0

    # Peak angular rate of the vortex core, 1/s. The azimuthal velocity is
    # omega * r * exp(-r^2 / (2 sigma^2)).
    omega: float = 4.0

    # Vortex radius, m. The speed peaks at r = sigma.
    sigma: float = 0.45

    # Depth at the center, m. The surface rises outward.
    h_center: float = 0.40

    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5

    # Stations on each side of the square for the soft boundary penalty.
    soft_n_edge: int = 32

    # Multipliers on the boundary penalties for h, u, and v.
    soft_w_h: float = 50.0
    soft_w_u: float = 50.0
    soft_w_v: float = 50.0

@dataclass
class SteadyEddiesParameters:
    """Fourier eddy lattice on a carved bed, with an exact steady solution.

    Modes (mode_m, mode_n) and (mode_n, mode_m) share one eigenvalue on the
    square. Depth is H0 + gamma * psi. The bed is cut from Bernoulli's
    relation so the residual vanishes. Set problem_name to "steady_eddies"
    and point storage_url and study_name at a new study before training it.
    """

    g: float = 9.81

    # Stream-function amplitude, m^2/s.
    amplitude: float = 0.015

    # Mean depth, m. Also the wall depth, since psi vanishes on the boundary.
    h_mean: float = 0.80

    # Depth scales with the stream function as H0 + gamma * psi.
    gamma: float = 4.0

    # Degenerate Fourier pair. Both (mode_m, mode_n) and (mode_n, mode_m)
    # appear in the stream function.
    mode_m: int = 7
    mode_n: int = 4

    # Bernoulli constant C in eta = C - |u|^2/(2g) - mu psi^2/(2g).
    # Keep C large enough that the bed stays below the free surface.
    bernoulli_c: float = 1.0

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    # Stations on each side of the square for the soft boundary penalty.
    soft_n_edge: int = 32

    # Soft mode enforces slip walls only: depth equals h_mean, and the normal
    # velocity vanishes. Tangential speed is free. soft_w_u and soft_w_v are
    # averaged into the normal-velocity weight.
    soft_w_h: float = 50.0
    soft_w_u: float = 50.0
    soft_w_v: float = 50.0

USER_PINN_INPUTS = UserPINNInputs()
TIDAL_TEST_PARAMETERS = TidalTestParameters()
DAM_BREAK_PARAMETERS = DamBreakParameters()
TRANSCRITICAL_BUMP_PARAMETERS = TranscriticalBumpParameters()
STEADY_ROTATION_PARAMETERS = SteadyRotationParameters()
STEADY_VORTEX_PARAMETERS = SteadyVortexParameters()
STEADY_EDDIES_PARAMETERS = SteadyEddiesParameters()
