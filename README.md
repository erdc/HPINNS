# SWE PINN package with selective Optuna tuning

This package lets you choose exactly which variables Optuna is allowed to tune.
Everything else stays fixed at explicit values from `user_input.py`.

## Main controls

In `user_input.py`:

- `use_optuna = True` enables Optuna
- `tune_variables = (...)` lists the only variables Optuna may tune
- `*_value` fields are the fixed values used whenever a variable is not tuned
- `*_min`, `*_max`, and `*_step` fields are only used for variables listed in `tune_variables`

Supported names in `tune_variables`:

- `"model_family"` — categorical over `model_family_options`
- `"width"`
- `"depth"`
- `"lr"`
- `"log_w_mass"`
- `"log_w_mom"`
- `"n_qubits"`
- `"q_layers"`

`model_family_options` defaults to `classical`, `hybrid_quantum`, and `full_quantum`.

- `classical`: tanh MLP trunk, no circuit
- `hybrid_quantum`: the same trunk plus a residual quantum filter
- `full_quantum`: pure quantum. Each field is `readout(quantum(encode(x, t)))`, with no classical hidden trunk. `width` and `depth` are not used.

An existing Optuna study freezes the `model_family` choices. Point `study_name` at a new study before adding or removing a family.

## Examples

### Tune only quantum settings

```python
use_optuna = True
tune_variables = ("n_qubits", "q_layers")

model_family = "hybrid_quantum"
width_value = 192
depth_value = 3
lr_value = 2.9260791881920547e-4
log_w_mass_value = 1.8993874603050303
log_w_mom_value = 1.9769958820830553
```

### Tune only learning rate

```python
use_optuna = True
tune_variables = ("lr",)
```

### Skip Optuna entirely

```python
use_optuna = False
tune_variables = ()
```

## Run

```bash
pip install tensorflow optuna scipy matplotlib pennylane
python run_pinn.py
```


## New option: skip L-BFGS-B

In `user_input.py`, set:

```python
use_lbfgs = False
```

When `use_lbfgs=False`, the package will train with Adam only, skip the final SciPy L-BFGS-B stage, and still save model weights, config, plots, and data archives.


## Outputs

Running the solver writes a `best_hyperparameters.txt` summary with the chosen hyperparameters plus total, classical, and quantum trainable parameter counts.


## Added controls

- `show_epoch_output`, `epoch_output_every`
- `show_trial_epoch_output`, `trial_epoch_output_every`
- `best_hyperparameters.txt` now includes total/classical/quantum trainable parameter counts

## Transcritical bump benchmark

A third built-in shallow-water benchmark is available as:

```python
problem_name = "transcritical_bump"
```

It implements the standard steady, frictionless transcritical flow over a parabolic
bump with a stationary hydraulic jump:

- domain: `x = 0 ... 25 m`
- bed: `z = 0.2 - 0.05 (x-10)^2` for `8 < x < 12`, zero elsewhere
- upstream unit discharge: `q = 0.18 m^2/s`
- downstream depth: `h = 0.33 m`

The reference solution is generated from the steady specific-energy relation on the
smooth branches and the Rankine--Hugoniot momentum condition at the hydraulic jump.
Both soft and hard enforcement of the mixed physical boundary conditions are
implemented.  The shared solver retains a dummy time coordinate, but this problem
forces the network output to be independent of time.

A ready-to-copy input preset is included as `user_input.py.bump`.  It uses a separate
Optuna database (`optuna_transcritical_bump_soft.db`) and study name so bump trials do
not mix with the dam-break study.  To use it, copy it over `user_input.py` or reproduce
those settings in your active input file.
