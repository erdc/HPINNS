#!/usr/bin/env python3
"""Study report, publication figures, and analytic comparison for one problem.

Reads the Optuna study named in user_input.py. The database stays where the
solver wrote it. Reports, tables, best-model files, and PDF figures go in
``analysis/<database name>/data`` and ``analysis/<database name>/figures``.

    python3 analysis/analyze_study.py
    python3 analysis/analyze_study.py --skip-train
    python3 analysis/analyze_study.py --db optuna_tidal.db --study tidal_study --problem tidal_test

The text report is the numeric tradeoff summary. When the problem has an exact solution, the best completed
trial is trained (or the saved final weights are loaded) and compared with
that solution on the figure mesh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState

from user_input import USER_PINN_INPUTS


PARAM_COUNT_KEYS = (
    "total_params",
    "n_params",
    "num_params",
    "total_parameters",
    "params",
    "model_params",
)
RAW_ERROR_KEYS = (
    "raw_objective",
    "raw_error",
    "validation_error",
    "val_error",
    "mse",
    "loss",
)
MODEL_FAMILY_KEYS = ("model_family", "family", "model_type", "architecture")
Q_LAYER_KEYS = ("q_layers", "quantum_layers", "n_q_layers", "num_q_layers")
QUBIT_KEYS = ("n_qubits", "qubits", "num_qubits")

FAMILY_ORDER = (
    "classical",
    "hybrid_quantum",
    "hybrid_quantum_reupload",
    "full_quantum",
    "full_quantum_reupload",
)
FAMILY_MARKERS = {
    "classical": "o",
    "hybrid_quantum": "s",
    "hybrid_quantum_reupload": "D",
    "full_quantum": "^",
    "full_quantum_reupload": "v",
}
FAMILY_COLORS = {
    "classical": "#0072B2",
    "hybrid_quantum": "#D55E00",
    "hybrid_quantum_reupload": "#CC79A7",
    "full_quantum": "#009E73",
    "full_quantum_reupload": "#56B4E9",
}

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
})


def family_marker(name: str) -> str:
    return FAMILY_MARKERS.get(str(name), "D")


def family_color(name: str) -> str:
    return FAMILY_COLORS.get(str(name), "#333333")


def family_label(name: str) -> str:
    return {
        "hybrid_quantum": "Hybrid quantum",
        "hybrid_quantum_reupload": "Hybrid quantum",
        "classical": "Classical",
        "full_quantum": "Full quantum",
        "full_quantum_reupload": "Full quantum",
    }.get(str(name), str(name).replace("_", " "))


def series_name(family: str, embedding) -> str:
    """Family name used in comparison figures.

    Re-uploading is its own series. A missing layout, and ``once``, stay on
    the original family name so an older study keeps three series.
    """
    family = str(family)
    if str(embedding) == "reupload" and family in ("hybrid_quantum", "full_quantum"):
        return f"{family}_reupload"
    return family


def with_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "embedding" not in out.columns:
        out["series"] = out["model_family"].astype(str)
        return out
    out["series"] = [
        series_name(family, embedding)
        for family, embedding in zip(out["model_family"], out["embedding"])
    ]
    return out


def ordered_families(names: Sequence[str]) -> list[str]:
    present = {str(name) for name in names}
    ordered = [name for name in FAMILY_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def apply_publication_axes(ax) -> None:
    ax.tick_params(direction="out", length=5, width=1.0, labelsize=13)
    ax.grid(True, which="major", linewidth=0.55, alpha=0.22)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.10)
    ax.set_axisbelow(True)


def first_existing(mapping: dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def as_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def sqlite_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve()}"


def resolve_db(url_or_path: str, problem_name: str) -> Path:
    """Resolve a study database.

    The solver stores a relative database at
    ``analysis/<database name>/<filename>``. Older runs stored it at
    ``analysis/<problem_name>/<filename>`` or in the working directory.
    A path that already contains a directory is used as given when that file
    exists, and otherwise falls back to the database-name folder.
    """
    text = str(url_or_path)
    if text.startswith("sqlite:///"):
        text = text[len("sqlite:///"):]
    path = Path(text)
    if path.is_absolute():
        return path
    by_name = ROOT / "analysis" / path.stem / path.name
    named = ROOT / "analysis" / problem_name / path.name
    legacy = ROOT / path.name
    if len(path.parts) > 1:
        given = (Path.cwd() / path).resolve()
        if given.is_file() or not by_name.is_file():
            return given
        return by_name
    for candidate in (by_name, named, legacy):
        if candidate.is_file():
            return candidate
    return by_name


def _storage(url: str):
    return optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def get_study(storage: str, study_name: Optional[str]) -> optuna.Study:
    backend = _storage(storage)
    summaries = optuna.study.get_all_study_summaries(backend)
    if not summaries:
        raise RuntimeError(f"No Optuna studies found in {storage}")
    if study_name is None:
        if len(summaries) != 1:
            names = ", ".join(item.study_name for item in summaries)
            raise RuntimeError(f"Several studies are in this database ({names}). Pass --study.")
        study_name = summaries[0].study_name
    return optuna.load_study(study_name=study_name, storage=backend)


def trial_to_row(trial: optuna.trial.FrozenTrial) -> dict:
    attrs = dict(trial.user_attrs)
    params = dict(trial.params)
    values = trial.values
    if values is None:
        values = [trial.value] if trial.value is not None else []
    objective_0 = as_float(values[0]) if values else np.nan

    raw_error = first_existing(attrs, RAW_ERROR_KEYS, None)
    if raw_error is None:
        raw_error = first_existing(params, RAW_ERROR_KEYS, None)
    raw_error = as_float(raw_error, objective_0)

    total_params = first_existing(attrs, PARAM_COUNT_KEYS, None)
    if total_params is None:
        total_params = first_existing(params, PARAM_COUNT_KEYS, None)
    if total_params is None and len(values) > 1:
        total_params = values[1]

    model_family = first_existing(attrs, MODEL_FAMILY_KEYS, None)
    if model_family is None:
        model_family = first_existing(params, MODEL_FAMILY_KEYS, None)
    if model_family is None:
        nq = first_existing(params, QUBIT_KEYS, first_existing(attrs, QUBIT_KEYS, 0))
        model_family = "hybrid_quantum" if as_float(nq, 0.0) > 0 else "classical"

    q_layers = first_existing(attrs, Q_LAYER_KEYS, first_existing(params, Q_LAYER_KEYS, np.nan))
    n_qubits = first_existing(attrs, QUBIT_KEYS, first_existing(params, QUBIT_KEYS, np.nan))
    embedding = first_existing(attrs, ("embedding",), first_existing(params, ("embedding",), ""))
    embedding = "" if embedding is None else str(embedding)

    row = {
        "number": trial.number,
        "state": str(trial.state).replace("TrialState.", ""),
        "raw_objective": raw_error,
        "raw_error": raw_error,
        "optuna_objective": objective_0,
        "total_params": as_float(total_params),
        "classical_params": as_float(attrs.get("classical_params")),
        "quantum_params": as_float(attrs.get("quantum_params")),
        "model_family": str(model_family),
        "q_layers": as_float(q_layers),
        "n_qubits": as_float(n_qubits),
        "embedding": embedding,
        "width": as_float(first_existing(params, ("width",), attrs.get("width"))),
        "depth": as_float(first_existing(params, ("depth",), attrs.get("depth"))),
    }
    if row["model_family"] == "full_quantum":
        row["width"] = 0.0
        row["depth"] = 0.0
    if row["model_family"] == "classical":
        row["n_qubits"] = 0.0
        row["q_layers"] = 0.0
        row["embedding"] = ""
    return row


def study_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    rows = [trial_to_row(trial) for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not rows:
        raise RuntimeError(f"Study {study.study_name!r} has no completed trials.")
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["raw_objective", "total_params"]).copy()
    valid = valid[(valid["total_params"] > 0) & (valid["raw_objective"] >= 0)]
    valid = valid.sort_values(["total_params", "raw_objective"], ascending=[True, True])
    best_loss = float("inf")
    keep = []
    for _, row in valid.iterrows():
        loss = float(row["raw_objective"])
        if loss < best_loss:
            keep.append(True)
            best_loss = loss
        else:
            keep.append(False)
    return valid.loc[keep].copy()


def family_spread(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["raw_objective", "model_family"]).copy()
    loss = pd.to_numeric(valid["raw_objective"], errors="coerce")
    valid = valid[loss.notna() & loss.abs().lt(float("inf"))]
    valid = with_series(valid)
    rows = []
    for name in ordered_families(valid["series"].astype(str)):
        scores = valid.loc[valid["series"].astype(str) == name, "raw_objective"]
        rows.append({
            "model_family": name,
            "n": int(scores.shape[0]),
            "best": float(scores.min()),
            "p25": float(scores.quantile(0.25)),
            "median": float(scores.median()),
            "p75": float(scores.quantile(0.75)),
            "worst": float(scores.max()),
        })
    return pd.DataFrame(rows)


def _fmt_frame(df: pd.DataFrame, sci_cols: Sequence[str]) -> pd.DataFrame:
    shown = df.copy()
    for col in sci_cols:
        if col in shown.columns:
            shown[col] = shown[col].map(lambda value: f"{float(value):.6e}" if pd.notna(value) else "")
    for col in ("number", "n_qubits", "q_layers", "width", "depth", "total_params", "classical_params", "quantum_params", "n"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else str(int(value)))
    return shown


def write_tradeoff_report(df: pd.DataFrame, outpath: Path, target: Optional[float]) -> None:
    cols = [
        "number", "model_family", "embedding", "width", "depth", "n_qubits", "q_layers",
        "classical_params", "quantum_params", "total_params",
        "raw_objective", "optuna_objective",
    ]
    cols = [col for col in cols if col in df.columns]
    lines = [
        "Parameter tradeoff",
        "The recorded objective is the strong-form loss on the score lattice, including the soft penalty.",
        "It is not the analytic solution error. That comparison is in analytic_comparison.txt when an exact solution exists.",
        "",
        "Best trials by recorded objective",
        _fmt_frame(df.sort_values("raw_objective").head(15)[cols], ["raw_objective", "optuna_objective"]).to_string(index=False),
        "",
        "Recorded-objective spread by model family",
    ]
    spread = family_spread(df)
    if spread.empty:
        lines.append("No completed trials with a finite recorded objective.")
    else:
        lines.append(_fmt_frame(spread, ["best", "p25", "median", "p75", "worst"]).to_string(index=False))

    front = pareto_front(df)
    lines.extend(["", "Pareto front: fewer parameters and a lower recorded objective"])
    if front.empty:
        lines.append("No finite positive objectives with a positive parameter count.")
    else:
        lines.append(_fmt_frame(front[cols], ["raw_objective", "optuna_objective"]).to_string(index=False))

    if target is not None:
        ok = df[(df["raw_objective"] <= target) & df["total_params"].notna()].copy()
        lines.extend(["", f"Trials with recorded objective <= {target:.6e}"])
        if ok.empty:
            lines.append("No trials meet the target.")
        else:
            lines.append(_fmt_frame(ok.sort_values("total_params")[cols], ["raw_objective", "optuna_objective"]).to_string(index=False))
            classical = ok[ok["model_family"].astype(str) == "classical"].sort_values("total_params").head(1)
            hybrid = ok[ok["model_family"].astype(str) == "hybrid_quantum"].sort_values("total_params").head(1)
            if len(classical) and len(hybrid):
                c = classical.iloc[0]
                h = hybrid.iloc[0]
                reduction = 100.0 * (1.0 - float(h["total_params"]) / float(c["total_params"]))
                lines.extend([
                    "",
                    "Smallest hybrid and smallest classical at or below the target",
                    f"classical trial {int(c['number'])}: params={int(c['total_params'])}, objective={float(c['raw_objective']):.6e}",
                    f"hybrid trial {int(h['number'])}: params={int(h['total_params'])}, objective={float(h['raw_objective']):.6e}",
                    f"parameter reduction of hybrid relative to classical: {reduction:.2f}%",
                ])
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {outpath}")


def scatter_by_family(df, x, y, xlabel, ylabel, outpath, logx=False, logy=True) -> None:
    plot_df = with_series(df[[x, y, "model_family"] + (["embedding"] if "embedding" in df.columns else [])].dropna(subset=[x, y, "model_family"]))
    plot_df = plot_df[(plot_df[x] > 0) & (plot_df[y] > 0)]
    if plot_df.empty:
        print(f"Skipping {outpath.name}: no valid data.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for fam in ordered_families(plot_df["series"]):
        sub = plot_df[plot_df["series"].astype(str) == fam]
        ax.scatter(
            sub[x], sub[y], s=58, alpha=0.92,
            marker=family_marker(fam), color=family_color(fam),
            edgecolors="black", linewidths=0.45, label=family_label(fam),
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_publication_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, fancybox=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def plot_objective_history(df: pd.DataFrame, outpath: Path) -> None:
    columns = ["number", "raw_objective", "model_family"]
    if "embedding" in df.columns:
        columns.append("embedding")
    d = with_series(df[columns].dropna(subset=["number", "raw_objective", "model_family"]))
    d = d[d["raw_objective"] > 0]
    if d.empty:
        print(f"Skipping {outpath.name}: no objective data.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for fam in ordered_families(d["series"]):
        sub = d[d["series"].astype(str) == fam]
        ax.scatter(
            sub["number"], sub["raw_objective"], s=52, alpha=0.90,
            marker=family_marker(fam), color=family_color(fam),
            edgecolors="black", linewidths=0.4, label=family_label(fam),
        )
    best = d.sort_values("number").copy()
    best["best_so_far"] = best["raw_objective"].cummin()
    ax.step(best["number"], best["best_so_far"], where="post", linewidth=2.2, color="black", label="Best so far")
    ax.set_yscale("log")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Recorded objective")
    apply_publication_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, fancybox=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def plot_pareto(df: pd.DataFrame, outpath: Path) -> None:
    columns = ["total_params", "raw_objective", "model_family", "number"]
    if "embedding" in df.columns:
        columns.append("embedding")
    d = with_series(df[columns].dropna(subset=["total_params", "raw_objective", "model_family", "number"]))
    d = d[(d["total_params"] > 0) & (d["raw_objective"] > 0)]
    if d.empty:
        print(f"Skipping {outpath.name}: no parameter/objective data.")
        return
    pareto = pareto_front(d.rename(columns={"raw_objective": "raw_objective"})).sort_values("total_params")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for fam in ordered_families(d["series"]):
        sub = d[d["series"].astype(str) == fam]
        ax.scatter(
            sub["total_params"], sub["raw_objective"],
            s=54, alpha=0.80, marker=family_marker(fam),
            color=family_color(fam), edgecolors="black", linewidths=0.35,
            label=family_label(fam),
        )
    if not pareto.empty:
        ax.plot(
            pareto["total_params"], pareto["raw_objective"],
            linewidth=2.4, color="black", label="Pareto front", zorder=4,
        )
        for fam in ordered_families(pareto["series"]):
            sub = pareto[pareto["series"].astype(str) == fam]
            ax.scatter(
                sub["total_params"], sub["raw_objective"],
                s=115, marker=family_marker(fam),
                facecolors=family_color(fam), edgecolors="black", linewidths=1.4,
                zorder=6,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total trainable parameters")
    ax.set_ylabel("Recorded objective")
    apply_publication_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, fancybox=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def _draw_family_boxes(ax, families: Sequence[str], groups: Sequence[np.ndarray]) -> np.ndarray:
    positions = 1.0 + np.arange(len(families)) * 1.12
    boxes = ax.boxplot(
        list(groups),
        positions=positions,
        showfliers=False,
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "#E6E6E6", "edgecolor": "black", "linewidth": 1.0, "zorder": 1},
        medianprops={"color": "C1", "linewidth": 1.5, "zorder": 4},
        whiskerprops={"color": "black", "linewidth": 1.0, "zorder": 1},
        capprops={"color": "black", "linewidth": 1.0, "zorder": 1},
    )
    for artist in boxes["boxes"] + boxes["whiskers"] + boxes["caps"]:
        artist.set_zorder(1)
    for artist in boxes["medians"]:
        artist.set_zorder(4)
    ax.set_yscale("log")
    ax.set_xticks(positions, [family_label(fam) for fam in families])
    for pos, fam, values in zip(positions, families, groups):
        if len(values) == 0:
            continue
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), pos) + offsets, values,
            s=36, alpha=0.85, marker=family_marker(fam),
            color=family_color(fam), edgecolors="black", linewidths=0.35,
            zorder=3,
        )
        median = float(np.median(values))
        exponent = int(np.floor(np.log10(median)))
        mantissa = median / 10**exponent
        ax.annotate(
            rf"${mantissa:.2f}\times 10^{{{exponent}}}$",
            xy=(pos + 0.32, median),
            ha="left",
            va="center",
            fontsize=9.75,
            color="black",
            zorder=5,
            annotation_clip=False,
        )
    return positions


def plot_family_comparison(df: pd.DataFrame, outpath: Path) -> None:
    needed = ("model_family", "raw_objective", "total_params")
    if any(column not in df.columns for column in needed):
        print(f"Skipping {outpath.name}: no family data.")
        return
    columns = list(needed)
    if "embedding" in df.columns:
        columns.append("embedding")
    d = with_series(df.loc[:, columns].dropna(subset=list(needed)))
    d = d[(d["raw_objective"] > 0) & (d["total_params"] > 0)].copy()
    families = ordered_families(d["series"])
    if not families:
        print(f"Skipping {outpath.name}: no family data.")
        return
    fig, ax = plt.subplots(figsize=(7.4 if len(families) <= 3 else 9.6, 4.6))
    groups = [d.loc[d["series"] == fam, "raw_objective"].to_numpy() for fam in families]
    positions = _draw_family_boxes(ax, families, groups)
    ax.set_ylabel("Recorded objective")
    apply_publication_axes(ax)
    if len(positions):
        ax.set_xlim(float(positions[0]) - 0.55, float(positions[-1]) + 0.95)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def plot_group_best(df: pd.DataFrame, group_col: str, xlabel: str, outpath: Path) -> None:
    columns = [group_col, "raw_objective", "model_family"]
    if "embedding" in df.columns:
        columns.append("embedding")
    d = df[columns].dropna(subset=[group_col, "raw_objective", "model_family"])
    d = d[(d[group_col] > 0) & (d["raw_objective"] > 0)]
    d = d[d["model_family"].astype(str).isin(("hybrid_quantum", "full_quantum"))]
    d = with_series(d)
    if d.empty:
        print(f"Skipping {outpath.name}: no {group_col} data.")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for fam in ordered_families(d["series"]):
        sub = d[d["series"].astype(str) == fam]
        best = sub.groupby(group_col, as_index=False)["raw_objective"].min().sort_values(group_col)
        ax.plot(
            best[group_col], best["raw_objective"],
            marker=family_marker(fam), markersize=8, linewidth=2.0,
            color=family_color(fam), markeredgecolor="black", markeredgewidth=0.5,
            label=family_label(fam),
        )
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best recorded objective")
    apply_publication_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, fancybox=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


_IMPORTANCE_LABELS = {
    "width": "classical width",
    "depth": "classical depth",
    "n_qubits": "qubits",
    "q_layers": "variational layers",
    "embedding": "coordinate embedding",
    "lr": "Learning rate",
    "log_w_mass": "Mass weight",
    "log_w_mom": "Momentum weight",
}


def _varying_param_names(trials: list) -> list[str]:
    """Parameters that exist on every trial in the group and take more than one value."""
    if not trials:
        return []
    names = []
    for name in trials[0].params:
        if name == "model_family":
            continue
        if any(name not in trial.params for trial in trials):
            continue
        if len({trial.params[name] for trial in trials}) < 2:
            continue
        names.append(name)
    return names


def _unified_distribution(trials: list, name: str):
    """One distribution covering every value this parameter took.

    Optuna refuses fANOVA when the recorded distribution object changes.
    An integer parameter uses the smallest unit-step range that contains
    every completed trial. A string parameter stays categorical.
    """
    dists = [trial.distributions[name] for trial in trials if name in trial.distributions]
    if dists and all(dist == dists[0] for dist in dists):
        return dists[0]
    values = [trial.params[name] for trial in trials if name in trial.params]
    if any(isinstance(value, str) for value in values):
        choices = tuple(dict.fromkeys(str(value) for value in values))
        return optuna.distributions.CategoricalDistribution(choices)
    low = int(min(values))
    high = int(max(values))
    return optuna.distributions.IntDistribution(low, high)


def _importance_scores(study: optuna.Study, trials: list, names: list[str]) -> dict[str, float]:
    distributions = {name: _unified_distribution(trials, name) for name in names}
    subset = optuna.create_study(direction=study.direction)
    for trial in trials:
        subset.add_trial(optuna.trial.create_trial(
            params={name: trial.params[name] for name in names},
            distributions=distributions,
            value=trial.value,
        ))
    scores = optuna.importance.get_param_importances(subset, params=names)
    return {name: float(scores[name]) for name in names}


def _importance_by_family(study: optuna.Study) -> list[tuple[str, dict]]:
    """fANOVA importance of the parameters that vary inside each family.

    Optuna's default evaluator keeps only parameters present in every trial.
    Width and depth are absent from full-quantum trials, and qubits and
    layers are absent from classical trials, so that default leaves only
    ``model_family`` with importance 1. Every completed trial in the family
    is scored together.
    """
    grouped: dict[str, list] = {}
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            continue
        family = trial.params.get("model_family", trial.user_attrs.get("model_family"))
        if family is None:
            continue
        grouped.setdefault(str(family), []).append(trial)

    panels = []
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        for family in ordered_families(grouped):
            trials = grouped[family]
            if len(trials) < 8:
                print(f"Skipping importance for {family}: {len(trials)} completed trials.")
                continue
            names = _varying_param_names(trials)
            if not names:
                continue
            try:
                scores = _importance_scores(study, trials, names)
            except Exception as exc:
                print(f"Skipping importance for {family}: {exc}")
                continue
            panels.append((family, scores))
    finally:
        optuna.logging.set_verbosity(verbosity)
    return panels


def plot_parameter_importance(study: optuna.Study, outpath: Path) -> None:
    panels = _importance_by_family(study)
    if not panels:
        print(f"Skipping {outpath.name}: no importances.")
        return
    height_ratios = [max(len(scores), 1) for _, scores in panels]
    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(7.4, 0.62 * sum(height_ratios) + 0.85 * len(panels)),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.55},
        constrained_layout=True,
    )
    for ax, (family, scores) in zip(axes[:, 0], panels):
        ranked = sorted(scores.items(), key=lambda item: item[1])
        labels = [_IMPORTANCE_LABELS.get(name, name) for name, _ in ranked]
        values = [score for _, score in ranked]
        ax.barh(
            labels, values, height=0.62,
            color=family_color(family), edgecolor="black", linewidth=0.5,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.55, len(labels) - 0.45)
        ax.set_title(family_label(family), loc="left", fontsize=13, pad=4)
        apply_publication_axes(ax)
    axes[-1, 0].set_xlabel("Relative importance within family")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def write_latex(outdir: Path) -> None:
    text = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.75\textwidth]{optuna_objective_history.pdf}
\caption{Recorded objective of each completed trial. The step line is the best value attained so far.}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.75\textwidth]{optuna_pareto_front.pdf}
\caption{Recorded objective against trainable-parameter count. Larger markers lie on the Pareto front.}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.75\textwidth]{optuna_model_family_comparison.pdf}
\caption{Distribution of the recorded objective by model family.}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.72\textwidth]{optuna_parameter_importance.pdf}
\caption{Relative importance of the hyperparameters that vary within each model family. Each panel is normalized separately.}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.92\textwidth]{model_vs_analytic.pdf}
\caption{Best completed trial compared with the analytic solution. Each row is one field. The right column is the pointwise error.}
\end{figure}
""".strip()
    path = outdir / "figures.tex"
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote: {path}")


def _field_metrics(model: np.ndarray, analytic: np.ndarray) -> dict:
    error = np.asarray(model, dtype=np.float64) - np.asarray(analytic, dtype=np.float64)
    denom = np.linalg.norm(analytic.ravel())
    rel = np.linalg.norm(error.ravel()) / denom if denom > 0.0 else np.linalg.norm(error.ravel())
    return {
        "l2": float(np.linalg.norm(error.ravel())),
        "rel_l2": float(rel),
        "linf": float(np.max(np.abs(error))) if error.size else float("nan"),
    }


def _weights_match(config_path: Path, hp) -> bool:
    if not config_path.is_file():
        return False
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    keys = ("model_family", "width", "depth", "n_qubits", "q_layers", "lr", "embedding")
    for key in keys:
        saved_value = saved.get(key)
        hp_value = getattr(hp, key, "once" if key == "embedding" else None)
        if key == "embedding":
            if str(saved_value or "once") != str(hp_value or "once"):
                return False
            continue
        if isinstance(hp_value, float):
            if saved_value is None or abs(float(saved_value) - float(hp_value)) > 1e-12:
                return False
        elif str(saved_value) != str(hp_value):
            return False
    return True


def compare_best_with_analytic(cfg, study: optuna.Study, datadir: Path, figdir: Path) -> None:
    """Train or load the best trial and write field errors against the exact solution."""
    import tensorflow as tf
    from swe_pinn.core.solver import (
        HyperParams,
        _count_trainable_parameters,
        _hyperparams_from_completed_trial,
        _is_hybrid,
        _normalize_constraints,
        build_model,
        train_adam,
    )
    from swe_pinn.physics.common import DTYPE
    from swe_pinn.problems import build_problem

    problem = build_problem(cfg)
    if problem.exact_solution is None:
        note = datadir / "analytic_comparison.txt"
        note.write_text(
            f"Problem {problem.name} has no exact solution, so no analytic comparison was written.\n",
            encoding="utf-8",
        )
        print(f"Wrote: {note}")
        return

    def recorded(trial) -> float:
        for key in ("raw_objective", "raw_error"):
            if key in trial.user_attrs:
                return float(trial.user_attrs[key])
        return float(trial.value)

    completed = [
        trial for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    best = min(completed, key=recorded)
    hp: HyperParams = _hyperparams_from_completed_trial(cfg, best)
    constraints = _normalize_constraints(cfg.constraints)
    model = build_model(hp, n_outputs=int(getattr(problem, "n_outputs", 2)))
    weights_path = datadir / "best_optuna_model.weights.h5"
    config_path = datadir / "best_optuna_model_config.json"
    source = "trained from the best trial with a fresh initialization"
    loaded = False
    if weights_path.is_file() and _weights_match(config_path, hp):
        try:
            model.load_weights(str(weights_path))
            loaded = True
            source = f"loaded weights {weights_path.name}"
        except Exception as exc:
            print(f"Saved weights do not match this architecture ({exc}). Training the best trial.")
    if not loaded:
        warmup = 0
        if _is_hybrid(hp.model_family, hp.n_qubits):
            warmup = int(round(float(getattr(cfg, "hybrid_classical_warmup_fraction", 0.0)) * int(cfg.final_adam_epochs)))
        print(
            f"Training best trial {best.number} ({hp.model_family}) "
            f"for {int(cfg.final_adam_epochs)} Adam epochs."
        )
        train_adam(
            model,
            problem,
            hp,
            int(cfg.final_adam_epochs),
            constraints,
            nx=int(cfg.final_adam_nx),
            nt=int(cfg.final_adam_nt),
            show_epoch_output=False,
            stage_label="Analytic comparison",
            hybrid_classical_warmup_epochs=warmup,
        )
        from dataclasses import asdict
        model.save_weights(str(datadir / "best_optuna_model.weights.h5"))
        (datadir / "best_optuna_model_config.json").write_text(
            json.dumps(asdict(hp), indent=2) + "\n", encoding="utf-8"
        )

    nx, nt = int(cfg.nx), int(cfg.nt)
    x_axis = np.linspace(problem.domain.x_min, problem.domain.x_max, nx)
    y_axis = np.linspace(problem.domain.t_min, problem.domain.t_max, nt)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    x_tf = tf.convert_to_tensor(x_grid.reshape(-1, 1), dtype=DTYPE)
    y_tf = tf.convert_to_tensor(y_grid.reshape(-1, 1), dtype=DTYPE)
    predicted = problem.output_fn(model, x_tf, y_tf, constraints)
    analytic = problem.exact_solution(x_tf, y_tf)
    n_fields = len(predicted)
    names = ("h", "u", "v")[:n_fields]
    labels = (r"$h$", r"$u$", r"$v$")[:n_fields]
    second_label = "y" if n_fields == 3 else "t"

    panels = []
    metric_rows = []
    for name, label, model_tf, analytic_tf in zip(names, labels, predicted, analytic):
        model_np = model_tf.numpy().reshape(nx, nt)
        analytic_np = analytic_tf.numpy().reshape(nx, nt)
        metrics = _field_metrics(model_np, analytic_np)
        metric_rows.append((name, metrics))
        panels.append((label, model_np, analytic_np, model_np - analytic_np))

    fig, axes = plt.subplots(
        n_fields, 3, figsize=(11.2, 3.15 * n_fields),
        sharex=True, sharey=True, constrained_layout=True,
    )
    if n_fields == 1:
        axes = np.array([axes])
    extent = [float(y_axis.min()), float(y_axis.max()), float(x_axis.min()), float(x_axis.max())]
    for row, (label, model_np, analytic_np, error_np) in enumerate(panels):
        vmin = float(min(np.min(model_np), np.min(analytic_np)))
        vmax = float(max(np.max(model_np), np.max(analytic_np)))
        for col, (field, title) in enumerate((
            (model_np, f"Model {label}"),
            (analytic_np, f"Analytic {label}"),
        )):
            image = axes[row, col].imshow(
                field, extent=extent, origin="lower", aspect="auto",
                vmin=vmin, vmax=vmax, cmap="cividis",
            )
            axes[row, col].set_title(title)
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
        limit = float(np.max(np.abs(error_np)))
        if limit == 0.0:
            limit = 1.0
        error_image = axes[row, 2].imshow(
            error_np, extent=extent, origin="lower", aspect="auto",
            vmin=-limit, vmax=limit, cmap="coolwarm",
        )
        axes[row, 2].set_title(f"Error {label}")
        fig.colorbar(error_image, ax=axes[row, 2], fraction=0.046, pad=0.04)
        rel = metric_rows[row][1]["rel_l2"]
        axes[row, 2].text(
            0.03, 0.97, f"rel $L_2$ = {rel:.2e}",
            transform=axes[row, 2].transAxes, va="top", ha="left", fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
        )
        axes[row, 0].set_ylabel("x")
    for col in range(3):
        axes[-1, col].set_xlabel(second_label)
    fig.savefig(figdir / "model_vs_analytic.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {figdir / 'model_vs_analytic.pdf'}")

    total_params, classical_params, quantum_params = _count_trainable_parameters(model)
    lines = [
        "Analytic comparison of the best completed trial",
        f"problem: {problem.name}",
        f"trial: {best.number}",
        f"recorded objective: {recorded(best):.6e}",
        f"family: {hp.model_family}",
        f"width: {hp.width}",
        f"depth: {hp.depth}",
        f"n_qubits: {hp.n_qubits}",
        f"q_layers: {hp.q_layers}",
        f"embedding: {hp.embedding}",
        f"lr: {hp.lr:.6e}",
        f"parameters: total={total_params}, classical={classical_params}, quantum={quantum_params}",
        f"weights: {source}",
        f"mesh: {nx} by {nt}",
        "",
        f"{'field':<8} {'L2':>14} {'relative L2':>14} {'Linf':>14}",
    ]
    for name, metrics in metric_rows:
        lines.append(
            f"{name:<8} {metrics['l2']:14.6e} {metrics['rel_l2']:14.6e} {metrics['linf']:14.6e}"
        )
    report = datadir / "analytic_comparison.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report}")
    np.savez(
        datadir / "analytic_comparison.npz",
        x=x_grid,
        y=y_grid,
        **{f"{name}_model": panel[1] for name, panel in zip(names, panels)},
        **{f"{name}_analytic": panel[2] for name, panel in zip(names, panels)},
        **{f"{name}_error": panel[3] for name, panel in zip(names, panels)},
    )


def write_search_figures(df: pd.DataFrame, study: optuna.Study, datadir: Path, figdir: Path) -> None:
    df.to_csv(datadir / "trials.csv", index=False)
    print(f"Wrote: {datadir / 'trials.csv'}")
    plot_objective_history(df, figdir / "optuna_objective_history.pdf")
    scatter_by_family(
        df, "total_params", "raw_objective",
        "Total trainable parameters", "Recorded objective",
        figdir / "optuna_error_vs_parameters.pdf",
        logx=True, logy=True,
    )
    plot_pareto(df, figdir / "optuna_pareto_front.pdf")
    plot_family_comparison(df, figdir / "optuna_model_family_comparison.pdf")
    plot_group_best(df, "q_layers", "Quantum circuit layers", figdir / "optuna_quantum_layers_vs_error.pdf")
    plot_group_best(df, "n_qubits", "Number of qubits", figdir / "optuna_qubits_vs_error.pdf")
    plot_parameter_importance(study, figdir / "optuna_parameter_importance.pdf")
    write_latex(figdir)


_PROBLEM_CONFIGS = {
    "dam_break": "user_input.py.damBreak",
    "tidal_test": "user_input.py.tide",
    "steady_rotation": "user_input.py.steady_rotation",
    "steady_vortex": "user_input.py.steady_vortex",
    "steady_eddies": "user_input.py.steady_eddies",
    "transcritical_bump": "user_input.py.bump",
}


def _config_for_problem(problem: str):
    """Settings for the test case named by the database, not the active problem.

    ``user_input.py`` holds one active test at a time. A saved
    ``user_input.py.<case>`` file holds that case's constraints, grids, and
    learning rate. The physical dataclass in the active file is still what
    ``build_problem`` reads.
    """
    from dataclasses import replace
    import sys
    import types

    live = USER_PINN_INPUTS
    if live.problem_name == problem:
        return live
    snapshot_name = _PROBLEM_CONFIGS.get(problem)
    snapshot = ROOT / snapshot_name if snapshot_name else None
    if snapshot is not None and snapshot.is_file():
        module_name = f"user_input_case_{problem}"
        module = types.ModuleType(module_name)
        module.__file__ = str(snapshot)
        sys.modules[module_name] = module
        exec(compile(snapshot.read_text(encoding="utf-8"), str(snapshot), "exec"), module.__dict__)
        cfg = module.USER_PINN_INPUTS
    else:
        cfg = live
    if cfg.problem_name != problem:
        cfg = replace(cfg, problem_name=problem)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Write the study report and analytic comparison for one problem.")
    parser.add_argument("--db", default=None, help="SQLite path or sqlite:/// URL.")
    parser.add_argument("--study", default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--target", type=float, default=None, help="Optional recorded-objective ceiling in the text report.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the analytic comparison training step.")
    args = parser.parse_args(argv)

    cfg = USER_PINN_INPUTS
    problem = args.problem if args.problem is not None else cfg.problem_name
    db_arg = args.db if args.db is not None else cfg.storage_url
    db_path = resolve_db(db_arg, problem)
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")

    # A database still stored in analysis/<problem>/ names that test case.
    # A database stored in analysis/<database name>/ does not: that folder
    # matches the file name, and the test case stays the one requested above.
    # Reports and figures use the database name either way.
    analysis_root = (ROOT / "analysis").resolve()
    try:
        relative = db_path.resolve().relative_to(analysis_root)
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) >= 2 and relative.parts[0] != db_path.stem:
        problem = relative.parts[0]
    outdir = ROOT / "analysis" / db_path.stem
    datadir = outdir / "data"
    figdir = outdir / "figures"
    datadir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    if args.study is not None:
        study_name = args.study
    elif args.db is None:
        study_name = cfg.study_name
    else:
        study_name = None
    study = get_study(sqlite_url(db_path), study_name)
    print(f"Study {study.study_name} from {db_path}")
    print(f"Output directory: {outdir}")
    df = study_to_dataframe(study)
    write_tradeoff_report(df, datadir / "param_tradeoff.txt", args.target)
    write_search_figures(df, study, datadir, figdir)
    if args.skip_train:
        print("Analytic comparison skipped (--skip-train).")
        return
    case_cfg = _config_for_problem(problem)
    print(f"Analytic comparison for {problem}.")
    compare_best_with_analytic(case_cfg, study, datadir, figdir)


if __name__ == "__main__":
    main()
