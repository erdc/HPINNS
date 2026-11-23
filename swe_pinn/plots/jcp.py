"""! @file swe_pinn/plots/jcp.py
@brief JCP-style plotting helpers.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "axes.linewidth": 1.0,
})


def save_two_col_losses(adam_hist, lbfgs_hist, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(adam_hist["total"], label="total")
    ax.plot(adam_hist["mass"], label="mass")
    ax.plot(adam_hist["mom"], label="momentum")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log(Loss)")
    ax.set_title("Adam history")
    ax.legend(loc="upper right")

    ax = axes[1]
    if len(lbfgs_hist["total"]) > 0:
        ax.plot(lbfgs_hist["total"], label="total")
        ax.plot(lbfgs_hist["mass"], label="mass")
        ax.plot(lbfgs_hist["mom"], label="momentum")
        ax.set_yscale("log")
        ax.set_xlabel("L-BFGS-B evaluation")
        ax.set_ylabel("log(Loss)")
        ax.set_title("L-BFGS-B history")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No L-BFGS-B history", ha="center", va="center")
        ax.axis("off")

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_comparison_2d(TIMEH, XKM, A, M, title_a, title_m, clabel, outpath, *, xlabel="time", ylabel="x"):
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.2), sharey=True, constrained_layout=True
    )

    vmin = min(np.min(A), np.min(M))
    vmax = max(np.max(A), np.max(M))

    im0 = axes[0].imshow(
        A,
        extent=[TIMEH.min(), TIMEH.max(), XKM.min(), XKM.max()],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(title_a)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    axes[1].imshow(
        M,
        extent=[TIMEH.min(), TIMEH.max(), XKM.min(), XKM.max()],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(title_m)
    axes[1].set_xlabel(xlabel)

    cbar = fig.colorbar(im0, ax=axes, pad=0.03)
    cbar.set_label(clabel)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_trial_model_vs_analytic_2x2(
    TIME,
    X,
    H_model,
    H_analytic,
    U_model,
    U_analytic,
    outpath,
    *,
    title=None,
):
    """Save a 2x2 model-vs-analytic spacetime figure for the trained model.

    Layout:
      top-left     model depth h(x,t)
      top-right    analytic depth h(x,t)
      bottom-left  model velocity u(x,t)
      bottom-right analytic velocity u(x,t)
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True, constrained_layout=True
    )

    extent = [float(TIME.min()), float(TIME.max()), float(X.min()), float(X.max())]
    h_vmin = min(np.min(H_model), np.min(H_analytic))
    h_vmax = max(np.max(H_model), np.max(H_analytic))
    u_vmin = min(np.min(U_model), np.min(U_analytic))
    u_vmax = max(np.max(U_model), np.max(U_analytic))

    panels = (
        (axes[0, 0], H_model, "Model depth $h$", h_vmin, h_vmax, r"$h$"),
        (axes[0, 1], H_analytic, "Analytic depth $h$", h_vmin, h_vmax, r"$h$"),
        (axes[1, 0], U_model, "Model velocity $u$", u_vmin, u_vmax, r"$u$"),
        (axes[1, 1], U_analytic, "Analytic velocity $u$", u_vmin, u_vmax, r"$u$"),
    )

    for ax, field, panel_title, vmin, vmax, clabel in panels:
        im = ax.imshow(
            field,
            extent=extent,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(clabel)

    axes[0, 0].set_ylabel("x")
    axes[1, 0].set_ylabel("x")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")

    if title:
        fig.suptitle(title)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_three_field_vs_analytic(Y, X, fields, outpath, *, title=None):
    """Save model and analytic panels for depth and both velocity components.

    ``fields`` is a sequence of ``(name, model, analytic, colorbar label)``.
    Arrays are shaped ``(nx, ny)``. The horizontal axis is ``y``.
    """
    fig, axes = plt.subplots(
        len(fields), 2, figsize=(11.0, 3.6 * len(fields)),
        sharex=True, sharey=True, constrained_layout=True,
    )
    extent = [float(np.min(Y)), float(np.max(Y)), float(np.min(X)), float(np.max(X))]
    for row, (name, model, analytic, clabel) in enumerate(fields):
        vmin = min(np.min(model), np.min(analytic))
        vmax = max(np.max(model), np.max(analytic))
        for col, (field, side) in enumerate(((model, "Model"), (analytic, "Analytic"))):
            im = axes[row, col].imshow(
                field, extent=extent, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
            )
            axes[row, col].set_title(f"{side} {name}")
            cbar = fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            cbar.set_label(clabel)
        axes[row, 0].set_ylabel("x")
    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    if title:
        fig.suptitle(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_error_2d(TIMEH, XKM, E1, E2, outpath):
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.2), sharey=True, constrained_layout=True
    )

    vmin = min(np.min(E1), np.min(E2))
    vmax = max(np.max(E1), np.max(E2))

    im0 = axes[0].imshow(
        E1,
        extent=[TIMEH.min(), TIMEH.max(), XKM.min(), XKM.max()],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Elevation error")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("x")

    axes[1].imshow(
        E2,
        extent=[TIMEH.min(), TIMEH.max(), XKM.min(), XKM.max()],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Velocity error")
    axes[1].set_xlabel("time")

    cbar = fig.colorbar(im0, ax=axes, pad=0.03)
    cbar.set_label("error")

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_profiles_boundary(X, ETA_A, ETA_M, tau_test, u_left_m, u_left_a, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    profile_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    model_labeled = False

    for frac in profile_fracs:
        j = int(frac * (ETA_A.shape[1] - 1))

        axes[0].plot(
            X[:, j],
            ETA_A[:, j],
            "-",
            color="black",
            linewidth=1.6,
            label=fr"$t={frac:.2f}$",
        )

        step = max(1, len(X[:, j]) // 25)
        axes[0].plot(
            X[::step, j],
            ETA_M[::step, j],
            "o",
            color="black",
            markersize=3.5,
            markerfacecolor="none",
            label="model" if not model_labeled else None,
        )
        model_labeled = True

    axes[0].set_title("Elevation profiles")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\eta$")
    axes[0].legend(loc="lower left", ncol=2, fontsize=9)

    axes[1].plot(
        tau_test.squeeze(),
        u_left_a.squeeze(),
        "-",
        color="black",
        linewidth=1.6,
        label="analytic",
    )
    axes[1].plot(
        tau_test.squeeze(),
        u_left_m.squeeze(),
        "o",
        color="black",
        markersize=3.5,
        markerfacecolor="none",
        label="model",
    )

    axes[1].set_title("Boundary check")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$u(x_{\min},t)$")
    axes[1].legend(loc="lower left")

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
