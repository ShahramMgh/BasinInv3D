"""Figures: depth maps, 3D interface surfaces, seismograms, convergence."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_depth_maps(grid, zb_true, zb_inv, zb_init=None, path="depth_maps.png",
                    survey=None):
    n = 4 if zb_init is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), constrained_layout=True)
    vmax = max(zb_true.max(), zb_inv.max())
    ext = [0, grid.x[-1], 0, grid.y[-1]]
    panels = [("True basin depth", zb_true, "viridis", (0, vmax))]
    if zb_init is not None:
        panels.append(("Initial guess", zb_init, "viridis", (0, vmax)))
    panels.append(("Inverted", zb_inv, "viridis", (0, vmax)))
    panels.append(("Inverted - true", zb_inv - zb_true, "coolwarm",
                   (-0.5 * vmax, 0.5 * vmax)))
    for ax, (title, data, cmap, clim) in zip(axes, panels):
        im = ax.imshow(data.T, origin="lower", extent=ext, cmap=cmap,
                       vmin=clim[0], vmax=clim[1])
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="depth (m)")
        if survey is not None:
            dx = grid.dx
            ax.plot(survey.rec_xy[:, 0] * dx, survey.rec_xy[:, 1] * dx, "k.",
                    ms=2, alpha=0.5)
            ax.plot(survey.shot_xy[:, 0] * dx, survey.shot_xy[:, 1] * dx, "r*",
                    ms=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_interface_3d(grid, zb_true, zb_inv, path="interface_3d.png"):
    fig = plt.figure(figsize=(12, 5))
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    for i, (title, zb) in enumerate([("True interface", zb_true),
                                     ("Inverted interface", zb_inv)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(X, Y, -zb, cmap="viridis_r", linewidth=0,
                        antialiased=True)
        ax.set_zlim(-1.1 * max(zb_true.max(), zb_inv.max(), 1.0), 0)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_seismograms(d_obs, d_syn, dt, shot=0, comp=2, path="seismograms.png",
                     max_traces=25):
    nrec = d_obs.shape[1]
    step = max(1, nrec // max_traces)
    idx = np.arange(0, nrec, step)
    t = np.arange(d_obs.shape[-1]) * dt
    amp = np.abs(d_obs[shot, :, comp]).max() + 1e-30
    fig, ax = plt.subplots(figsize=(9, 7))
    for row, r in enumerate(idx):
        off = row * 2.0
        ax.plot(t, d_obs[shot, r, comp] / amp + off, "k", lw=0.8,
                label="observed" if row == 0 else None)
        ax.plot(t, d_syn[shot, r, comp] / amp + off, "r--", lw=0.8,
                label="inverted" if row == 0 else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("receiver (offset traces)")
    ax.set_title(f"Vertical-component waveform fit, shot {shot}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_model_summary(grid, zb, survey, mat, path):
    """Stage-1 figure: true depth map with acquisition + an x-z section."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ext = [0, grid.x[-1], 0, grid.y[-1]]
    im = axes[0].imshow(zb.T, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="basin depth (m)")
    dx = grid.dx
    axes[0].plot(survey.rec_xy[:, 0] * dx, survey.rec_xy[:, 1] * dx, "w.",
                 ms=3, label="receivers")
    axes[0].plot(survey.shot_xy[:, 0] * dx, survey.shot_xy[:, 1] * dx, "r*",
                 ms=10, label="shots")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("True basin + acquisition")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    j = grid.ny // 2
    axes[1].fill_between(grid.x, -zb[:, j], 0, color="#c8a97e",
                         label=f"sediments (vs={mat.vs_sed:.0f} m/s)")
    axes[1].fill_between(grid.x, -grid.z[-1], -zb[:, j], color="#8a8f98",
                         label=f"bedrock (vs={mat.vs_rock:.0f} m/s)")
    axes[1].set_xlim(0, grid.x[-1])
    axes[1].set_ylim(-grid.z[-1], 0)
    axes[1].set_title(f"Section y = {j * dx:.0f} m")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_wavefield(grid, solver, zb, shot_ij, it, nt, path, clim):
    """Live two-panel wavefield frame: surface vz + vertical slice with the
    basin interface overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ext = [0, grid.x[-1], 0, grid.y[-1]]
    axes[0].imshow(solver.vz[:, :, 0].T, origin="lower", extent=ext,
                   cmap="RdBu_r", vmin=-clim, vmax=clim)
    axes[0].plot([shot_ij[0] * grid.dx], [shot_ij[1] * grid.dx], "k*", ms=10)
    axes[0].set_title(f"surface vz   t = {it * solver.dt:.2f} s "
                      f"({it + 1}/{nt})")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    j = int(shot_ij[1])
    axes[1].imshow(solver.vz[:, j, :].T, origin="upper",
                   extent=[0, grid.x[-1], grid.z[-1], 0],
                   cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto")
    axes[1].plot(grid.x, zb[:, j], "k-", lw=1.2)
    axes[1].set_title(f"section y = {j * grid.dx:.0f} m (interface in black)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def render_live_inversion(grid, zb_true, zb_cur, path):
    """Live three-panel state of the inversion: true, current, difference."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    ext = [0, grid.x[-1], 0, grid.y[-1]]
    vmax = max(zb_true.max(), zb_cur.max(), 1.0)
    for ax, (title, data, cmap, clim) in zip(axes, [
            ("True (hidden from inversion)", zb_true, "viridis", (0, vmax)),
            ("Current inverted basin", zb_cur, "viridis", (0, vmax)),
            ("Current - true", zb_cur - zb_true, "coolwarm",
             (-0.5 * vmax, 0.5 * vmax))]):
        im = ax.imshow(data.T, origin="lower", extent=ext, cmap=cmap,
                       vmin=clim[0], vmax=clim[1])
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="depth (m)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_convergence(misfits, path="convergence.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(misfits) + 1), misfits, "o-")
    ax.set_xlabel("gradient evaluation")
    ax.set_ylabel("normalized misfit")
    ax.set_title("Inversion convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
