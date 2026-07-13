"""Figures for the microtremor (HVSR) studio: station map, HVSR fits, Vs
cross-sections, bedrock-depth comparison, convergence."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VS_CMAP = "turbo"       # shared with the in-browser 3-D Vs viewer


def render_stations(grid, true_depth, xy, path, obs_f0=None):
    """Station layout over the true bedrock-depth map."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ext = [0, grid.x[-1], 0, grid.y[-1]]
    im = ax.imshow(true_depth.T, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.85, label="true bedrock depth (m)")
    if obs_f0 is not None:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=obs_f0, cmap="turbo_r",
                        edgecolor="w", s=70, linewidth=0.8, zorder=3)
        fig.colorbar(sc, ax=ax, shrink=0.85, label="observed HVSR f₀ (Hz)")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c="w", edgecolor="k", s=55, zorder=3)
    ax.set_title(f"{len(xy)} microtremor stations")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.savefig(path, dpi=125); plt.close(fig)


def render_hvsr_fits(freqs, hv_obs, hv_pred, xy, path, n_show=6):
    """A few stations: observed vs modelled H/V curve."""
    n = min(n_show, len(xy))
    idx = np.linspace(0, len(xy) - 1, n).astype(int)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow),
                             constrained_layout=True, squeeze=False)
    for a, si in enumerate(idx):
        ax = axes[a // ncol][a % ncol]
        ax.semilogx(freqs, hv_obs[si], "k", lw=1.6, label="observed")
        if hv_pred is not None:
            ax.semilogx(freqs, hv_pred[si], "r--", lw=1.5, label="modelled")
        ax.set_title(f"station {si}  ({xy[si,0]:.0f}, {xy[si,1]:.0f}) m",
                     fontsize=9)
        ax.grid(alpha=0.3, which="both")
        if a == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel("f (Hz)", fontsize=8); ax.set_ylabel("H/V", fontsize=8)
    for a in range(n, nrow * ncol):
        axes[a // ncol][a % ncol].axis("off")
    fig.savefig(path, dpi=120); plt.close(fig)


def _section(ax, grid, vol, j, axis_label, title, vmin, vmax):
    if axis_label == "x":
        data = vol[:, j, :]            # (nx, nz)
        ext = [0, grid.x[-1], grid.z[-1], 0]
    else:
        data = vol[j, :, :]            # (ny, nz)
        ext = [0, grid.y[-1], grid.z[-1], 0]
    im = ax.imshow(data.T, origin="upper", extent=ext, cmap=VS_CMAP,
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{axis_label} (m)"); ax.set_ylabel("depth (m)")
    return im


def render_vs_sections(model, params_inv, path, params_true=None):
    """Vs cross-sections through the basin centre (true vs inverted)."""
    vol_i = model.vs_volume(params_inv)
    ic, jc = model.grid.nx // 2, model.grid.ny // 2
    vmin, vmax = float(vol_i.min()), float(np.percentile(vol_i, 99))
    rows = 2 if params_true is not None else 1
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.4 * rows),
                             constrained_layout=True, squeeze=False)
    if params_true is not None:
        vol_t = model.vs_volume(params_true)
        vmax = max(vmax, float(np.percentile(vol_t, 99)))
        _section(axes[0][0], model.grid, vol_t, jc, "x", "True Vs — section y=centre", vmin, vmax)
        _section(axes[0][1], model.grid, vol_t, ic, "y", "True Vs — section x=centre", vmin, vmax)
        r = 1
    else:
        r = 0
    im = _section(axes[r][0], model.grid, vol_i, jc, "x", "Inverted Vs — section y=centre", vmin, vmax)
    _section(axes[r][1], model.grid, vol_i, ic, "y", "Inverted Vs — section x=centre", vmin, vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Vs (m/s)")
    fig.savefig(path, dpi=120); plt.close(fig)


def render_depth_compare(grid, true_depth, inv_depth, xy, path):
    """True / inverted / difference bedrock-depth maps."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    ext = [0, grid.x[-1], 0, grid.y[-1]]
    vmax = max(true_depth.max(), inv_depth.max(), 1.0)
    panels = [("True bedrock depth", true_depth, "viridis", (0, vmax)),
              ("Inverted bedrock depth", inv_depth, "viridis", (0, vmax)),
              ("Inverted − true", inv_depth - true_depth, "coolwarm",
               (-0.5 * vmax, 0.5 * vmax))]
    for ax, (title, data, cmap, clim) in zip(axes, panels):
        im = ax.imshow(data.T, origin="lower", extent=ext, cmap=cmap,
                       vmin=clim[0], vmax=clim[1])
        ax.scatter(xy[:, 0], xy[:, 1], c="k", s=8, alpha=0.5)
        ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="depth (m)")
    fig.savefig(path, dpi=120); plt.close(fig)


def render_live_vs(model, params_cur, true_depth, path):
    """Live inversion figure: current bedrock depth, its error, centre section."""
    inv_depth = model.interface_depths(params_cur)[-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    ext = [0, model.grid.x[-1], 0, model.grid.y[-1]]
    vmax = max(true_depth.max(), inv_depth.max(), 1.0)
    im0 = axes[0].imshow(inv_depth.T, origin="lower", extent=ext, cmap="viridis",
                         vmin=0, vmax=vmax)
    axes[0].set_title("Current bedrock depth")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="depth (m)")
    im1 = axes[1].imshow((inv_depth - true_depth).T, origin="lower", extent=ext,
                         cmap="coolwarm", vmin=-0.5 * vmax, vmax=0.5 * vmax)
    axes[1].set_title("Current − true")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Δ depth (m)")
    vol = model.vs_volume(params_cur)
    im2 = _section(axes[2], model.grid, vol, model.grid.ny // 2, "x",
                   "Vs section y=centre", float(vol.min()),
                   float(np.percentile(vol, 99)))
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Vs (m/s)")
    for ax in axes[:2]:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.savefig(path, dpi=115); plt.close(fig)


def render_microtremor(ts, dt, path, station=0):
    """Example 3-component microtremor time series for one station."""
    t = np.arange(ts.shape[-1]) * dt
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    for c, lab, col in zip(range(3), ("N", "E", "Z"), ("#2c7", "#27c", "#c62")):
        ax.plot(t, ts[c] / (np.abs(ts).max() + 1e-30) + (2 - c) * 2.2,
                lw=0.6, color=col, label=lab)
    ax.set_title(f"Example ambient-noise record — station {station}")
    ax.set_xlabel("time (s)"); ax.set_yticks([]); ax.legend(loc="upper right",
                                                            fontsize=8)
    fig.savefig(path, dpi=115); plt.close(fig)


def render_uncertainty(grid, mean_depth, std_depth, true_depth, xy, path,
                       origin=(0.0, 0.0)):
    """Ensemble bedrock-depth uncertainty: mean, per-cell std (uncertainty),
    and — when a ground truth exists — whether it sits within ±2σ."""
    ox, oy = origin
    ncol = 3 if true_depth is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol + 0.4, 4.0),
                             constrained_layout=True)
    ext = [ox, ox + grid.x[-1], oy, oy + grid.y[-1]]
    vmax = max(mean_depth.max(),
               true_depth.max() if true_depth is not None else 0.0, 1.0)
    im0 = axes[0].imshow(mean_depth.T, origin="lower", extent=ext,
                         cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("Ensemble-mean bedrock depth")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="depth (m)")
    im1 = axes[1].imshow(std_depth.T, origin="lower", extent=ext, cmap="magma",
                         vmin=0)
    axes[1].set_title("Depth uncertainty (±1σ)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="σ depth (m)")
    if true_depth is not None:
        within = (np.abs(mean_depth - true_depth) <= 2 * std_depth + 1e-9)
        axes[2].imshow(within.T, origin="lower", extent=ext, cmap="RdYlGn",
                       vmin=0, vmax=1)
        axes[2].set_title(f"Truth within ±2σ  ({100*within.mean():.0f}% of area)")
    for ax in axes:
        ax.scatter(xy[:, 0], xy[:, 1], c="k", s=8, alpha=0.5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.savefig(path, dpi=120); plt.close(fig)


# ------------------------------------------------------- field-data figures


def render_field_map(xy, f0, path, sids=None, ok=None, grid=None,
                     depth=None, origin=(0.0, 0.0)):
    """Field campaign map: stations coloured by measured f0; SESAME failures
    ringed in red; optional inverted bedrock-depth background."""
    ox, oy = origin
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    if depth is not None and grid is not None:
        ext = [ox, ox + grid.x[-1], oy, oy + grid.y[-1]]
        im = ax.imshow(depth.T, origin="lower", extent=ext, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.85, label="inverted bedrock depth (m)")
    xy = np.atleast_2d(xy)
    f0 = np.asarray(f0, float)
    edge = ["w" if (ok is None or ok[i]) else "#f85149"
            for i in range(len(xy))]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=f0, cmap="turbo_r", s=90,
                    edgecolor=edge, linewidth=1.4, zorder=3,
                    norm=matplotlib.colors.LogNorm(vmin=max(f0.min(), 1e-2),
                                                   vmax=f0.max()))
    fig.colorbar(sc, ax=ax, shrink=0.85, label="measured f₀ (Hz)")
    if sids is not None:
        for i, s in enumerate(sids):
            ax.annotate(s, xy[i], textcoords="offset points", xytext=(5, 5),
                        fontsize=6.5, color="#333" if depth is None else "w")
    bad = 0 if ok is None else int(len(ok) - np.count_nonzero(ok))
    ax.set_title(f"{len(xy)} field stations — f₀ map"
                 + (f"  ({bad} fail SESAME)" if bad else ""))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=125); plt.close(fig)


def render_proc_qc(results, sids, path, n_show=9):
    """Per-station processing QC: mean H/V ±σ band, picked f0, window counts
    and SESAME verdicts."""
    n = min(n_show, len(results))
    idx = np.linspace(0, len(results) - 1, n).astype(int)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow),
                             constrained_layout=True, squeeze=False)
    for a, si in enumerate(idx):
        r = results[si]
        ax = axes[a // ncol][a % ncol]
        ax.fill_between(r["freqs"], r["hv"] * np.exp(-r["sigma"]),
                        r["hv"] * np.exp(r["sigma"]), alpha=0.25,
                        color="#4675ed", lw=0)
        ax.semilogx(r["freqs"], r["hv"], color="#16324f", lw=1.6)
        ax.axvline(r["f0"], color="#e05a2a", lw=1.2, ls="--")
        rel = r.get("reliable", True); clr = r.get("clear_peak", True)
        tag = ("✓ reliable" if rel else "✗ unreliable") + \
              (" · clear peak" if clr else " · unclear peak")
        col = "#1a7f37" if (rel and clr) else "#b35900" if rel else "#c0392b"
        ax.set_title(f"{sids[si]}   f₀={r['f0']:.2f} Hz", fontsize=9)
        note = f"{r.get('n_win', '–')} win"
        if r.get("n_rej"):
            note += f" ({r['n_rej']} rej)"
        if r.get("kind") == "hv":
            note = "pre-processed .hv"
        ax.text(0.03, 0.94, f"{tag}\n{note}", transform=ax.transAxes,
                fontsize=7.2, va="top", color=col)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("f (Hz)", fontsize=8); ax.set_ylabel("H/V", fontsize=8)
    for a in range(n, nrow * ncol):
        axes[a // ncol][a % ncol].axis("off")
    fig.savefig(path, dpi=120); plt.close(fig)


def render_live_field(model, params_cur, path, origin=(0.0, 0.0), xy=None):
    """Live field-inversion figure (no ground truth): current bedrock depth
    and a centre Vs cross-section."""
    ox, oy = origin
    inv_depth = model.interface_depths(params_cur)[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    ext = [ox, ox + model.grid.x[-1], oy, oy + model.grid.y[-1]]
    im0 = axes[0].imshow(inv_depth.T, origin="lower", extent=ext,
                         cmap="viridis", vmin=0)
    axes[0].set_title("Current bedrock depth")
    if xy is not None:
        axes[0].scatter(xy[:, 0], xy[:, 1], c="w", s=10, alpha=0.7,
                        edgecolor="k", linewidth=0.4)
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="depth (m)")
    vol = model.vs_volume(params_cur)
    im1 = _section(axes[1], model.grid, vol, model.grid.ny // 2, "x",
                   "Vs section y=centre", float(vol.min()),
                   float(np.percentile(vol, 99)))
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Vs (m/s)")
    fig.savefig(path, dpi=115); plt.close(fig)


def render_field_depth(model, params, xy, path, origin=(0.0, 0.0),
                       true_depth=None):
    """Final field result: per-interface depth maps (+ optional truth diff)."""
    ox, oy = origin
    depth = model.interface_depths(params)
    nL = depth.shape[0]
    ncol = nL + (1 if true_depth is not None else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.3 * ncol + 0.4, 4.0),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    ext = [ox, ox + model.grid.x[-1], oy, oy + model.grid.y[-1]]
    vmax = float(max(depth[-1].max(), 1.0))
    for L in range(nL):
        name = "bedrock" if L == nL - 1 else f"interface {L + 1}"
        im = axes[L].imshow(depth[L].T, origin="lower", extent=ext,
                            cmap="viridis", vmin=0, vmax=vmax)
        axes[L].set_title(f"Depth of {name}")
        axes[L].scatter(xy[:, 0], xy[:, 1], c="k", s=8, alpha=0.5)
        axes[L].set_xlabel("x (m)"); axes[L].set_ylabel("y (m)")
        fig.colorbar(im, ax=axes[L], shrink=0.8, label="depth (m)")
    if true_depth is not None:
        d = depth[-1] - true_depth
        im = axes[-1].imshow(d.T, origin="lower", extent=ext, cmap="coolwarm",
                             vmin=-0.5 * vmax, vmax=0.5 * vmax)
        axes[-1].set_title("Bedrock: inverted − truth (validation)")
        axes[-1].scatter(xy[:, 0], xy[:, 1], c="k", s=8, alpha=0.5)
        axes[-1].set_xlabel("x (m)"); axes[-1].set_ylabel("y (m)")
        fig.colorbar(im, ax=axes[-1], shrink=0.8, label="Δ depth (m)")
    fig.savefig(path, dpi=120); plt.close(fig)


def plot_convergence(misfits, path):
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.semilogy(np.arange(1, len(misfits) + 1), misfits, "o-", color="#a371f7")
    ax.set_xlabel("evaluation"); ax.set_ylabel("HVSR misfit")
    ax.set_title("Inversion convergence"); ax.grid(alpha=0.3, which="both")
    fig.savefig(path, dpi=120); plt.close(fig)
