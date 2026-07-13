#!/usr/bin/env python3
"""Ambient-noise mode demo: microtremor wavefield + H/V spectral ratios.

Simulates distributed random sources over a basin model, records 3-component
motion at stations across the valley, and extracts H/V spectral ratios.
Physics check: stations over deep sediments should show an H/V peak near the
1D resonance f0 ~ vs_sed / (4 h), moving to higher frequency (and fading) as
the sediments thin toward the basin edge.

Usage: /usr/bin/python3 scripts/run_noise_demo.py [--duration 8.0]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basininv import GridSpec, Materials, build_model, gaussian_basin
from basininv.noise import hvsr, simulate_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = GridSpec(nx=44, ny=44, nz=26, dx=30.0)
    mat = Materials(vs_sed=400.0)
    zb = gaussian_basin(grid, [dict(amp=420.0, x0=0.5 * grid.x[-1],
                                    y0=0.5 * grid.y[-1], sx=0.20 * grid.x[-1],
                                    sy=0.20 * grid.y[-1])], margin_cells=12)
    vp, vs, rho = build_model(grid, zb, mat)

    # stations along a radial line: basin center -> edge -> outside
    ic = grid.nx // 2
    rec = np.array([[ic + k, ic] for k in (0, 4, 7, 10, 14)])
    h_sed = zb[rec[:, 0], rec[:, 1]]
    print("station sediment thicknesses:", np.round(h_sed), "m")
    print(f"expected 1D resonance f0=vs/4h:",
          np.round(mat.vs_sed / (4 * np.maximum(h_sed, 1)), 2), "Hz")

    t0 = time.time()
    seis, dt = simulate_noise(vp, vs, rho, grid.dx, rec,
                              duration=args.duration, n_sources=50, seed=3)
    print(f"noise simulation: {args.duration}s of wavefield in "
          f"{time.time() - t0:.0f}s wall time, dt={dt * 1e3:.2f} ms")
    assert np.all(np.isfinite(seis))

    f, hv = hvsr(seis, dt, f_min=0.4, f_max=6.0)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(rec)))
    for i in range(len(rec)):
        ax.plot(f, hv[i], color=shades[i], lw=1.8,
                label=f"h = {h_sed[i]:.0f} m sediments")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("H/V spectral ratio")
    ax.set_title("Ambient-noise H/V across the basin (center -> edge)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path = os.path.join(args.out, "hvsr.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("figure:", os.path.abspath(path))

    ipk = np.argmax(hv[:, :], axis=1)
    print("H/V peak frequencies:", np.round(f[ipk], 2), "Hz")


if __name__ == "__main__":
    main()
