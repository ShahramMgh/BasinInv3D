#!/usr/bin/env python3
"""End-to-end synthetic experiment:

1. Build a "true" 3D sediment-filled valley (sum of Gaussians) in bedrock.
2. Record active-source shots on a surface receiver grid (observed data).
3. Invert basin control-node depths + sediment vs from a bad initial guess.
4. Plot recovered vs. true geometry, waveform fits, convergence.

Usage:  python3 scripts/run_demo.py [--quick] [--maxiter N] [--workers N]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from basininv import (BasinParameterization, GridSpec, Materials, Survey,
                      WaveformInversion, build_model, fit_nodes_to_map,
                      forward, forward_from_params, gaussian_basin)
from basininv import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small grid / short record for a fast sanity run")
    ap.add_argument("--maxiter", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------ setup
    if args.quick:
        grid = GridSpec(nx=50, ny=50, nz=30, dx=30.0)
        t_max, f0 = 1.6, 2.2
        nshot_side, nrec_side = 2, 6
        ncx = ncy = 3
    else:
        grid = GridSpec(nx=66, ny=66, nz=40, dx=25.0)
        t_max, f0 = 2.4, 2.4
        nshot_side, nrec_side = 2, 8
        ncx = ncy = 3

    margin = 16
    mat_true = Materials(vs_sed=400.0)

    bumps = [
        dict(amp=0.55 * grid.nz * grid.dx, x0=0.42 * grid.x[-1],
             y0=0.55 * grid.y[-1], sx=0.22 * grid.x[-1],
             sy=0.13 * grid.y[-1], theta=0.5),
        dict(amp=0.30 * grid.nz * grid.dx, x0=0.65 * grid.x[-1],
             y0=0.38 * grid.y[-1], sx=0.12 * grid.x[-1],
             sy=0.18 * grid.y[-1], theta=-0.3),
    ]
    zb_true = gaussian_basin(grid, bumps, margin_cells=margin)
    vp, vs, rho = build_model(grid, zb_true, mat_true)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}, dx={grid.dx} m, "
          f"max basin depth {zb_true.max():.0f} m")

    survey = Survey.regular(grid, nshot_side=nshot_side, nrec_side=nrec_side,
                            margin_cells=margin + 2, f0=f0, t_max=t_max)
    print(f"{len(survey.shot_xy)} shots, {len(survey.rec_xy)} receivers, "
          f"f0={f0} Hz, T={t_max} s")

    # ------------------------------------------------- observed data
    t0 = time.time()
    d_obs = forward(survey, vp, vs, rho, workers=args.workers)
    print(f"observed data computed in {time.time() - t0:.1f}s, "
          f"shape {d_obs.shape}, peak |v| {np.abs(d_obs).max():.3e}")
    np.save(os.path.join(args.out, "d_obs.npy"), d_obs)

    # ------------------------------------------------------ inversion
    param = BasinParameterization(grid, ncx=ncx, ncy=ncy, margin_cells=margin)
    max_depth = 0.75 * grid.nz * grid.dx

    # initial guess: uniform shallow flat basin, wrong sediment velocity
    x0 = param.pack(np.full((ncx, ncy), 0.15 * max_depth), 550.0)
    nodes_best = fit_nodes_to_map(param, zb_true)   # best achievable target
    print(f"unknowns: {param.n_params} "
          f"({ncx}x{ncy} depth nodes + sediment vs)")

    mat_inv = Materials()   # bedrock known; vs_sed overwritten by params
    inv = WaveformInversion(survey, param, d_obs, mat_inv,
                            workers=args.workers)
    print("initial misfit:", f"{inv.misfit(x0):.4e}")

    res, log = inv.run(x0, max_depth=max_depth, maxiter=args.maxiter)
    nd_inv, vs_inv = param.unpack(res.x)
    print("\n--- result ---")
    print(f"final misfit  {res.fun:.4e}  ({len(log.misfit)} gradient evals)")
    print(f"sediment vs   true 400.0  inverted {vs_inv:.1f} m/s")
    print("node depths (true-sampled vs inverted):")
    for r_t, r_i in zip(nodes_best, nd_inv):
        print("  ", " ".join(f"{a:6.0f}/{b:6.0f}" for a, b in zip(r_t, r_i)))

    zb_inv = param.depth_map(res.x)
    zb_init = param.depth_map(x0)
    rms = np.sqrt(np.mean((zb_inv - zb_true) ** 2))
    rms0 = np.sqrt(np.mean((zb_init - zb_true) ** 2))
    print(f"depth-map RMS error: initial {rms0:.1f} m -> inverted {rms:.1f} m")

    np.savez(os.path.join(args.out, "inversion_result.npz"),
             x=res.x, misfit=np.array(log.misfit), zb_true=zb_true,
             zb_inv=zb_inv, zb_init=zb_init)

    # ------------------------------------------------------- figures
    d_syn = forward_from_params(survey, param, res.x, mat_inv,
                                workers=args.workers)
    dt = survey.t_max / d_obs.shape[-1]
    viz.plot_depth_maps(grid, zb_true, zb_inv, zb_init,
                        path=os.path.join(args.out, "depth_maps.png"),
                        survey=survey)
    viz.plot_interface_3d(grid, zb_true, zb_inv,
                          path=os.path.join(args.out, "interface_3d.png"))
    viz.plot_seismograms(d_obs, d_syn, dt,
                         path=os.path.join(args.out, "seismograms.png"))
    viz.plot_convergence(log.misfit,
                         path=os.path.join(args.out, "convergence.png"))
    print(f"figures written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
