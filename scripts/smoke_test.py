#!/usr/bin/env python3
"""Fast sanity checks: stability, free surface, basin response, timing."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basininv import (BasinParameterization, GridSpec, Materials,
                      ElasticSolver3D, build_model, gaussian_basin, ricker)

grid = GridSpec(nx=40, ny=40, nz=28, dx=30.0)
mat = Materials()
zb = gaussian_basin(grid, [dict(amp=350, x0=600, y0=600, sx=250, sy=180)],
                    margin_cells=12)
vp, vs, rho = build_model(grid, zb, mat)
print("model:", vp.shape, "vp", vp.min(), "-", vp.max(),
      "vs", vs.min(), "-", vs.max())

solver = ElasticSolver3D(vp, vs, rho, grid.dx)
nt = int(1.2 / solver.dt)
w = ricker(2.5, nt, solver.dt)
rec = np.array([[20, 20], [12, 12], [28, 20]])
t0 = time.time()
seis, _ = solver.run(nt, sources=[(20, 20, 1, "fz", w)], receivers=rec)
el = time.time() - t0
print(f"{nt} steps in {el:.1f}s ({1e3 * el / nt:.1f} ms/step), dt={solver.dt * 1e3:.2f} ms")

assert np.all(np.isfinite(seis)), "NaN/Inf in seismograms!"
peak = np.abs(seis).max()
tail = np.abs(seis[:, :, -nt // 10:]).max()
print(f"peak amplitude {peak:.3e}, tail amplitude {tail:.3e} "
      f"(ratio {tail / peak:.3f})")
assert peak > 0, "dead wavefield"
assert tail < 2.0 * peak, "likely instability (growing tail)"

# basin vs no-basin must differ (the signal the inversion feeds on)
vp2, vs2, rho2 = build_model(grid, np.zeros_like(zb), mat)
solver2 = ElasticSolver3D(vp2, vs2, rho2, grid.dx)
seis2, _ = solver2.run(nt, sources=[(20, 20, 1, "fz", w)], receivers=rec)
rel = np.abs(seis - seis2).max() / peak
print(f"basin vs halfspace relative difference: {rel:.2f}")
assert rel > 0.1, "basin has no effect on records?"

print("SMOKE TEST PASSED")
