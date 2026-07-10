"""Ambient-noise mode: distributed random sources and H/V spectral ratios.

This provides the second data type (closer to real microtremor field
campaigns): many randomly placed, randomly timed force sources excite the
model for a long window; receivers record three components; H/V spectral
ratios are then extracted per station.  The same inversion machinery can be
pointed at an HVSR-curve misfit instead of the waveform misfit.
"""
from __future__ import annotations

import numpy as np

from .basin import GridSpec
from .solver import ElasticSolver3D, ricker


def random_noise_sources(grid: GridSpec, n_sources, nt, dt, rng,
                         f_band=(0.5, 8.0), margin_cells=16):
    """Random surface force sources with random Ricker bursts through time."""
    sources = []
    for _ in range(n_sources):
        ix = rng.integers(margin_cells, grid.nx - margin_cells)
        iy = rng.integers(margin_cells, grid.ny - margin_cells)
        comp = rng.choice(["fx", "fy", "fz"])
        w = np.zeros(nt, np.float32)
        n_bursts = max(1, int(nt * dt / 2.0))
        for _ in range(n_bursts):
            f0 = rng.uniform(*f_band)
            it0 = rng.integers(0, max(1, nt - int(2.0 / (f0 * dt))))
            burst = ricker(f0, min(nt - it0, int(3.0 / (f0 * dt))), dt, t0=1.2 / f0)
            w[it0:it0 + len(burst)] += rng.uniform(0.3, 1.0) * burst
        sources.append((int(ix), int(iy), 1, comp, w))
    return sources


def simulate_noise(vp, vs, rho, dx, rec_xy, duration, n_sources=60,
                   seed=0, cfl=0.45):
    rng = np.random.default_rng(seed)
    solver = ElasticSolver3D(vp, vs, rho, dx, cfl=cfl)
    nt = int(np.ceil(duration / solver.dt))
    sources = random_noise_sources(
        GridSpec(*vp.shape, dx), n_sources, nt, solver.dt, rng)
    seis, _ = solver.run(nt, sources=sources, receivers=rec_xy)
    return seis, solver.dt


def hvsr(seis, dt, f_min=0.3, f_max=10.0, smooth=7):
    """H/V spectral ratio per receiver from 3-component records.

    Returns (freqs, hv) with hv shaped (nrec, nfreq)."""
    nrec, _, nt = seis.shape
    win = np.hanning(nt).astype(np.float32)
    spec = np.abs(np.fft.rfft(seis * win, axis=-1))
    f = np.fft.rfftfreq(nt, dt)
    if smooth > 1:
        k = np.hanning(smooth)
        k /= k.sum()
        spec = np.apply_along_axis(lambda s: np.convolve(s, k, "same"), -1, spec)
    h = np.sqrt(0.5 * (spec[:, 0] ** 2 + spec[:, 1] ** 2))
    v = spec[:, 2]
    sel = (f >= f_min) & (f <= f_max)
    return f[sel], (h / np.maximum(v, 1e-30))[:, sel]
