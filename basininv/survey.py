"""Acquisition geometry and (parallel) forward modeling.

A Survey bundles shot positions, a surface receiver grid, the source wavelet
and the recording length.  `forward()` runs one solver per shot; shots are
independent, so they are farmed out to worker processes.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .basin import GridSpec, Materials, build_model
from .solver import ElasticSolver3D, ricker


@dataclass
class Survey:
    grid: GridSpec
    shot_xy: np.ndarray          # (nshot, 2) grid indices
    rec_xy: np.ndarray           # (nrec, 2) grid indices
    f0: float = 2.5              # Ricker peak frequency, Hz
    t_max: float = 2.5           # record length, s
    src_depth_cells: int = 1     # vertical force just below the surface
    cfl: float = 0.45

    @classmethod
    def regular(cls, grid: GridSpec, nshot_side=2, nrec_side=7, margin_cells=18,
                **kw):
        """Shots on an nshot_side^2 grid, receivers on an nrec_side^2 grid,
        both inside the taper margin."""
        def lattice(n):
            return np.round(np.linspace(margin_cells, grid.nx - 1 - margin_cells, n)).astype(int)

        sx = lattice(nshot_side)
        rx = lattice(nrec_side)
        shot = np.array([(i, j) for i in sx for j in sx])
        rec = np.array([(i, j) for i in rx for j in rx])
        return cls(grid=grid, shot_xy=shot, rec_xy=rec, **kw)

    def nt_dt(self, vp_max):
        dt = self.cfl * self.grid.dx / vp_max
        return int(np.ceil(self.t_max / dt)), dt


def _run_shot(args):
    vp, vs, rho, dx, cfl, shot, rec, f0, t_max, src_depth = args
    solver = ElasticSolver3D(vp, vs, rho, dx, cfl=cfl)
    nt = int(np.ceil(t_max / solver.dt))
    w = ricker(f0, nt, solver.dt)
    src = [(int(shot[0]), int(shot[1]), src_depth, "fz", w)]
    seis, _ = solver.run(nt, sources=src, receivers=rec)
    return seis


def forward(survey: Survey, vp, vs, rho, workers=1):
    """Seismograms for every shot: returns array (nshot, nrec, 3, nt)."""
    jobs = [(vp, vs, rho, survey.grid.dx, survey.cfl, s, survey.rec_xy,
             survey.f0, survey.t_max, survey.src_depth_cells)
            for s in survey.shot_xy]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_run_shot, jobs))
    else:
        out = [_run_shot(j) for j in jobs]
    return np.stack(out)


def forward_from_params(survey: Survey, param, params, mat: Materials,
                        workers=1):
    """Convenience: parameter vector -> model -> synthetic data."""
    zb = param.depth_map(params)
    m = Materials(**{**mat.__dict__})
    _, vs_sed = param.unpack(params)
    if vs_sed is not None:
        m.vs_sed = vs_sed
    vp, vs, rho = build_model(survey.grid, zb, m)
    return forward(survey, vp, vs, rho, workers=workers)
