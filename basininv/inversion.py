"""Waveform inversion of basin geometry (+ sediment velocity).

Least-squares waveform misfit against the synthetic "observed" data, mild
Tikhonov smoothing on the control-node depths, finite-difference gradients
(one full multi-shot forward per parameter), and scipy L-BFGS-B on top.

The FD forwards run in a *persistent* process pool whose workers receive the
static problem data (survey, parameterization, observed records) once at
startup; each job then ships only a parameter vector.  Live hooks:

    on_eval(k, misfit, params)   after every gradient evaluation
    on_forward(done, total)      as the FD forwards of one gradient finish

With ~10 parameters and small grids this is a practical, fully general
scheme; an adjoint-state gradient is the upgrade path for larger runs.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .basin import BasinParameterization, Materials, resample_params
from .survey import Survey, forward_from_params

# ------------------------------------------------------------------ workers
_W: dict = {}


def _init_worker(survey, param, mat, d_obs, norm):
    _W.update(survey=survey, param=param, mat=mat, d_obs=d_obs, norm=norm)


def _misfit_worker(params):
    d = forward_from_params(_W["survey"], _W["param"], params, _W["mat"],
                            workers=1)
    return 0.5 * float(np.sum((d.astype(np.float64) - _W["d_obs"]) ** 2)) / _W["norm"]


@dataclass
class InversionLog:
    params: list = field(default_factory=list)
    misfit: list = field(default_factory=list)
    t0: float = field(default_factory=time.time)


class WaveformInversion:
    def __init__(self, survey: Survey, param: BasinParameterization,
                 d_obs, mat: Materials, smooth_weight=1e-2,
                 workers=4, fd_step=None):
        self.survey = survey
        self.param = param
        self.d_obs = d_obs.astype(np.float32)
        self.mat = mat
        self.norm = float(np.sum(self.d_obs.astype(np.float64) ** 2))
        self.smooth_weight = smooth_weight
        self.workers = workers
        # FD step: meters for depths, m/s for vs
        h = np.full(param.n_params, 0.5 * survey.grid.dx)
        if param.invert_vs:
            h[-1] = 10.0
        self.fd_step = h if fd_step is None else fd_step
        self.log = InversionLog()
        self.on_eval = None       # callable(k, misfit, params)
        self.on_forward = None    # callable(done, total)
        self._pool = None

    # ------------------------------------------------------------- pool
    def _get_pool(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_init_worker,
                initargs=(self.survey, self.param, self.mat,
                          self.d_obs, self.norm))
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -------------------------------------------------------------- misfit
    def _data_misfit(self, params):
        return self._get_pool().submit(_misfit_worker, np.asarray(params, float)).result()

    def _smooth_penalty(self, params):
        """Roughness of the node-depth grid.  First differences dominate the
        penalty: they are what suppresses checkerboard patterns, which the
        waveform misfit alone cannot discriminate against."""
        nd, _ = self.param.unpack(params)
        r = (np.sum(np.diff(nd, 1, axis=0) ** 2)
             + np.sum(np.diff(nd, 1, axis=1) ** 2))
        if self.param.ncx > 2:
            r += 0.5 * np.sum(np.diff(nd, 2, axis=0) ** 2)
        if self.param.ncy > 2:
            r += 0.5 * np.sum(np.diff(nd, 2, axis=1) ** 2)
        scale = (self.survey.grid.nz * self.survey.grid.dx) ** 2
        return self.smooth_weight * r / scale

    def misfit(self, params):
        return self._data_misfit(params) + self._smooth_penalty(params)

    # ------------------------------------------------------------ gradient
    def gradient(self, params):
        """Forward-difference gradient; the (n+1) multi-shot forwards run
        concurrently on the persistent pool."""
        params = np.asarray(params, float)
        jobs = [params.copy()]
        for i in range(len(params)):
            p = params.copy()
            p[i] += self.fd_step[i]
            jobs.append(p)

        pool = self._get_pool()
        futs = {pool.submit(_misfit_worker, p): i for i, p in enumerate(jobs)}
        phis = [0.0] * len(jobs)
        done = 0
        for f in as_completed(futs):
            phis[futs[f]] = f.result()
            done += 1
            if self.on_forward is not None:
                self.on_forward(done, len(jobs))

        phi0 = phis[0]
        g = np.array([(phis[i + 1] - phi0) / self.fd_step[i]
                      for i in range(len(params))])
        # regularization gradient by cheap FD (no simulations involved)
        for i in range(len(params)):
            p = params.copy()
            p[i] += self.fd_step[i]
            g[i] += (self._smooth_penalty(p) - self._smooth_penalty(params)) / self.fd_step[i]
        return phi0 + self._smooth_penalty(params), g

    # ---------------------------------------------------------------- run
    def run(self, x0, max_depth, maxiter=15, verbose=True):
        bounds = self.param.bounds(max_depth)

        def fun(x):
            x = np.asarray(x, float)
            f, g = self.gradient(x)
            self.log.params.append(x.copy())
            self.log.misfit.append(f)
            if verbose:
                el = time.time() - self.log.t0
                print(f"  eval {len(self.log.misfit):3d}  misfit {f:.6e}  "
                      f"[{el:7.1f}s]", flush=True)
            if self.on_eval is not None:
                self.on_eval(len(self.log.misfit), f, x.copy())
            return f, g

        try:
            res = minimize(fun, np.asarray(x0, float), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-10,
                                    "gtol": 1e-12, "maxls": 8})
        finally:
            self.close()
        return res, self.log


@dataclass
class MultiscaleStage:
    ncx: int
    ncy: int
    smooth_weight: float
    maxiter: int


class MultiscaleInversion:
    """Coarse-to-fine geometry inversion.

    Solves the basin geometry on a sequence of increasingly fine control-node
    grids, warm-starting each stage from the previous solution and relaxing the
    roughness penalty as resolution grows.  A coarse strongly-smoothed stage
    locks in the large-scale basin shape (avoiding checkerboard minima), then
    finer weakly-smoothed stages add detail — recovering the central depth that
    a single coarse grid leaves too shallow, without introducing artifacts.

    The hooks mirror :class:`WaveformInversion` but always report the *active*
    parameterization so callers can map parameters -> depth without tracking the
    schedule themselves:

        on_stage(i, nstages, param)         when a stage begins
        on_eval(i, param, k, misfit, x)     after every gradient evaluation
        on_forward(i, done, total)          as one gradient's forwards finish
    """

    def __init__(self, survey: Survey, grid, d_obs, mat: Materials,
                 stages, margin_cells, max_depth, workers=4, invert_vs=True):
        self.survey = survey
        self.grid = grid
        self.d_obs = d_obs
        self.mat = mat
        self.stages = [s if isinstance(s, MultiscaleStage) else MultiscaleStage(*s)
                       for s in stages]
        self.margin_cells = margin_cells
        self.max_depth = max_depth
        self.workers = workers
        self.invert_vs = invert_vs
        self.on_stage = None
        self.on_eval = None
        self.on_forward = None
        self.history = []          # (param, x) per stage

    def run(self, x0, vs_init):
        x = None
        param = None
        for i, st in enumerate(self.stages):
            new_param = BasinParameterization(
                self.grid, ncx=st.ncx, ncy=st.ncy,
                margin_cells=self.margin_cells, invert_vs=self.invert_vs)
            if x is None:
                x_start = x0 if x0 is not None else new_param.pack(
                    np.zeros((st.ncx, st.ncy)), vs_init)
            else:
                x_start = resample_params(param, x, new_param)
            param = new_param
            if self.on_stage is not None:
                self.on_stage(i, len(self.stages), param)

            inv = WaveformInversion(self.survey, param, self.d_obs, self.mat,
                                    smooth_weight=st.smooth_weight,
                                    workers=self.workers)
            if self.on_forward is not None:
                inv.on_forward = (lambda i=i: (
                    lambda done, total: self.on_forward(i, done, total)))()
            if self.on_eval is not None:
                inv.on_eval = (lambda i=i, param=param: (
                    lambda k, f, xx: self.on_eval(i, param, k, f, xx)))()
            res, _ = inv.run(x_start, max_depth=self.max_depth,
                             maxiter=st.maxiter, verbose=False)
            x = res.x
            self.history.append((param, x))
        return param, x
