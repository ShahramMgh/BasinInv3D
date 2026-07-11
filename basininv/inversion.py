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

from .basin import BasinParameterization, Materials
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
                 d_obs, mat: Materials, smooth_weight=1e-3,
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
        nd, _ = self.param.unpack(params)
        d2 = 0.0
        if self.param.ncx > 2:
            d2 += np.sum(np.diff(nd, 2, axis=0) ** 2)
        if self.param.ncy > 2:
            d2 += np.sum(np.diff(nd, 2, axis=1) ** 2)
        scale = (self.survey.grid.nz * self.survey.grid.dx) ** 2
        return self.smooth_weight * d2 / scale

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
