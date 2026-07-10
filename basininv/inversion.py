"""Waveform inversion of basin geometry (+ sediment velocity).

Least-squares waveform misfit against the synthetic "observed" data, mild
Tikhonov smoothing on the control-node depths, finite-difference gradients
(one full multi-shot forward per parameter, run in parallel worker
processes), and scipy L-BFGS-B on top.

With ~10 parameters and small grids this is a practical, fully general
scheme; an adjoint-state gradient is the upgrade path for larger runs.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .basin import BasinParameterization, Materials
from .survey import Survey, forward_from_params


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
        n_nodes = param.ncx * param.ncy
        h = np.full(param.n_params, 0.5 * survey.grid.dx)
        if param.invert_vs:
            h[-1] = 10.0
        self.fd_step = h if fd_step is None else fd_step
        self.log = InversionLog()
        self._cache = {}

    # -------------------------------------------------------------- misfit
    def _data_misfit(self, params):
        key = tuple(np.round(params, 6))
        if key in self._cache:
            return self._cache[key]
        d = forward_from_params(self.survey, self.param, params, self.mat,
                                workers=self.workers)
        phi = 0.5 * float(np.sum((d.astype(np.float64) - self.d_obs) ** 2)) / self.norm
        self._cache[key] = phi
        if len(self._cache) > 200:
            self._cache.pop(next(iter(self._cache)))
        return phi

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
    def _fd_jobs(self, params):
        jobs = [params.copy()]
        for i in range(len(params)):
            p = params.copy()
            p[i] += self.fd_step[i]
            jobs.append(p)
        return jobs

    def gradient(self, params):
        """Forward-difference gradient; the (n+1) multi-shot forwards are
        spread over the worker pool at single-shot granularity."""
        jobs = self._fd_jobs(params)
        # run each parameter set with 1 worker but many sets concurrently
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = [ex.submit(_misfit_worker,
                              (self.survey, self.param, p, self.mat,
                               self.d_obs, self.norm))
                    for p in jobs]
            phis = [f.result() for f in futs]
        phi0 = phis[0]
        self._cache[tuple(np.round(params, 6))] = phi0
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
            f, g = self.gradient(np.asarray(x, float))
            self.log.params.append(np.asarray(x, float).copy())
            self.log.misfit.append(f)
            if verbose:
                el = time.time() - self.log.t0
                print(f"  eval {len(self.log.misfit):3d}  misfit {f:.6e}  "
                      f"[{el:7.1f}s]", flush=True)
            return f, g

        res = minimize(fun, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-10,
                                "gtol": 1e-12, "maxls": 8})
        return res, self.log


def _misfit_worker(args):
    survey, param, params, mat, d_obs, norm = args
    d = forward_from_params(survey, param, params, mat, workers=1)
    return 0.5 * float(np.sum((d.astype(np.float64) - d_obs) ** 2)) / norm
