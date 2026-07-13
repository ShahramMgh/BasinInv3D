#!/usr/bin/env python3
"""BasinInv3D Studio — live web front-end for the whole synthetic experiment.

Serves a dashboard that runs the full pipeline in a background thread and
streams every stage:

  1. build    — random "true" 3D basin + acquisition figure
  2. observe  — forward-model each shot, livestreaming wavefield frames
  3. invert   — L-BFGS-B on control-node depths + sediment vs, streaming the
                current basin, difference map and misfit curve every eval
  4. report   — final comparison figures + score

Stdlib http.server only (no flask on this machine).  Run with:

    /usr/bin/python3 webapp/app.py [--port 8642]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from basininv import (BasinParameterization, ElasticSolver3D, GridSpec,
                      Materials, MultiscaleInversion, Survey, build_model,
                      forward_from_params, gaussian_basin, ricker)
from basininv import viz

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT, "run")
os.makedirs(RUN_DIR, exist_ok=True)

# ncx / ncx_min bracket the coarse-to-fine multiscale node schedule; maxiter is
# the L-BFGS-B iteration budget *per scale*.
PRESETS = {
    "fast": dict(nx=44, ny=44, nz=26, dx=32.0, f0=2.0, t_max=1.4,
                 nshot_side=2, nrec_side=6, ncx=3, ncx_min=2, maxiter=6,
                 workers=3, seed=1, vs_true=400.0, vs_init=550.0,
                 init_depth_frac=0.15, noise_pct=0.0, margin=13),
    "standard": dict(nx=60, ny=60, nz=36, dx=25.0, f0=2.4, t_max=2.2,
                     nshot_side=2, nrec_side=8, ncx=4, ncx_min=3, maxiter=8,
                     workers=3, seed=1, vs_true=400.0, vs_init=550.0,
                     init_depth_frac=0.15, noise_pct=0.0, margin=15),
    "high": dict(nx=72, ny=72, nz=42, dx=20.0, f0=2.6, t_max=2.6,
                 nshot_side=3, nrec_side=9, ncx=5, ncx_min=3, maxiter=10,
                 workers=3, seed=1, vs_true=400.0, vs_init=550.0,
                 init_depth_frac=0.15, noise_pct=0.0, margin=16),
}

# smoothing relaxes geometrically from coarse (strong) to fine (weak) scale
_SMOOTH_HI, _SMOOTH_LO = 2.5e-2, 3e-3


def build_schedule(ncx_min, ncx_max, iters):
    ncs = list(range(max(2, ncx_min), max(2, ncx_max) + 1)) or [ncx_max]
    stages = []
    for i, nc in enumerate(ncs):
        frac = i / max(1, len(ncs) - 1)
        sw = _SMOOTH_HI * (_SMOOTH_LO / _SMOOTH_HI) ** frac
        stages.append((nc, nc, sw, int(iters)))
    return stages


def _surf3d(grid, zb_true, zb_inv=None, target=30):
    """Coarse, JSON-friendly interface payload for the live in-browser 3D view.
    Depths are positive-down in metres; the client renders z = -depth."""
    step = max(1, grid.nx // target, grid.ny // target)
    xs = grid.x[::step]
    ys = grid.y[::step]
    out = {"x": [round(float(v), 1) for v in xs],
           "y": [round(float(v), 1) for v in ys],
           "true": [[round(float(v), 1) for v in row]
                    for row in zb_true[::step, ::step]],
           "max_depth": float(max(zb_true.max(), 1.0))}
    if zb_inv is not None:
        out["inv"] = [[round(float(v), 1) for v in row]
                      for row in zb_inv[::step, ::step]]
        out["max_depth"] = float(max(zb_true.max(), zb_inv.max(), 1.0))
    return out

_LOCK = threading.Lock()


def _fresh_state():
    return {
        "running": False, "stop": False, "stage": "idle", "stage_note": "",
        "progress": 0.0, "log": [], "misfit": [], "rms": [],
        "images": {}, "result": None, "error": None, "config": None,
        "t_start": None, "surf3d": None, "stage_idx": 0, "n_stages": 1,
        "vs_now": None,
    }


STATE = _fresh_state()


def log(msg):
    with _LOCK:
        STATE["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        del STATE["log"][:-400]
    print(msg, flush=True)


def set_state(**kw):
    with _LOCK:
        STATE.update(kw)


def bump(name):
    with _LOCK:
        STATE["images"][name] = STATE["images"].get(name, 0) + 1


class StopRequested(Exception):
    pass


def _check_stop():
    with _LOCK:
        if STATE["stop"]:
            raise StopRequested


def random_basin(grid, rng, margin):
    """2-3 random smooth depressions inside the tapered interior."""
    lx, ly, lz = grid.x[-1], grid.y[-1], grid.nz * grid.dx
    bumps = []
    for _ in range(int(rng.integers(2, 4))):
        bumps.append(dict(
            amp=float(rng.uniform(0.30, 0.55) * lz),
            x0=float(rng.uniform(0.35, 0.65) * lx),
            y0=float(rng.uniform(0.35, 0.65) * ly),
            sx=float(rng.uniform(0.10, 0.24) * lx),
            sy=float(rng.uniform(0.10, 0.24) * ly),
            theta=float(rng.uniform(-1.2, 1.2)),
        ))
    return gaussian_basin(grid, bumps, margin_cells=margin)


# ===================================================================== job
def pipeline(cfg):
    try:
        t_all = time.time()
        set_state(t_start=t_all)

        # ---------------------------------------------------- 1. build
        set_state(stage="build", stage_note="constructing true basin",
                  progress=0.0)
        grid = GridSpec(nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"], dx=cfg["dx"])
        margin = cfg["margin"]
        rng = np.random.default_rng(cfg["seed"])
        zb_true = random_basin(grid, rng, margin)
        mat_true = Materials(vs_sed=cfg["vs_true"])
        vp, vs, rho = build_model(grid, zb_true, mat_true)
        survey = Survey.regular(grid, nshot_side=cfg["nshot_side"],
                                nrec_side=cfg["nrec_side"],
                                margin_cells=margin + 2,
                                f0=cfg["f0"], t_max=cfg["t_max"])
        viz.render_model_summary(grid, zb_true, survey, mat_true,
                                 os.path.join(RUN_DIR, "model.png"))
        set_state(surf3d=_surf3d(grid, zb_true))
        bump("model")
        log(f"grid {grid.nx}x{grid.ny}x{grid.nz} (dx={grid.dx:.0f} m), "
            f"max depth {zb_true.max():.0f} m, "
            f"{len(survey.shot_xy)} shots / {len(survey.rec_xy)} receivers")
        _check_stop()

        # -------------------------------------------------- 2. observe
        set_state(stage="observe", stage_note="forward-modeling shots")
        nshot = len(survey.shot_xy)
        d_obs_list = []
        peak = [1e-30]

        for s_i, shot in enumerate(survey.shot_xy):
            solver = ElasticSolver3D(vp, vs, rho, grid.dx, cfl=survey.cfl)
            nt = int(np.ceil(survey.t_max / solver.dt))
            w = ricker(survey.f0, nt, solver.dt)
            render_every = max(1, nt // 24)

            def on_step(it, ntt, sol, s_i=s_i, shot=shot,
                        render_every=render_every):
                _check_stop()
                if it % render_every == 0 or it == ntt - 1:
                    # robust scale: high percentile of the displayed slices,
                    # smoothed over frames so the colors don't flicker
                    p = max(float(np.percentile(np.abs(sol.vz[:, :, 0]), 99.0)),
                            float(np.percentile(np.abs(sol.vz[:, shot[1], :]), 99.0)),
                            1e-30)
                    peak[0] = max(p, 0.5 * peak[0])
                    viz.render_wavefield(
                        grid, sol, zb_true, shot, it, ntt,
                        os.path.join(RUN_DIR, "wave.png"),
                        clim=1.5 * peak[0])
                    bump("wave")
                    set_state(progress=(s_i * ntt + it + 1) / (nshot * ntt),
                              stage_note=f"shot {s_i + 1}/{nshot}, "
                                         f"step {it + 1}/{ntt}")

            log(f"shot {s_i + 1}/{nshot} at cell {tuple(int(v) for v in shot)} "
                f"({nt} steps, dt={solver.dt * 1e3:.2f} ms)")
            seis, _ = solver.run(
                nt, sources=[(int(shot[0]), int(shot[1]),
                              survey.src_depth_cells, "fz", w)],
                receivers=survey.rec_xy, on_step=on_step)
            d_obs_list.append(seis)

        d_obs = np.stack(d_obs_list)
        if cfg["noise_pct"] > 0:
            sigma = cfg["noise_pct"] / 100.0 * float(np.abs(d_obs).std())
            d_obs = (d_obs + rng.normal(0, sigma, d_obs.shape)).astype(np.float32)
            log(f"added {cfg['noise_pct']:.0f}% gaussian noise to records")
        np.save(os.path.join(RUN_DIR, "d_obs.npy"), d_obs)
        log(f"observed data ready: shape {d_obs.shape}")
        _check_stop()

        # --------------------------------------------------- 3. invert
        set_state(stage="invert", stage_note="starting multiscale L-BFGS-B",
                  progress=0.0)
        max_depth = 0.80 * grid.nz * grid.dx
        schedule = build_schedule(cfg.get("ncx_min", 3), cfg["ncx"],
                                  cfg["maxiter"])
        set_state(n_stages=len(schedule))
        # coarsest parameterization only to report the initial (flat) guess
        p0 = BasinParameterization(grid, ncx=schedule[0][0], ncy=schedule[0][1],
                                   margin_cells=margin)
        x0 = p0.pack(np.full((schedule[0][0], schedule[0][1]),
                             cfg["init_depth_frac"] * max_depth), cfg["vs_init"])
        zb_init = p0.depth_map(x0)
        rms0 = float(np.sqrt(np.mean((zb_init - zb_true) ** 2)))
        log("multiscale schedule: " + " -> ".join(
            f"{s[0]}x{s[1]}(sw={s[2]:.1g})" for s in schedule) +
            f"; initial RMS depth error {rms0:.1f} m")

        mat_inv = Materials()   # bedrock known, sediment vs from params
        # total gradient evals across all scales, for a smooth global progress bar
        evals_total = sum(s[3] + 1 for s in schedule)
        eval_count = [0]

        def on_stage(i, n, param):
            set_state(stage_idx=i,
                      stage_note=f"scale {i + 1}/{n}: {param.ncx}x{param.ncy} "
                                 f"nodes ({param.n_params} unknowns)")
            log(f"scale {i + 1}/{n}: {param.ncx}x{param.ncy} control nodes, "
                f"{param.n_params} unknowns")

        def on_forward(i, done, total):
            _check_stop()
            frac = (eval_count[0] + done / total) / max(1, evals_total)
            set_state(progress=min(1.0, frac))

        def on_eval(i, param, k, f, x):
            zb_cur = param.depth_map(x)
            rms = float(np.sqrt(np.mean((zb_cur - zb_true) ** 2)))
            _, vs_cur = param.unpack(x)
            eval_count[0] += 1
            with _LOCK:
                STATE["misfit"].append(float(f))
                STATE["rms"].append(rms)
                mis = list(STATE["misfit"])
            viz.render_live_inversion(grid, zb_true, zb_cur,
                                      os.path.join(RUN_DIR, "inversion.png"))
            viz.plot_convergence(mis,
                                 os.path.join(RUN_DIR, "convergence.png"))
            set_state(surf3d=_surf3d(grid, zb_true, zb_cur), vs_now=float(vs_cur),
                      progress=min(1.0, eval_count[0] / max(1, evals_total)))
            bump("inversion")
            bump("convergence")
            log(f"  scale {i + 1} eval {k}: misfit {f:.4e}, "
                f"RMS depth {rms:.1f} m, vs {vs_cur:.0f} m/s")
            _check_stop()

        inv = MultiscaleInversion(survey, grid, d_obs, mat_inv, schedule,
                                  margin_cells=margin, max_depth=max_depth,
                                  workers=cfg["workers"])
        inv.on_stage = on_stage
        inv.on_forward = on_forward
        inv.on_eval = on_eval
        param, x_final = inv.run(x0, vs_init=cfg["vs_init"])
        n_evals = eval_count[0]
        _check_stop()

        # --------------------------------------------------- 4. report
        set_state(stage="report", stage_note="final figures", progress=0.0)
        zb_inv = param.depth_map(x_final)
        _, vs_inv = param.unpack(x_final)
        rms = float(np.sqrt(np.mean((zb_inv - zb_true) ** 2)))
        d_syn = forward_from_params(survey, param, x_final, mat_inv,
                                    workers=cfg["workers"])
        dt = survey.t_max / d_obs.shape[-1]
        viz.plot_depth_maps(grid, zb_true, zb_inv, zb_init,
                            os.path.join(RUN_DIR, "depth_maps.png"),
                            survey=survey)
        viz.plot_interface_3d(grid, zb_true, zb_inv,
                              os.path.join(RUN_DIR, "interface_3d.png"))
        viz.plot_seismograms(d_obs, d_syn, dt,
                             path=os.path.join(RUN_DIR, "seismograms.png"))
        set_state(surf3d=_surf3d(grid, zb_true, zb_inv), vs_now=float(vs_inv))
        for n in ("depth_maps", "interface_3d", "seismograms"):
            bump(n)

        with _LOCK:
            mis0 = STATE["misfit"][0] if STATE["misfit"] else None
        result = dict(
            misfit0=mis0, misfit=float(STATE["misfit"][-1]) if STATE["misfit"]
            else None, rms0=rms0, rms=rms,
            vs_true=cfg["vs_true"], vs_inv=float(vs_inv),
            n_evals=n_evals, elapsed=time.time() - t_all,
            nodes=f"{param.ncx}x{param.ncy}",
            max_depth_true=float(zb_true.max()),
            max_depth_inv=float(zb_inv.max()))
        set_state(stage="done", stage_note="", progress=1.0, result=result)
        log(f"DONE in {result['elapsed']:.0f}s: RMS depth error "
            f"{rms0:.0f} m -> {rms:.0f} m, vs {vs_inv:.0f} m/s "
            f"(true {cfg['vs_true']:.0f})")
    except StopRequested:
        set_state(stage="stopped", stage_note="")
        log("stopped by user")
    except Exception:
        err = traceback.format_exc()
        set_state(stage="error", error=err)
        log("ERROR:\n" + err)
    finally:
        set_state(running=False)


# ================================================================== server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "static", "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/status":
            with _LOCK:
                snap = {k: v for k, v in STATE.items() if k != "log"}
                snap = json.loads(json.dumps(snap, default=str))
                snap["log"] = STATE["log"][-150:]
            self._send(200, json.dumps(snap).encode())
        elif path.startswith("/img/"):
            name = os.path.basename(path)
            fp = os.path.join(RUN_DIR, name)
            if name.endswith(".png") and os.path.isfile(fp):
                with open(fp, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"{}")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}')
            return
        if self.path == "/api/start":
            with _LOCK:
                if STATE["running"]:
                    self._send(409, b'{"error":"already running"}')
                    return
            preset = PRESETS.get(body.get("preset", "fast"), PRESETS["fast"])
            cfg = {**preset}
            for k, v in body.items():
                if k in cfg:
                    try:
                        cfg[k] = type(cfg[k])(v)
                    except (TypeError, ValueError):
                        pass
            with _LOCK:
                STATE.clear()
                STATE.update(_fresh_state())
            set_state(running=True, config=cfg)
            log(f"starting pipeline: {cfg}")
            threading.Thread(target=pipeline, args=(cfg,), daemon=True).start()
            self._send(200, b'{"ok":true}')
        elif self.path == "/api/stop":
            set_state(stop=True)
            log("stop requested...")
            self._send(200, b'{"ok":true}')
        else:
            self._send(404, b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"BasinInv3D Studio at http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
