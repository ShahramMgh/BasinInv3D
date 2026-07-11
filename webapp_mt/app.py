#!/usr/bin/env python3
"""BasinInv3D — Microtremor Studio.

A separate live web app (companion to the active-source ``webapp/``) that
inverts ambient-microtremor **HVSR curves** from a scattered set of surface
stations into a **3-D multi-layer sediment Vs structure**.  It shares the same
``basininv`` package and stdlib-only http.server pattern.

Pipeline streamed to the browser:

  1. data    — build an imaginary layered basin, place stations, synthesise a
               microtremor record per station and extract its H/V curve
  2. invert  — L-BFGS-B on the layer-thickness control nodes (+ optionally the
               layer Vs), matching the modelled HVSR to the observed curves;
               any parameter can be fixed (known bedrock, fixed Vs jump, …)
  3. report  — recovered Vs cross-sections, bedrock-depth score, HVSR fits

Run:  /usr/bin/python3 webapp_mt/app.py [--port 8643]
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from basininv import GridSpec
from basininv import hvsr as H
from basininv import hvsr_viz as V
from basininv.noise import hvsr as extract_hvsr
from scipy.interpolate import RectBivariateSpline

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT, "run")
os.makedirs(RUN_DIR, exist_ok=True)

PRESETS = {
    "fast":     dict(nx=36, ny=36, nz=20, dx=18.0, n_side=5, ncx=3,
                     maxiter=45, margin=6, fmin=0.3, fmax=12.0, nfreq=130,
                     max_total_depth=140.0),
    "standard": dict(nx=44, ny=44, nz=24, dx=15.0, n_side=6, ncx=3,
                     maxiter=70, margin=7, fmin=0.3, fmax=12.0, nfreq=150,
                     max_total_depth=150.0),
    "high":     dict(nx=52, ny=52, nz=28, dx=13.0, n_side=7, ncx=4,
                     maxiter=90, margin=8, fmin=0.25, fmax=14.0, nfreq=170,
                     max_total_depth=160.0),
}

_LOCK = threading.Lock()


def _fresh():
    return {"running": False, "stop": False, "stage": "idle", "stage_note": "",
            "progress": 0.0, "log": [], "misfit": [], "rms": [], "images": {},
            "result": None, "error": None, "config": None, "t_start": None,
            "vs3d": None, "layers": None}


STATE = _fresh()


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


def _vs3d_payload(grid, depth, vs, true_maps=None, target=26):
    """Compact stacked-interface + layer-Vs payload for the in-browser 3-D
    Vs viewer.  `depth` = cumulative interface depths (nL, nx, ny), positive
    down; client renders z = -depth."""
    step = max(1, grid.nx // target, grid.ny // target)
    depth = depth[:, ::step, ::step]
    ds = lambda a: [[round(float(v), 1) for v in row] for row in a]
    out = {"x": [round(float(v), 1) for v in grid.x[::step]],
           "y": [round(float(v), 1) for v in grid.y[::step]],
           "vs": [round(float(v), 0) for v in vs],
           "layers": [ds(depth[L]) for L in range(depth.shape[0])],
           "max_depth": float(max(depth[-1].max(), 1.0)),
           "vs_min": float(min(vs)), "vs_max": float(max(vs))}
    if true_maps is not None:
        td = np.cumsum(true_maps, axis=0)[:, ::step, ::step]
        out["true_layers"] = [ds(td[L]) for L in range(td.shape[0])]
    return out


def _vs3d(model, params, true_maps=None, target=26):
    return _vs3d_payload(model.grid, model.interface_depths(params),
                         model.thickness_grid(params)[1], true_maps, target)


def _build_free_mask(model, cfg, true_maps, x0):
    """Assemble the free/fixed parameter mask and pin fixed values into x0."""
    free = np.ones(model.n_params, bool)
    nL = model.n_layers
    # bedrock Vs is always fixed (deep reference)
    free[model.n_thick + nL] = False
    # per-layer Vs fixing
    for L in range(nL):
        if cfg["fix_vs"][L]:
            free[model.n_thick + L] = False
    # known bedrock: pin the deepest layer's thickness nodes to the truth
    if cfg["bedrock_known"]:
        spl = [RectBivariateSpline(model.grid.x, model.grid.y, true_maps[L])
               for L in range(nL)]
        Lb = nL - 1
        for a, ix in enumerate(model.node_x):
            for b, iy in enumerate(model.node_y):
                j = Lb * model.ncx * model.ncy + a * model.ncy + b
                x0[j] = float(np.clip(spl[Lb](ix, iy)[0, 0], 0, None))
                free[j] = False
        log("bedrock interface pinned to known values (borehole constraint)")
    return free, x0


# ===================================================================== job
def pipeline(cfg):
    try:
        t_all = time.time()
        set_state(t_start=t_all)
        grid = GridSpec(nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"], dx=cfg["dx"])
        margin = cfg["margin"]
        nL = cfg["n_layers"]
        freqs = np.linspace(cfg["fmin"], cfg["fmax"], cfg["nfreq"])

        # ------------------------------------------------------- 1. data
        set_state(stage="data", stage_note="building basin & stations",
                  progress=0.1)
        vs_true = np.array(cfg["vs_true"], float)
        true_maps, vs_true = H.make_true_basin(
            grid, n_layers=nL, seed=cfg["seed"], vs=list(vs_true),
            margin_cells=margin, max_total_depth=cfg["max_total_depth"])
        xy = H.station_lattice(grid, n_side=cfg["n_side"], margin_cells=margin,
                               jitter=0.25, seed=cfg["seed"] + 3)
        th_true = H.sample_true_columns(grid, true_maps, xy)
        th_true = np.hstack([th_true, np.full((len(xy), 1), 1.0e4)])
        true_depth = np.cumsum(true_maps, axis=0)[-1]
        log(f"true basin: {nL} layers, bedrock max {true_depth.max():.0f} m, "
            f"Vs {[int(v) for v in vs_true]}; {len(xy)} stations")
        _check_stop()

        # synthesise one microtremor record (for display) + observed HVSR
        ts, dt = H.synth_microtremor(vs_true, th_true[len(xy) // 2], freqs,
                                     seed=cfg["seed"])
        V.render_microtremor(ts, dt, os.path.join(RUN_DIR, "microtremor.png"),
                             station=len(xy) // 2)
        bump("microtremor")
        set_state(stage_note="extracting H/V curves", progress=0.5)
        hv_obs = H.observed_hvsr(vs_true, th_true, freqs,
                                 noise_pct=cfg["noise_pct"], seed=cfg["seed"] + 1)
        obs_f0 = H.soft_peak(freqs, hv_obs)
        V.render_stations(grid, true_depth, xy,
                          os.path.join(RUN_DIR, "stations.png"), obs_f0=obs_f0)
        V.render_hvsr_fits(freqs, hv_obs, None, xy,
                           os.path.join(RUN_DIR, "hvsr_fit.png"))
        for n in ("stations", "hvsr_fit"):
            bump(n)
        set_state(vs3d=_vs3d_payload(grid, np.cumsum(true_maps, axis=0),
                                     vs_true, true_maps))
        log(f"observed HVSR: f0 {obs_f0.min():.2f}–{obs_f0.max():.2f} Hz")
        _check_stop()

        # ------------------------------------------------------- 2. invert
        set_state(stage="invert", stage_note="starting HVSR inversion",
                  progress=0.0)
        model = H.MultiLayerBasin(grid, n_layers=nL, ncx=cfg["ncx"],
                                  ncy=cfg["ncx"], margin_cells=margin)
        spec = H.LayerSpec(vs=list(vs_true))
        x0 = H.initial_guess(model, spec, freqs, hv_obs)
        free, x0 = _build_free_mask(model, cfg, true_maps, x0)
        set_state(layers=[{"vs": int(v), "fixed": bool(cfg["fix_vs"][L]
                          if L < nL else True)} for L, v in enumerate(vs_true)])
        set_state(vs3d=_vs3d(model, x0, true_maps))
        rms0 = float(np.sqrt(np.mean(
            (model.interface_depths(x0)[-1] - true_depth) ** 2)))
        n_free = int(np.count_nonzero(free))
        log(f"unknowns: {n_free} free of {model.n_params} "
            f"({nL}×{model.ncx}×{model.ncy} thickness nodes + {nL+1} Vs); "
            f"initial bedrock RMS {rms0:.0f} m")

        inv = H.HVSRInversion(model, spec, xy, freqs, hv_obs, free_mask=free,
                              smooth_weight=cfg["smooth_weight"])
        maxiter = cfg["maxiter"]

        def on_eval(k, f, x):
            depth = model.interface_depths(x)[-1]
            rms = float(np.sqrt(np.mean((depth - true_depth) ** 2)))
            with _LOCK:
                STATE["misfit"].append(float(f))
                STATE["rms"].append(rms)
                mis = list(STATE["misfit"])
            V.render_live_vs(model, x, true_depth,
                             os.path.join(RUN_DIR, "live_vs.png"))
            V.plot_convergence(mis, os.path.join(RUN_DIR, "convergence.png"))
            set_state(vs3d=_vs3d(model, x, true_maps),
                      progress=min(0.98, k / (maxiter * 1.3)),
                      stage_note=f"eval {k}: misfit {f:.3e}, bedrock RMS {rms:.0f} m")
            bump("live_vs"); bump("convergence")
            log(f"  eval {k}: misfit {f:.4e}, bedrock RMS {rms:.1f} m")
            _check_stop()

        inv.on_eval = on_eval
        max_thick = 0.7 * grid.nz * grid.dx
        x_final, ilog = inv.run(x0, max_thick=max_thick, maxiter=maxiter,
                                verbose=False)
        _check_stop()

        # ------------------------------------------------------- 3. report
        set_state(stage="report", stage_note="final figures", progress=0.0)
        inv_depth = model.interface_depths(x_final)[-1]
        rms = float(np.sqrt(np.mean((inv_depth - true_depth) ** 2)))
        corr = float(np.corrcoef(inv_depth.ravel(), true_depth.ravel())[0, 1])
        hv_pred = inv.predict(x_final)
        _, vs_inv = model.thickness_grid(x_final)

        # reference "true" params (true maps sampled at nodes) for a fair
        # true-vs-inverted Vs section comparison
        tn_true = np.stack([
            RectBivariateSpline(grid.x, grid.y, true_maps[L])(
                model.node_x, model.node_y) for L in range(nL)])
        x_true = model.pack(tn_true, vs_true)

        V.render_vs_sections(model, x_final,
                             os.path.join(RUN_DIR, "vs_sections.png"),
                             params_true=x_true)
        V.render_depth_compare(grid, true_depth, inv_depth, xy,
                               os.path.join(RUN_DIR, "depth_compare.png"))
        V.render_hvsr_fits(freqs, hv_obs, hv_pred, xy,
                           os.path.join(RUN_DIR, "hvsr_fit.png"))
        set_state(vs3d=_vs3d(model, x_final, true_maps))
        for n in ("vs_sections", "depth_compare", "hvsr_fit"):
            bump(n)

        with _LOCK:
            mis0 = STATE["misfit"][0] if STATE["misfit"] else None
        result = dict(
            misfit0=mis0, misfit=float(ilog.misfit[-1]) if ilog.misfit else None,
            rms0=rms0, rms=rms, corr=corr, n_evals=len(ilog.misfit),
            elapsed=time.time() - t_all, n_layers=nL,
            vs_true=[int(v) for v in vs_true],
            vs_inv=[int(v) for v in vs_inv],
            max_depth_true=float(true_depth.max()),
            max_depth_inv=float(inv_depth.max()))
        set_state(stage="done", stage_note="", progress=1.0, result=result)
        log(f"DONE in {result['elapsed']:.0f}s: bedrock RMS {rms0:.0f}→{rms:.0f} m, "
            f"depth correlation {corr:.2f}")
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
def _parse_cfg(body):
    preset = PRESETS.get(body.get("preset", "fast"), PRESETS["fast"])
    cfg = {**preset}
    cfg["seed"] = int(body.get("seed", 1))
    cfg["noise_pct"] = float(body.get("noise_pct", 5.0))
    cfg["smooth_weight"] = float(body.get("smooth_weight", 4e-3))
    nL = int(np.clip(int(body.get("n_layers", 3)), 1, 4))
    cfg["n_layers"] = nL
    vs = body.get("vs_true") or list(np.linspace(300, 620, nL))
    vs = [float(v) for v in vs][:nL] + [1800.0]
    cfg["vs_true"] = vs
    fix = body.get("fix_vs") or []
    cfg["fix_vs"] = [bool(fix[L]) if L < len(fix) else False for L in range(nL)]
    cfg["bedrock_known"] = bool(body.get("bedrock_known", False))
    if "maxiter" in body:
        try:
            cfg["maxiter"] = int(body["maxiter"])
        except (TypeError, ValueError):
            pass
    return cfg


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
            cfg = _parse_cfg(body)
            with _LOCK:
                STATE.clear()
                STATE.update(_fresh())
            set_state(running=True, config=cfg)
            log(f"starting microtremor pipeline: {cfg}")
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
    ap.add_argument("--port", type=int, default=8643)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Microtremor Studio at http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
