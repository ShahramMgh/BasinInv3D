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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from basininv import GridSpec
from basininv import hvsr as H
from basininv import hvsr_viz as V
from basininv import fieldio as FIO
from basininv.hvproc import ProcConfig, process_record, interp_hv, _refine_peak
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
            "vs3d": None, "layers": None, "mode": "demo", "stations_table": None}


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
            max_depth_inv=float(inv_depth.max()),
            station_xy=[[round(float(a), 1) for a in p] for p in xy])

        # ------------------------------------- optional uncertainty ensemble
        n_ens = cfg.get("ensemble", 1)
        if n_ens > 1:
            set_state(stage_note=f"uncertainty ensemble (0/{n_ens - 1})",
                      progress=0.0)
            log(f"running {n_ens - 1} extra ensemble members for uncertainty")
            # Members differ not just by data noise but by the analyst's
            # subjective choices — starting model and smoothing strength — since
            # those, not measurement noise, dominate HVSR non-uniqueness.  A
            # noise-only ensemble is badly over-confident.
            erng = np.random.default_rng(cfg["seed"] + 777)
            jobs = []
            for mem in range(1, n_ens):
                hv_m = H.observed_hvsr(vs_true, th_true, freqs,
                                       noise_pct=cfg["noise_pct"],
                                       seed=cfg["seed"] + 100 + mem)
                sw_m = cfg["smooth_weight"] * float(erng.uniform(0.4, 2.5))
                # vary node resolution too: representation/resolution is the
                # dominant uncertainty, and a fixed grid hides it
                ncx_m = int(np.clip(cfg["ncx"] + erng.integers(-1, 2), 2, 5))
                model_m = H.MultiLayerBasin(grid, n_layers=nL, ncx=ncx_m,
                                            ncy=ncx_m, margin_cells=margin)
                x0m = H.initial_guess(model_m, spec, freqs, hv_m,
                                      frac=float(erng.uniform(0.7, 1.3)))
                free_m, x0m = _build_free_mask(model_m, cfg, true_maps, x0m)
                jobs.append((model_m, spec, xy, freqs, hv_m, free_m, x0m,
                             max_thick, cfg["maxiter"], sw_m))
            depths = [inv_depth]
            with ProcessPoolExecutor(max_workers=min(3, len(jobs))) as ex:
                for k, dm in enumerate(ex.map(H.invert_member, jobs)):
                    depths.append(dm)
                    set_state(progress=(k + 1) / len(jobs),
                              stage_note=f"uncertainty ensemble ({k + 1}/{n_ens - 1})")
                    _check_stop()
            depths = np.stack(depths)
            mean_d, std_d = depths.mean(0), depths.std(0)
            V.render_uncertainty(grid, mean_d, std_d, true_depth, xy,
                                 os.path.join(RUN_DIR, "uncertainty.png"))
            bump("uncertainty")
            within2 = float(np.mean(np.abs(mean_d - true_depth) <= 2 * std_d + 1e-9))
            result["ensemble"] = n_ens
            result["depth_std_median"] = float(np.median(std_d))
            result["depth_std_max"] = float(std_d.max())
            result["coverage_2sigma"] = within2
            log(f"ensemble: median depth σ {np.median(std_d):.0f} m, "
                f"truth within ±2σ over {100*within2:.0f}% of area")

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


# ============================================================ field pipeline
def _proc_config(cfg):
    return ProcConfig(win_len=cfg["win_len"], overlap=cfg["overlap"],
                      taper=cfg["taper"], fmin=cfg["fmin"], fmax=cfg["fmax"],
                      nfreq=cfg["nfreq"], ko_b=cfg["ko_b"],
                      horizontal=cfg["horizontal"], reject=cfg["reject"],
                      sta=cfg["sta"], lta=cfg["lta"],
                      min_ratio=cfg["min_ratio"], max_ratio=cfg["max_ratio"])


def _process_campaign(cfg, freqs):
    """Read the campaign folder and reduce every station to an H/V curve on
    the inversion frequency grid.  Returns (xy_field, hv_obs, sigma, table,
    qc_results, example_record, scan)."""
    scan = FIO.scan_campaign(cfg["data_dir"])
    pcfg = _proc_config(cfg)
    xy, hv_obs, sig_obs, table, qc, example = [], [], [], [], [], None
    usable = [e for e in scan["stations"] if e.kind in ("record", "hv")]
    skipped = [e for e in scan["stations"] if e.kind not in ("record", "hv")]
    for e in skipped:
        log(f"  {e.sid}: skipped ({e.kind})")
    for k, e in enumerate(usable):
        _check_stop()
        payload = FIO.load_station(e)
        if payload[0] == "record":
            _, data, dt = payload
            res = process_record(data, dt, pcfg)
            hv, sg = interp_hv(freqs, res.freqs, res.hv, res.sigma)
            row = dict(sid=e.sid, x=e.x, y=e.y, kind="record",
                       f0=round(res.f0, 3), f0_sigma=round(res.f0_sigma, 3),
                       a0=round(res.a0, 2), n_win=res.n_win, n_rej=res.n_rej,
                       duration=round(res.duration, 1),
                       reliable=res.reliable, clear_peak=res.clear_peak,
                       sesame=res.sesame)
            qc.append(dict(freqs=res.freqs, hv=res.hv, sigma=res.sigma,
                           f0=res.f0, n_win=res.n_win, n_rej=res.n_rej,
                           reliable=res.reliable, clear_peak=res.clear_peak,
                           kind="record"))
            if example is None:
                example = (data, dt, e.sid)
        else:
            _, fin, hvin, sgin = payload
            sgin = sgin if sgin is not None else np.full_like(fin, 0.05)
            hv, sg = interp_hv(freqs, fin, hvin, sgin)
            i0 = int(np.argmax(hvin))
            f0, a0 = _refine_peak(np.asarray(fin, float),
                                  np.asarray(hvin, float), i0)
            row = dict(sid=e.sid, x=e.x, y=e.y, kind="hv", f0=round(f0, 3),
                       f0_sigma=None, a0=round(a0, 2), n_win=None, n_rej=None,
                       duration=None, reliable=True, clear_peak=None,
                       sesame=None)
            qc.append(dict(freqs=np.asarray(fin, float),
                           hv=np.asarray(hvin, float),
                           sigma=np.asarray(sgin, float), f0=f0,
                           reliable=True, clear_peak=True, kind="hv"))
        xy.append([e.x, e.y]); hv_obs.append(hv); sig_obs.append(sg)
        table.append(row)
        log(f"  {e.sid}: {row['kind']}, f0 {row['f0']:.2f} Hz"
            + (f", {row['n_win']} win ({row['n_rej']} rej)"
               if row["kind"] == "record" else "")
            + ("" if row["reliable"] else "  ⚠ SESAME unreliable"))
        set_state(progress=0.1 + 0.7 * (k + 1) / len(usable),
                  stage_note=f"processing {e.sid} ({k + 1}/{len(usable)})")
    return (np.asarray(xy, float), np.asarray(hv_obs), np.asarray(sig_obs),
            table, qc, example, scan)


def _field_grid(xy_field, cfg):
    """Local inversion grid covering the station bbox + margin.  Returns
    (grid, origin, xy_local, margin_cells)."""
    margin = 7
    nxy = 44
    xmin, ymin = xy_field.min(axis=0)
    span = float(max(np.ptp(xy_field[:, 0]), np.ptp(xy_field[:, 1]), 50.0))
    dx = span / (nxy - 1 - 2 * margin)
    # depth extent from the deepest resolvable column: H ~ Vs / (4 f0_min)
    vs_avg = float(np.mean(cfg["vs_init"]))
    h_est = vs_avg / (4.0 * cfg["fmin"])
    max_thick = float(cfg["max_depth"] or min(1.4 * h_est, 4.0 * span))
    nz = int(np.clip(np.ceil(1.25 * max_thick / dx) + 2, 10, 72))
    grid = GridSpec(nx=nxy, ny=nxy, nz=nz, dx=dx)
    origin = np.array([xmin - margin * dx, ymin - margin * dx])
    xy_local = xy_field - origin
    return grid, origin, xy_local, margin, max_thick


def _truth_on_grid(truth, grid, origin):
    """Interpolate a demo campaign's true bedrock depth onto the field grid
    (validation scoring only)."""
    tm = truth["true_maps"]
    depth_t = np.cumsum(tm, axis=0)[-1]
    tx = truth["origin"][0] + np.arange(tm.shape[1]) * truth["dx"]
    ty = truth["origin"][1] + np.arange(tm.shape[2]) * truth["dx"]
    spl = RectBivariateSpline(tx, ty, depth_t)
    gx = np.clip(origin[0] + grid.x, tx[0], tx[-1])
    gy = np.clip(origin[1] + grid.y, ty[0], ty[-1])
    return np.clip(spl(gx, gy), 0.0, None)


def pipeline_field(cfg):
    try:
        t_all = time.time()
        set_state(t_start=t_all, mode="field")
        freqs = np.geomspace(cfg["fmin"], cfg["fmax"], cfg["nfreq"])

        # ------------------------------------------------ 1. data / process
        set_state(stage="data", stage_note="reading campaign folder",
                  progress=0.02)
        log(f"campaign: {cfg['data_dir']}")
        (xy_field, hv_obs, sig_obs, table, qc, example,
         scan) = _process_campaign(cfg, freqs)
        set_state(stations_table=table)
        if example is not None:
            V.render_microtremor(example[0], example[1],
                                 os.path.join(RUN_DIR, "microtremor.png"),
                                 station=example[2])
            bump("microtremor")
        sids = [t["sid"] for t in table]
        ok = [bool(t["reliable"]) for t in table]
        f0s = np.array([t["f0"] for t in table], float)
        V.render_field_map(xy_field, f0s, os.path.join(RUN_DIR, "stations.png"),
                           sids=sids, ok=ok)
        V.render_proc_qc(qc, sids, os.path.join(RUN_DIR, "proc_qc.png"))
        V.render_hvsr_fits(freqs, hv_obs, None, xy_field,
                           os.path.join(RUN_DIR, "hvsr_fit.png"))
        for n in ("stations", "proc_qc", "hvsr_fit"):
            bump(n)
        n_bad = len(ok) - sum(ok)
        log(f"{len(table)} stations processed; f0 {f0s.min():.2f}–"
            f"{f0s.max():.2f} Hz" + (f"; {n_bad} fail SESAME" if n_bad else ""))
        if cfg["only_reliable"] and 0 < sum(ok) < len(ok):
            keep = np.array(ok, bool)
            xy_field, hv_obs, sig_obs = xy_field[keep], hv_obs[keep], sig_obs[keep]
            log(f"excluding {n_bad} unreliable station(s) from the inversion")
        _check_stop()

        # ------------------------------------------------------- 2. invert
        set_state(stage="invert", stage_note="building inversion grid",
                  progress=0.0)
        grid, origin, xy, margin, max_thick = _field_grid(xy_field, cfg)
        log(f"grid: {grid.nx}×{grid.ny}×{grid.nz} @ dx={grid.dx:.1f} m, "
            f"origin ({origin[0]:.0f}, {origin[1]:.0f}); "
            f"max depth {max_thick:.0f} m")
        nL = cfg["n_layers"]
        model = H.MultiLayerBasin(grid, n_layers=nL, ncx=cfg["ncx"],
                                  ncy=cfg["ncx"], margin_cells=margin)
        spec = H.LayerSpec(vs=list(cfg["vs_init"]) + [cfg["bedrock_vs"]],
                           vpvs=cfg["vpvs"], Qs=cfg["Qs"], Qp=cfg["Qp"])
        x0 = H.initial_guess(model, spec, freqs, hv_obs)
        free = np.ones(model.n_params, bool)
        free[model.n_thick + nL] = False           # bedrock Vs fixed
        for L in range(nL):
            if cfg["fix_vs"][L]:
                free[model.n_thick + L] = False
        set_state(layers=[{"vs": int(v), "fixed": bool(cfg["fix_vs"][L]
                          if L < nL else True)}
                          for L, v in enumerate(spec.vs)])
        set_state(vs3d=_vs3d(model, x0))
        n_free = int(np.count_nonzero(free))
        log(f"unknowns: {n_free} free of {model.n_params} "
            f"({nL}×{model.ncx}×{model.ncy} thickness nodes + {nL + 1} Vs)")

        inv = H.HVSRInversion(model, spec, xy, freqs, hv_obs, free_mask=free,
                              smooth_weight=cfg["smooth_weight"],
                              peak_weight=cfg["peak_weight"],
                              data_weight=cfg["data_weight"])
        maxiter = cfg["maxiter"]
        obs_f0 = inv.fpk_obs

        def on_eval(k, f, x):
            fpk = H.soft_peak(freqs, inv.predict(x))
            f0res = 100.0 * float(np.median(np.abs(np.log(fpk / obs_f0))))
            with _LOCK:
                STATE["misfit"].append(float(f))
                STATE["rms"].append(f0res)
                mis = list(STATE["misfit"])
            V.render_live_field(model, x, os.path.join(RUN_DIR, "live_vs.png"),
                                origin=origin, xy=xy_field)
            V.plot_convergence(mis, os.path.join(RUN_DIR, "convergence.png"))
            set_state(vs3d=_vs3d(model, x),
                      progress=min(0.98, k / (maxiter * 1.3)),
                      stage_note=f"eval {k}: misfit {f:.3e}, "
                                 f"median f₀ residual {f0res:.1f}%")
            bump("live_vs"); bump("convergence")
            log(f"  eval {k}: misfit {f:.4e}, f0 residual {f0res:.1f}%")
            _check_stop()

        inv.on_eval = on_eval
        x_final, ilog = inv.run(x0, max_thick=max_thick, maxiter=maxiter,
                                verbose=False)
        _check_stop()

        # ------------------------------------------------------- 3. report
        set_state(stage="report", stage_note="final figures", progress=0.0)
        truth = FIO.load_truth(cfg["data_dir"]) if scan["has_truth"] else None
        true_depth = _truth_on_grid(truth, grid, origin) if truth else None
        inv_depth = model.interface_depths(x_final)[-1]
        hv_pred = inv.predict(x_final)
        _, vs_inv = model.thickness_grid(x_final)
        fpk = H.soft_peak(freqs, hv_pred)
        f0res = 100.0 * float(np.median(np.abs(np.log(fpk / obs_f0))))

        V.render_field_map(xy_field, f0s, os.path.join(RUN_DIR, "stations.png"),
                           sids=sids, ok=ok, grid=grid, depth=inv_depth,
                           origin=origin)
        V.render_field_depth(model, x_final, xy_field,
                             os.path.join(RUN_DIR, "depth_compare.png"),
                             origin=origin, true_depth=true_depth)
        V.render_vs_sections(model, x_final,
                             os.path.join(RUN_DIR, "vs_sections.png"))
        V.render_hvsr_fits(freqs, hv_obs, hv_pred, xy_field,
                           os.path.join(RUN_DIR, "hvsr_fit.png"))
        set_state(vs3d=_vs3d(model, x_final))
        for n in ("stations", "depth_compare", "vs_sections", "hvsr_fit"):
            bump(n)

        with _LOCK:
            mis0 = STATE["misfit"][0] if STATE["misfit"] else None
        result = dict(
            misfit0=mis0, misfit=float(ilog.misfit[-1]) if ilog.misfit else None,
            f0_residual_pct=f0res, n_evals=len(ilog.misfit),
            elapsed=time.time() - t_all, n_layers=nL, mode="field",
            n_stations=len(table), n_used=len(xy_field),
            vs_init=[int(v) for v in spec.vs],
            vs_inv=[int(v) for v in vs_inv],
            max_depth_inv=float(inv_depth.max()),
            origin=[float(origin[0]), float(origin[1])],
            grid=dict(nx=grid.nx, ny=grid.ny, nz=grid.nz, dx=float(grid.dx)),
            station_xy=[[round(float(a), 1) for a in p] for p in xy_field])
        if true_depth is not None:
            rms = float(np.sqrt(np.mean((inv_depth - true_depth) ** 2)))
            corr = float(np.corrcoef(inv_depth.ravel(),
                                     true_depth.ravel())[0, 1])
            result["rms"] = rms
            result["corr"] = corr
            log(f"validation vs hidden truth: bedrock RMS {rms:.0f} m, "
                f"correlation {corr:.2f}")

        # ------------------------------------- optional uncertainty ensemble
        n_ens = cfg.get("ensemble", 1)
        if n_ens > 1:
            set_state(stage_note=f"uncertainty ensemble (0/{n_ens - 1})",
                      progress=0.0)
            log(f"running {n_ens - 1} extra ensemble members for uncertainty")
            # field version: perturb each curve within its *measured* window
            # scatter, and vary starting model, smoothing and node resolution
            erng = np.random.default_rng(cfg["seed"] + 777)
            jobs = []
            for mem in range(1, n_ens):
                hv_m = hv_obs * np.exp(erng.normal(0.0, 1.0, hv_obs.shape)
                                       * sig_obs)
                sw_m = cfg["smooth_weight"] * float(erng.uniform(0.4, 2.5))
                ncx_m = int(np.clip(cfg["ncx"] + erng.integers(-1, 2), 2, 5))
                model_m = H.MultiLayerBasin(grid, n_layers=nL, ncx=ncx_m,
                                            ncy=ncx_m, margin_cells=margin)
                x0m = H.initial_guess(model_m, spec, freqs, hv_m,
                                      frac=float(erng.uniform(0.7, 1.3)))
                free_m = np.ones(model_m.n_params, bool)
                free_m[model_m.n_thick + nL] = False
                for L in range(nL):
                    if cfg["fix_vs"][L]:
                        free_m[model_m.n_thick + L] = False
                jobs.append((model_m, spec, xy, freqs, hv_m, free_m, x0m,
                             max_thick, maxiter, sw_m))
            depths = [inv_depth]
            with ProcessPoolExecutor(max_workers=min(3, len(jobs))) as ex:
                for k, dm in enumerate(ex.map(H.invert_member, jobs)):
                    depths.append(dm)
                    set_state(progress=(k + 1) / len(jobs),
                              stage_note=f"uncertainty ensemble ({k + 1}/{n_ens - 1})")
                    _check_stop()
            depths = np.stack(depths)
            mean_d, std_d = depths.mean(0), depths.std(0)
            V.render_uncertainty(grid, mean_d, std_d, true_depth, xy_field,
                                 os.path.join(RUN_DIR, "uncertainty.png"),
                                 origin=origin)
            bump("uncertainty")
            result["ensemble"] = n_ens
            result["depth_std_median"] = float(np.median(std_d))
            result["depth_std_max"] = float(std_d.max())
            if true_depth is not None:
                within2 = float(np.mean(np.abs(mean_d - true_depth)
                                        <= 2 * std_d + 1e-9))
                result["coverage_2sigma"] = within2
            log(f"ensemble: median depth σ {np.median(std_d):.0f} m")

        set_state(stage="done", stage_note="", progress=1.0, result=result)
        log(f"DONE in {result['elapsed']:.0f}s: max bedrock depth "
            f"{inv_depth.max():.0f} m, final f0 residual {f0res:.1f}%")
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
def _parse_field_cfg(body):
    def num(k, d, lo=None, hi=None):
        try:
            v = float(body.get(k, d))
        except (TypeError, ValueError):
            v = d
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    nL = int(np.clip(int(body.get("n_layers", 3) or 3), 1, 4))
    vs = body.get("vs_init") or list(np.linspace(300, 620, nL))
    vs = [float(v) for v in vs][:nL]
    while len(vs) < nL:
        vs.append(vs[-1] + 120.0)
    fix = body.get("fix_vs") or []
    cfg = dict(
        mode="field",
        data_dir=str(body.get("data_dir", "")).strip(),
        # --- processing
        win_len=num("win_len", 40.0, 5.0, 600.0),
        overlap=num("overlap", 0.5, 0.0, 0.9),
        taper=num("taper", 0.10, 0.01, 0.45),
        ko_b=num("ko_b", 40.0, 10.0, 100.0),
        horizontal=("quadratic" if body.get("horizontal") == "quadratic"
                    else "geometric"),
        reject=bool(body.get("reject", True)),
        sta=num("sta", 1.0, 0.1, 30.0), lta=num("lta", 30.0, 1.0, 600.0),
        min_ratio=num("min_ratio", 0.3, 0.0, 1.0),
        max_ratio=num("max_ratio", 2.5, 1.0, 20.0),
        only_reliable=bool(body.get("only_reliable", False)),
        # --- band
        fmin=num("fmin", 0.3, 0.05, 5.0), fmax=num("fmax", 12.0, 1.0, 50.0),
        nfreq=int(num("nfreq", 120, 40, 300)),
        # --- inversion
        n_layers=nL, vs_init=vs,
        fix_vs=[bool(fix[L]) if L < len(fix) else False for L in range(nL)],
        bedrock_vs=num("bedrock_vs", 1800.0, 500.0, 4500.0),
        vpvs=num("vpvs", 2.2, 1.4, 4.0),
        Qs=num("Qs", 25.0, 5.0, 200.0), Qp=num("Qp", 40.0, 5.0, 300.0),
        ncx=int(num("ncx", 3, 2, 5)),
        smooth_weight=num("smooth_weight", 4e-3, 0.0, 1.0),
        peak_weight=num("peak_weight", 4.0, 0.0, 20.0),
        data_weight=num("data_weight", 0.4, 0.0, 10.0),
        maxiter=int(num("maxiter", 70, 5, 400)),
        max_depth=num("max_depth", 0.0, 0.0, 3000.0) or None,
        ensemble=int(np.clip(int(body.get("ensemble", 1) or 1), 1, 8)),
        seed=int(num("seed", 1)),
    )
    if cfg["fmax"] <= cfg["fmin"] * 1.5:
        cfg["fmax"] = cfg["fmin"] * 1.5 + 1.0
    return cfg


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
    cfg["ensemble"] = int(np.clip(int(body.get("ensemble", 1) or 1), 1, 8))
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
            mode = body.get("mode", "demo")
            if mode == "field":
                cfg = _parse_field_cfg(body)
                if not cfg["data_dir"] or not os.path.isdir(cfg["data_dir"]):
                    self._send(400, json.dumps(
                        {"error": f"campaign folder not found: "
                                  f"{cfg['data_dir'] or '(empty)'}"}).encode())
                    return
                target = pipeline_field
            else:
                cfg = _parse_cfg(body)
                target = pipeline
            with _LOCK:
                STATE.clear()
                STATE.update(_fresh())
            set_state(running=True, config=cfg, mode=mode)
            log(f"starting {mode} pipeline: {cfg}")
            threading.Thread(target=target, args=(cfg,), daemon=True).start()
            self._send(200, b'{"ok":true}')
        elif self.path == "/api/scan":
            folder = str(body.get("data_dir", "")).strip()
            try:
                scan = FIO.scan_campaign(folder)
                out = {"folder": scan["folder"], "bbox": scan["bbox"],
                       "has_truth": scan["has_truth"],
                       "stations": [dict(sid=e.sid, x=e.x, y=e.y, kind=e.kind,
                                         **(e.meta or {}))
                                    for e in scan["stations"]]}
                self._send(200, json.dumps(out).encode())
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode())
        elif self.path == "/api/make_demo":
            folder = str(body.get("data_dir", "")).strip() or \
                os.path.join(RUN_DIR, "field_demo")
            try:
                info = FIO.make_demo_campaign(
                    folder, seed=int(body.get("seed", 2) or 2),
                    n_layers=int(np.clip(int(body.get("n_layers", 3) or 3), 1, 4)),
                    n_side=int(np.clip(int(body.get("n_side", 5) or 5), 3, 8)),
                    noise_pct=float(body.get("noise_pct", 5.0) or 5.0))
                log(f"demo campaign written to {info['folder']} "
                    f"({info['n_stations']} stations)")
                self._send(200, json.dumps(info).encode())
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode())
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
