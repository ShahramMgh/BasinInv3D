#!/usr/bin/env python3
"""BasinInv3D — Field Dashboard.

Map-oriented live dashboard for inverting real microtremor surveys into 3-D
basin structure.  The workflow is data-first:

  * load a folder of HVSR data (raw 3-C recordings or processed .hv curves)
    with a coordinate file — stations appear on an OSM map, are processed to
    H/V curves with SESAME QC, and can be configured point by point;
  * load any number of auxiliary geophysical point datasets (boreholes,
    resistivity soundings, GPR interpretations, geology…) whose interpreted
    depths become **constraints** in the inversion;
  * enrich points manually (fixed depths, exclusions, notes, new points);
  * run the constrained HVSR inversion live and see the bedrock-depth /
    uncertainty rasters georeferenced on the map.

Everything is persisted in a project workspace so a field campaign can be
built up across sessions.  Synthetic demo data exists for validation only.

Run:  /usr/bin/python3 webapp_field/app.py [--port 8644]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RectBivariateSpline

from basininv import GridSpec
from basininv import hvsr as H
from basininv import fieldio as FIO
from basininv.hvproc import ProcConfig, process_record, interp_hv, _refine_peak

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "run", "project")
DS_DIR = os.path.join(WORK, "datasets")
PROC_DIR = os.path.join(WORK, "proc")
OVL_DIR = os.path.join(WORK, "overlays")
for d in (DS_DIR, PROC_DIR, OVL_DIR):
    os.makedirs(d, exist_ok=True)

DEFAULT_ANCHOR = (45.1885, 5.7245)    # Grenoble basin — default map location,
                                      # also used for datasets that only have x/y

# sequential blue ramp (dataviz reference palette), light->dark
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#104281", "#0d366b"]
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth_blue", BLUE_RAMP)
SIGMA_RAMP = ["#d9f2e7", "#8fd9bc", "#45bd92", "#1baf7a", "#128a60",
              "#0c6b4a", "#074d35"]                       # aqua ramp
SIGMA_CMAP = LinearSegmentedColormap.from_list("sigma_aqua", SIGMA_RAMP)
F0_RAMP = ["#e6e2f7", "#c3bbee", "#a096e0", "#7f71cd", "#5f4fb8",
           "#4a3aa7", "#372b85", "#251d61"]               # violet ramp
F0_CMAP = LinearSegmentedColormap.from_list("f0_violet", F0_RAMP)

_LOCK = threading.Lock()


# ================================================================ project
def _empty_project():
    return {"datasets": {}, "anchor": None}


def _load_project():
    p = os.path.join(WORK, "project.json")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return _empty_project()


PROJ = _load_project()


def _save_project():
    with _LOCK:
        blob = json.dumps(PROJ, indent=1)
    tmp = os.path.join(WORK, "project.json.tmp")
    with open(tmp, "w") as f:
        f.write(blob)
    os.replace(tmp, os.path.join(WORK, "project.json"))


def _fresh_job():
    return {"running": False, "stop": False, "kind": None, "stage": "idle",
            "note": "", "progress": 0.0, "log": [], "misfit": [], "f0res": [],
            "overlays": {}, "result": None, "error": None, "t_start": None,
            "vs3d": None}


JOB = _fresh_job()


def log(msg):
    with _LOCK:
        JOB["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        del JOB["log"][:-300]
    print(msg, flush=True)


def set_job(**kw):
    with _LOCK:
        JOB.update(kw)


class StopRequested(Exception):
    pass


def _check_stop():
    with _LOCK:
        if JOB["stop"]:
            raise StopRequested


def _anchor():
    """Project-wide geographic anchor of the local metric frame."""
    with _LOCK:
        a = PROJ.get("anchor")
    return tuple(a) if a else None


def _set_anchor_from_stations():
    lls = []
    with _LOCK:
        for ds in PROJ["datasets"].values():
            for s in ds.get("stations", []):
                if s.get("lat") is not None:
                    lls.append((s["lat"], s["lon"]))
    if lls:
        a = (float(np.mean([p[0] for p in lls])),
             float(np.mean([p[1] for p in lls])))
        with _LOCK:
            PROJ["anchor"] = list(a)
        return a
    return None


def _localize(obj):
    """Fill x, y (project-local metres) for a point dict with lat/lon."""
    a = _anchor() or DEFAULT_ANCHOR
    if obj.get("lat") is not None:
        x, y = FIO.latlon_to_local(obj["lat"], obj["lon"], *a)
        obj["x"], obj["y"] = float(x), float(y)
    return obj


def _relocalize_all():
    """Recompute local coords of every point after the anchor moved; give
    synthetic lat/lon to x/y-only points so they can sit on the map."""
    a = _anchor() or DEFAULT_ANCHOR
    with _LOCK:
        items = [(ds, p) for ds in PROJ["datasets"].values()
                 for p in ds.get("stations", []) + ds.get("points", [])]
    for _, p in items:
        if p.get("lat") is not None:
            x, y = FIO.latlon_to_local(p["lat"], p["lon"], *a)
            p["x"], p["y"] = float(x), float(y)
        elif p.get("x") is not None:
            la, lo = FIO.local_to_latlon(p["x"], p["y"], *a)
            p["lat"], p["lon"] = float(la), float(lo)
            p["approx_pos"] = True


# ======================================================== dataset ingestion
def _safe_name(name):
    return re.sub(r"[^\w.-]", "_", name)[:60] or "dataset"


DEPTH_ATTR_RE = re.compile(r"depth|bedrock|interface|boundary", re.I)


def _default_point_cfg(attrs):
    """Auto-wire an imported point as an inversion constraint when it carries
    a depth-like numeric attribute."""
    attr = next((k for k, v in attrs.items()
                 if isinstance(v, (int, float)) and DEPTH_ATTR_RE.search(k)),
                None)
    return {"use": attr is not None, "attr": attr, "layer": -1,
            "weight": 1.0, "note": ""}


def commit_hvsr_dataset(name, proc_kw=None):
    """Parse an uploaded HVSR campaign folder and process every record to an
    H/V curve (background-thread worker)."""
    folder = os.path.join(DS_DIR, name)
    try:
        scan = FIO.scan_campaign(folder)
        cfg = ProcConfig(**(proc_kw or {}))
        stations, curves = [], {}
        usable = [e for e in scan["stations"] if e.kind in ("record", "hv")]
        set_job(stage="processing", progress=0.02,
                note=f"{name}: {len(usable)} stations")
        for k, e in enumerate(usable):
            _check_stop()
            payload = FIO.load_station(e)
            if payload[0] == "record":
                _, data, dt = payload
                r = process_record(data, dt, cfg)
                row = dict(sid=e.sid, lat=e.lat, lon=e.lon, x=e.x, y=e.y,
                           kind="record", f0=round(r.f0, 3),
                           f0_sigma=round(r.f0_sigma, 3), a0=round(r.a0, 2),
                           n_win=r.n_win, n_rej=r.n_rej,
                           duration=round(r.duration, 1),
                           reliable=r.reliable, clear_peak=r.clear_peak,
                           sesame=r.sesame)
                curves[e.sid] = (r.freqs, r.hv, r.sigma)
            else:
                _, fin, hvin, sgin = payload
                fin = np.asarray(fin, float)
                hvin = np.asarray(hvin, float)
                sgin = (np.asarray(sgin, float) if sgin is not None
                        else np.full_like(fin, 0.05))
                f0, a0 = _refine_peak(fin, hvin, int(np.argmax(hvin)))
                row = dict(sid=e.sid, lat=e.lat, lon=e.lon, x=e.x, y=e.y,
                           kind="hv", f0=round(f0, 3), f0_sigma=None,
                           a0=round(a0, 2), n_win=None, n_rej=None,
                           duration=None, reliable=True, clear_peak=None,
                           sesame=None)
                curves[e.sid] = (fin, hvin, sgin)
            row["cfg"] = {"exclude": False, "fix_depth": None, "layer": -1,
                          "weight": 1.0, "note": ""}
            stations.append(row)
            set_job(progress=0.02 + 0.96 * (k + 1) / len(usable),
                    note=f"{name}: {e.sid} ({k + 1}/{len(usable)})")
            log(f"  {e.sid}: {row['kind']}, f0 {row['f0']:.2f} Hz"
                + ("" if row["reliable"] else "  ⚠ SESAME unreliable"))
        np.savez_compressed(
            os.path.join(PROC_DIR, name + ".npz"),
            **{f"{sid}::{part}": arr for sid, (f, h, s) in curves.items()
               for part, arr in (("f", f), ("hv", h), ("sig", s))})
        with _LOCK:
            PROJ["datasets"][name].update(
                stations=stations, status="ready", error=None,
                has_truth=scan["has_truth"], n=len(stations))
        _set_anchor_from_stations()
        _relocalize_all()
        _save_project()
        log(f"dataset '{name}' ready: {len(stations)} stations")
        set_job(stage="idle", note="", progress=1.0)
    except StopRequested:
        with _LOCK:
            PROJ["datasets"].pop(name, None)
        set_job(stage="idle", note="")
        log(f"processing of '{name}' cancelled")
    except Exception:
        err = traceback.format_exc()
        with _LOCK:
            PROJ["datasets"][name].update(status="error", error=err)
        set_job(stage="idle", error=err)
        log(f"ERROR in dataset '{name}':\n{err}")
    finally:
        set_job(running=False)
        _save_project()


def commit_aux_dataset(name, aux_type):
    """Parse an uploaded auxiliary point CSV (boreholes, resistivity, GPR…)."""
    folder = os.path.join(DS_DIR, name)
    csvs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
    if not csvs:
        raise FileNotFoundError("no .csv point file in the uploaded data")
    pts = []
    for fname in csvs:
        pts.extend(FIO.read_points_csv(os.path.join(folder, fname)))
    for p in pts:
        p["cfg"] = _default_point_cfg(p["attrs"])
        _localize(p)
    with _LOCK:
        PROJ["datasets"][name].update(points=pts, status="ready",
                                      error=None, n=len(pts))
    _relocalize_all()
    _save_project()
    log(f"aux dataset '{name}' ({aux_type}): {len(pts)} points, "
        f"{sum(1 for p in pts if p['cfg']['use'])} auto-wired as constraints")


def _curves(name):
    """Load a dataset's processed curves {sid: (f, hv, sig)}."""
    out = {}
    p = os.path.join(PROC_DIR, name + ".npz")
    if not os.path.isfile(p):
        return out
    with np.load(p) as z:
        for key in z.files:
            sid, part = key.split("::")
            out.setdefault(sid, {})[part] = np.asarray(z[key], float)
    return {sid: (d["f"], d["hv"], d["sig"]) for sid, d in out.items()}


# ============================================================== inversion
def _collect_inputs(cfg):
    """Gather stations, curves and constraints from the whole project."""
    freqs = np.geomspace(cfg["fmin"], cfg["fmax"], cfg["nfreq"])
    xy, hv, sig, used = [], [], [], []
    constraints = []
    with _LOCK:
        datasets = json.loads(json.dumps(PROJ["datasets"]))
    for name, ds in datasets.items():
        if ds.get("kind") == "hvsr" and ds.get("status") == "ready":
            curves = _curves(name)
            for s in ds["stations"]:
                c = s.get("cfg", {})
                if c.get("exclude"):
                    continue
                if cfg["only_reliable"] and not s.get("reliable", True):
                    continue
                if s["sid"] not in curves:
                    continue
                f, h, sg = curves[s["sid"]]
                hi, si = interp_hv(freqs, f, h, sg)
                xy.append([s["x"], s["y"]])
                hv.append(hi)
                sig.append(si)
                used.append(f"{name}:{s['sid']}")
                if c.get("fix_depth"):
                    constraints.append(dict(
                        x=s["x"], y=s["y"], layer=c.get("layer", -1),
                        depth=float(c["fix_depth"]),
                        weight=float(c.get("weight", 1.0))))
        elif ds.get("kind") == "aux" and cfg["use_constraints"]:
            for p in ds.get("points", []):
                c = p.get("cfg", {})
                v = p["attrs"].get(c.get("attr") or "")
                if c.get("use") and isinstance(v, (int, float)) and v > 0:
                    constraints.append(dict(
                        x=p["x"], y=p["y"], layer=c.get("layer", -1),
                        depth=float(v), weight=float(c.get("weight", 1.0))))
    if len(xy) < 4:
        raise ValueError(f"only {len(xy)} usable stations — load an HVSR "
                         "dataset (and check exclusions) first")
    return (freqs, np.asarray(xy, float), np.asarray(hv), np.asarray(sig),
            constraints, used)


def _grid_for(xy, cfg):
    margin, nxy = 7, 44
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 50.0))
    dx = span / (nxy - 1 - 2 * margin)
    vs_avg = float(np.mean([l["vs"] for l in cfg["layers"]]))
    h_est = vs_avg / (4.0 * cfg["fmin"])
    max_thick = float(cfg["max_depth"] or min(1.4 * h_est, 4.0 * span))
    nz = int(np.clip(np.ceil(1.25 * max_thick / dx) + 2, 10, 72))
    grid = GridSpec(nx=nxy, ny=nxy, nz=nz, dx=dx)
    origin = np.array([xy[:, 0].min() - margin * dx,
                       xy[:, 1].min() - margin * dx])
    return grid, origin, margin, max_thick


def _overlay_png(grid, origin, field, path, cmap, vmin, vmax, alpha=0.72):
    """Write a georeferenced RGBA raster of a (nx, ny) field and return its
    Leaflet bounds [[south, west], [north, east]]."""
    img = np.clip((field.T[::-1, :] - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = cmap(img)
    rgba[..., 3] = alpha
    plt.imsave(path, rgba)
    a = _anchor() or DEFAULT_ANCHOR
    s, w = FIO.local_to_latlon(origin[0], origin[1], *a)
    n, e = FIO.local_to_latlon(origin[0] + grid.x[-1], origin[1] + grid.y[-1], *a)
    return [[float(s), float(w)], [float(n), float(e)]]


def _truth_depth(grid, origin):
    """If any dataset folder carries a demo truth.npz, interpolate its bedrock
    depth onto the inversion grid (validation scoring only)."""
    a = _anchor() or DEFAULT_ANCHOR
    with _LOCK:
        names = [n for n, d in PROJ["datasets"].items() if d.get("has_truth")]
    for name in names:
        t = FIO.load_truth(os.path.join(DS_DIR, name))
        if not t or "lat0" not in t:
            continue
        tm = t["true_maps"]
        depth = np.cumsum(tm, axis=0)[-1]
        # truth grid nodes -> project-local coords
        gx = np.arange(tm.shape[1]) * t["dx"]
        gy = np.arange(tm.shape[2]) * t["dx"]
        la, lo = FIO.local_to_latlon(gx, np.zeros_like(gx), t["lat0"], t["lon0"])
        tx, _ = FIO.latlon_to_local(la, lo, *a)
        la, lo = FIO.local_to_latlon(np.zeros_like(gy), gy, t["lat0"], t["lon0"])
        _, ty = FIO.latlon_to_local(la, lo, *a)
        spl = RectBivariateSpline(tx, ty, depth)
        ex = np.clip(origin[0] + grid.x, tx[0], tx[-1])
        ey = np.clip(origin[1] + grid.y, ty[0], ty[-1])
        return np.clip(spl(ex, ey), 0.0, None)
    return None


def _vs3d_payload(model, params, target=26, stations=None):
    grid = model.grid
    depth = model.interface_depths(params)
    _, vs = model.thickness_grid(params)
    step = max(1, grid.nx // target)
    depth = depth[:, ::step, ::step]
    ds = lambda a: [[round(float(v), 1) for v in row] for row in a]
    out = {"x": [round(float(v), 1) for v in grid.x[::step]],
           "y": [round(float(v), 1) for v in grid.y[::step]],
           "vs": [round(float(v), 0) for v in vs],
           "layers": [ds(depth[L]) for L in range(depth.shape[0])],
           "max_depth": float(max(depth[-1].max(), 1.0)),
           "vs_min": float(min(vs)), "vs_max": float(max(vs))}
    if stations is not None:
        out["stations"] = [[round(float(x), 1), round(float(y), 1)]
                           for x, y in stations]
    return out


# --------------------------------------------- result persistence & queries
RESULTS_FILE = os.path.join(WORK, "results.json")
MODEL_STATE = None      # rebuildable final model for sections / profiles


def _persist_results():
    with _LOCK:
        blob = {"result": JOB["result"], "overlays": JOB["overlays"],
                "vs3d": JOB["vs3d"], "misfit": JOB["misfit"],
                "f0res": JOB["f0res"], "model_state": MODEL_STATE}
    with open(RESULTS_FILE, "w") as f:
        json.dump(blob, f)


def _restore_results():
    """Bring the last inversion back after a server restart."""
    global MODEL_STATE
    if not os.path.isfile(RESULTS_FILE):
        return
    try:
        with open(RESULTS_FILE) as f:
            blob = json.load(f)
        MODEL_STATE = blob.get("model_state")
        with _LOCK:
            JOB.update(result=blob.get("result"),
                       overlays=blob.get("overlays") or {},
                       vs3d=blob.get("vs3d"),
                       misfit=blob.get("misfit") or [],
                       f0res=blob.get("f0res") or [],
                       stage="done" if blob.get("result") else "idle")
    except Exception:
        pass


def _rebuild_model():
    """MultiLayerBasin + final params + origin from the persisted state."""
    if not MODEL_STATE:
        return None
    ms = MODEL_STATE
    grid = GridSpec(**ms["grid"])
    model = H.MultiLayerBasin(grid, n_layers=ms["n_layers"], ncx=ms["ncx"],
                              ncy=ms["ncx"], margin_cells=ms["margin"])
    return model, np.asarray(ms["params"], float), np.asarray(ms["origin"])


def profile_at(lat, lon):
    """1-D layer column of the final model under a map point."""
    rb = _rebuild_model()
    if not rb:
        return None
    model, params, origin = rb
    a = _anchor() or DEFAULT_ANCHOR
    x, y = FIO.latlon_to_local(lat, lon, *a)
    gx = float(np.clip(x - origin[0], 0, model.grid.x[-1]))
    gy = float(np.clip(y - origin[1], 0, model.grid.y[-1]))
    inside = (abs(gx - (x - origin[0])) < 1e-6
              and abs(gy - (y - origin[1])) < 1e-6)
    th, vs = model.columns_at(params, [[gx, gy]])
    depths = np.cumsum(th[0, :-1])
    return {"interfaces": [round(float(d), 1) for d in depths],
            "vs": [int(v) for v in vs[:-1]], "bedrock_vs": int(vs[-1]),
            "bedrock_depth": round(float(depths[-1]), 1), "inside": inside}


def section_png(lat1, lon1, lat2, lon2, ns=140):
    """Vs cross-section of the final model along a line drawn on the map."""
    rb = _rebuild_model()
    if not rb:
        return None
    model, params, origin = rb
    a = _anchor() or DEFAULT_ANCHOR
    x1, y1 = FIO.latlon_to_local(lat1, lon1, *a)
    x2, y2 = FIO.latlon_to_local(lat2, lon2, *a)
    t = np.linspace(0, 1, ns)
    px = np.clip(x1 + (x2 - x1) * t - origin[0], 0, model.grid.x[-1])
    py = np.clip(y1 + (y2 - y1) * t - origin[1], 0, model.grid.y[-1])
    dist = t * float(np.hypot(x2 - x1, y2 - y1))
    th, vs = model.columns_at(params, np.column_stack([px, py]))
    depth = np.cumsum(th[:, :-1], axis=1)          # (ns, nL)
    nL = depth.shape[1]
    fig, ax = plt.subplots(figsize=(8.6, 3.4), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#f1efe9")                    # bedrock backdrop
    prev = np.zeros(ns)
    for L in range(nL):
        col = BLUE_RAMP[min(1 + L * 2, len(BLUE_RAMP) - 2)]
        ax.fill_between(dist, prev, depth[:, L], color=col,
                        label=f"L{L + 1}  Vs {int(vs[L])} m/s")
        ax.plot(dist, depth[:, L], color="#fcfcfb", lw=1.2)
        prev = depth[:, L]
    zmax = max(float(depth[-1].max()), float(depth.max())) * 1.25 + 5
    ax.text(0.99, 0.03, f"bedrock  Vs {int(vs[-1])} m/s",
            transform=ax.transAxes, ha="right", fontsize=9, color="#52514e")
    ax.set_ylim(zmax, 0)
    ax.set_xlim(0, dist[-1])
    ax.set_xlabel("distance along section (m)", fontsize=9, color="#52514e")
    ax.set_ylabel("depth (m)", fontsize=9, color="#52514e")
    ax.set_title("A — B", fontsize=10, color="#0b0b0b")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3, color="#e1e0d9", lw=0.7)
    ax.tick_params(colors="#898781", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, facecolor="#fcfcfb")
    plt.close(fig)
    return buf.getvalue()


def f0_overlay():
    """Interpolated map of the measured f0 over the station footprint (IDW) —
    field QC before any inversion (low f0 = thick sediments)."""
    xy, f0 = [], []
    with _LOCK:
        for ds in PROJ["datasets"].values():
            for s in ds.get("stations", []):
                if s.get("f0") and not (s.get("cfg") or {}).get("exclude"):
                    xy.append([s["x"], s["y"]])
                    f0.append(s["f0"])
    if len(xy) < 3:
        return None
    xy = np.asarray(xy, float)
    f0 = np.asarray(f0, float)
    margin, nxy = 5, 60
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 50.0))
    dx = span / (nxy - 1 - 2 * margin)
    origin = np.array([xy[:, 0].min() - margin * dx,
                       xy[:, 1].min() - margin * dx])
    gx = origin[0] + np.arange(nxy) * dx
    gy = origin[1] + np.arange(nxy) * dx
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    d2 = ((GX[..., None] - xy[:, 0]) ** 2
          + (GY[..., None] - xy[:, 1]) ** 2)
    w = 1.0 / (d2 + (0.35 * span / np.sqrt(len(xy))) ** 2)
    # interpolate log f0 (resonance is log-natured)
    fld = np.exp((w * np.log(f0)).sum(-1) / w.sum(-1))
    vmin, vmax = float(fld.min()), float(fld.max())
    a = _anchor() or DEFAULT_ANCHOR
    img = np.clip((fld.T[::-1, :] - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = F0_CMAP(1.0 - img)              # dark violet = low f0 = deep
    rgba[..., 3] = 0.62
    plt.imsave(os.path.join(OVL_DIR, "f0.png"), rgba)
    s, wl = FIO.local_to_latlon(gx[0], gy[0], *a)
    n, e = FIO.local_to_latlon(gx[-1], gy[-1], *a)
    return {"url": "/overlays/f0.png", "v": int(time.time()),
            "bounds": [[float(s), float(wl)], [float(n), float(e)]],
            "vmin": vmin, "vmax": vmax,
            "label": "measured f₀ (Hz) — dark = low = deep"}


# -------------------------------------------------- Vs30 & site engineering
def _model_vs(n_layers):
    """Layer Vs (nL + bedrock) of the final model, with fallbacks for
    results persisted by older versions."""
    ms = MODEL_STATE or {}
    if ms.get("vs"):
        return np.asarray(ms["vs"], float)
    with _LOCK:
        r = JOB.get("result") or {}
    vi = list(r.get("vs_inv") or [])
    if len(vi) == n_layers + 1:
        return np.asarray(vi, float)
    if len(vi) == n_layers:
        return np.asarray(vi + [1800.0], float)
    return None


def _vs30_grid():
    """Time-averaged Vs of the top 30 m over the model grid (the standard
    engineering site parameter)."""
    rb = _rebuild_model()
    if not rb:
        return None
    model, params, origin = rb
    vs = _model_vs(model.n_layers)                 # nL + bedrock
    if vs is None:
        return None
    maps, _ = model.thickness_grid(params)         # (nL, nx, ny)
    remain = np.full(maps.shape[1:], 30.0)
    tt = np.zeros(maps.shape[1:])
    for L in range(maps.shape[0]):
        h = np.minimum(maps[L], remain)
        tt += h / vs[L]
        remain -= h
    tt += remain / vs[-1]                          # rest of 30 m in bedrock
    return 30.0 / np.maximum(tt, 1e-9), model, origin


def ec8_class(vs30):
    if vs30 > 800:
        return "A"
    if vs30 >= 360:
        return "B"
    if vs30 >= 180:
        return "C"
    return "D"


def _register_vs30_overlay():
    out = _vs30_grid()
    if not out:
        return None
    v30, model, origin = out
    vmin, vmax = float(v30.min()), float(v30.max())
    bounds = _overlay_png(model.grid, origin, v30,
                          os.path.join(OVL_DIR, "vs30.png"),
                          SIGMA_CMAP, vmin, vmax, alpha=0.68)
    o = {"url": "/overlays/vs30.png", "v": int(time.time()), "bounds": bounds,
         "vmin": vmin, "vmax": vmax, "label": "Vs30 (m/s) — dark = stiff"}
    with _LOCK:
        JOB["overlays"]["vs30"] = o
    return o


def _station_rows():
    """Flat station table with model-derived values where available."""
    rb = _rebuild_model()
    rows = []
    with _LOCK:
        datasets = json.loads(json.dumps(PROJ["datasets"])).items()
    for name, ds in datasets:
        for s in ds.get("stations", []):
            r = dict(dataset=name, **{k: s.get(k) for k in
                     ("sid", "lat", "lon", "x", "y", "kind", "f0", "f0_sigma",
                      "a0", "n_win", "n_rej", "reliable", "clear_peak")})
            r["excluded"] = bool((s.get("cfg") or {}).get("exclude"))
            rows.append(r)
    if rb:
        model, params, origin = rb
        vs = _model_vs(model.n_layers)
        if vs is None:
            return rows
        for r in rows:
            gx = float(np.clip(r["x"] - origin[0], 0, model.grid.x[-1]))
            gy = float(np.clip(r["y"] - origin[1], 0, model.grid.y[-1]))
            th, _ = model.columns_at(params, [[gx, gy]])
            r["bedrock_depth_model"] = round(float(
                np.cumsum(th[0, :-1])[-1]), 1)
            r["f0_model"] = round(float(H.fundamental_frequency(vs, np.append(
                th[0, :-1], 1e4))), 3)
            remain, tt = 30.0, 0.0
            for L in range(model.n_layers):
                h = min(float(th[0, L]), remain)
                tt += h / vs[L]
                remain -= h
            tt += remain / vs[-1]
            r["vs30"] = round(30.0 / max(tt, 1e-9))
            r["site_class_ec8"] = ec8_class(r["vs30"])
    return rows


def stations_csv():
    rows = _station_rows()
    if not rows:
        return None
    cols = ["dataset", "sid", "lat", "lon", "kind", "f0", "f0_sigma", "a0",
            "n_win", "n_rej", "reliable", "clear_peak", "excluded",
            "f0_model", "bedrock_depth_model", "vs30", "site_class_ec8"]
    out = [",".join(cols)]
    for r in rows:
        out.append(",".join("" if r.get(c) is None else str(r.get(c))
                            for c in cols))
    return "\n".join(out).encode()


# ------------------------------------------------------------ chart figures
INK, INK2, MUTED, GRIDC, BASEC = ("#0b0b0b", "#52514e", "#898781",
                                  "#e1e0d9", "#c3c2b7")


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(alpha=0.35, which="both", color=GRIDC, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sname in ("top", "right"):
        ax.spines[sname].set_visible(False)
    for sname in ("left", "bottom"):
        ax.spines[sname].set_color(BASEC)
    return fig, ax


def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return buf.getvalue()


def _all_curves():
    """[(sid, f, hv, f0)] over every ready hvsr dataset."""
    out = []
    with _LOCK:
        items = [(n, json.loads(json.dumps(d))) for n, d in
                 PROJ["datasets"].items() if d.get("kind") == "hvsr"
                 and d.get("status") == "ready"]
    for name, ds in items:
        curves = _curves(name)
        for s in ds["stations"]:
            if s["sid"] in curves and not (s.get("cfg") or {}).get("exclude"):
                f, hv, _ = curves[s["sid"]]
                out.append((s["sid"], f, hv, s.get("f0") or 1.0))
    return out


def chart_png(kind):
    if kind == "curves":
        data = _all_curves()
        if not data:
            return None
        f0s = np.array([d[3] for d in data])
        lo, hi = np.log(f0s.min()), np.log(max(f0s.max(), f0s.min() * 1.01))
        fig, ax = _fig(7.6, 4.2)
        for sid, f, hv, f0 in data:
            t = (np.log(f0) - lo) / max(hi - lo, 1e-9)
            ax.semilogx(f, hv, color=F0_CMAP(1 - 0.85 * t), lw=1.1, alpha=0.75)
        fgrid = np.geomspace(min(d[1][0] for d in data),
                             max(d[1][-1] for d in data), 160)
        med = np.median(np.stack([np.interp(np.log(fgrid), np.log(d[1]), d[2])
                                  for d in data]), axis=0)
        ax.semilogx(fgrid, med, color=INK, lw=2.4, label="median of survey")
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
        ax.set_xlabel("frequency (Hz)", fontsize=9, color=INK2)
        ax.set_ylabel("H/V", fontsize=9, color=INK2)
        ax.set_title(f"All H/V curves ({len(data)} stations) — "
                     "colour: dark violet = low f₀ (deep)",
                     fontsize=10, color=INK)
        return _fig_bytes(fig)

    if kind == "f0a0":
        rows = [r for r in _station_rows() if r.get("f0")]
        if not rows:
            return None
        fig, ax = _fig(6.4, 4.2)
        for r in rows:
            ok = r.get("reliable", True)
            ax.scatter(r["f0"], r["a0"], s=52,
                       c="#2a78d6" if ok else "#fcfcfb",
                       edgecolors="#0b0b0b" if ok else "#d03b3b",
                       linewidths=1.0, zorder=3)
            ax.annotate(r["sid"], (r["f0"], r["a0"]), fontsize=6.6,
                        color=MUTED, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.axhline(2.0, color=BASEC, lw=1.1, ls="--")
        ax.text(ax.get_xlim()[1], 2.05, "A₀ = 2 (SESAME amplitude) ",
                ha="right", fontsize=7.5, color=MUTED)
        ax.set_xlabel("f₀ (Hz)", fontsize=9, color=INK2)
        ax.set_ylabel("peak amplitude A₀", fontsize=9, color=INK2)
        ax.set_title("Resonance strength — filled = SESAME reliable, "
                     "red ring = failed", fontsize=10, color=INK)
        return _fig_bytes(fig)

    if kind == "fit":
        rows = [r for r in _station_rows()
                if r.get("f0") and r.get("f0_model") and not r["excluded"]]
        if not rows:
            return None
        rows.sort(key=lambda r: r["f0"])
        res = [100 * np.log(r["f0_model"] / r["f0"]) for r in rows]
        fig, ax = _fig(7.6, 3.8)
        cols = ["#2a78d6" if abs(v) < 15 else "#eb6834" for v in res]
        ax.bar(range(len(rows)), res, color=cols, width=0.72)
        ax.axhline(0, color=BASEC, lw=1.2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["sid"] for r in rows], rotation=60,
                           fontsize=6.6, ha="right")
        ax.set_ylabel("model vs measured f₀  (log %, model−data)",
                      fontsize=8.5, color=INK2)
        ax.set_title("Per-station resonance fit of the inverted model "
                     "(orange: |residual| > 15 %)", fontsize=10, color=INK)
        return _fig_bytes(fig)

    if kind == "depthdist":
        rb = _rebuild_model()
        if not rb:
            return None
        model, params, _ = rb
        depth = model.interface_depths(params)[-1].ravel()
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4),
                                 constrained_layout=True)
        fig.patch.set_facecolor("#fcfcfb")
        for ax in axes:
            ax.set_facecolor("#fcfcfb")
            ax.grid(alpha=0.35, color=GRIDC, lw=0.7)
            ax.tick_params(colors=MUTED, labelsize=8)
            for sname in ("top", "right"):
                ax.spines[sname].set_visible(False)
            for sname in ("left", "bottom"):
                ax.spines[sname].set_color(BASEC)
        axes[0].hist(depth, bins=26, color="#6da7ec", edgecolor="#fcfcfb")
        axes[0].set_xlabel("bedrock depth (m)", fontsize=8.5, color=INK2)
        axes[0].set_ylabel("grid cells", fontsize=8.5, color=INK2)
        axes[0].set_title("Depth distribution", fontsize=9.5, color=INK)
        zs = np.sort(depth)
        axes[1].plot(zs, 100 * (1 - np.arange(len(zs)) / len(zs)),
                     color="#2a78d6", lw=2)
        axes[1].set_xlabel("depth z (m)", fontsize=8.5, color=INK2)
        axes[1].set_ylabel("% of area deeper than z", fontsize=8.5, color=INK2)
        axes[1].set_title("Hypsometry of the basin", fontsize=9.5, color=INK)
        return _fig_bytes(fig)
    return None


def report_html():
    """Standalone printable campaign report."""
    with _LOCK:
        datasets = json.loads(json.dumps(PROJ["datasets"]))
        result = JOB.get("result")
    rows = _station_rows()
    a = _anchor() or DEFAULT_ANCHOR
    ncons = sum(1 for d in datasets.values() for p in d.get("points", [])
                if (p.get("cfg") or {}).get("use"))
    def esc(x):
        return str(x).replace("<", "&lt;")
    ds_rows = "".join(
        f"<tr><td>{esc(n)}</td><td>{esc(d.get('type'))}</td>"
        f"<td>{d.get('n')}</td><td>{esc(d.get('status'))}</td></tr>"
        for n, d in datasets.items())
    st_rows = "".join(
        "<tr>" + "".join(f"<td>{'' if r.get(c) is None else esc(r.get(c))}</td>"
                         for c in ("sid", "kind", "f0", "a0", "n_win",
                                   "reliable", "f0_model",
                                   "bedrock_depth_model", "vs30",
                                   "site_class_ec8")) + "</tr>"
        for r in rows)
    res_html = ""
    if result:
        pairs = [("stations used", result.get("n_stations")),
                 ("depth constraints", result.get("n_constraints")),
                 ("final misfit", f"{result.get('misfit'):.3g}"),
                 ("median f₀ residual", f"{result.get('f0_residual_pct'):.1f} %"),
                 ("max bedrock depth", f"{result.get('max_depth'):.0f} m"),
                 ("layer Vs (m/s)", " / ".join(map(str, result.get("vs_inv", []))))]
        if result.get("rms_vs_truth") is not None:
            pairs.append(("validation vs hidden truth",
                          f"RMS {result['rms_vs_truth']:.0f} m · "
                          f"corr {result['corr_vs_truth']:.2f}"))
        if result.get("depth_std_median") is not None:
            pairs.append(("ensemble depth σ (median)",
                          f"{result['depth_std_median']:.0f} m"))
        res_html = "<table class='kv'>" + "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pairs) + "</table>"
    charts = "".join(
        f"<figure><img src='/api/chart.png?k={k}'>"
        for k in ("curves", "f0a0", "fit", "depthdist"))
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>BasinInv3D — campaign report</title><style>
body{{font:13px/1.5 system-ui,sans-serif;color:#0b0b0b;background:#fff;
  max-width:880px;margin:24px auto;padding:0 18px}}
h1{{font-size:20px}} h2{{font-size:14px;margin:22px 0 8px;color:#52514e;
  text-transform:uppercase;letter-spacing:.06em}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
td,th{{border:1px solid #e1e0d9;padding:4px 8px;text-align:left}}
th{{background:#f4f4f1}} .kv td:first-child{{color:#52514e;width:40%}}
img{{width:100%;border:1px solid #e1e0d9;border-radius:6px;margin:6px 0}}
figure{{margin:0}} .muted{{color:#898781;font-size:11.5px}}
button{{padding:8px 18px;font-size:13px;border:1px solid #c3c2b7;
  border-radius:8px;background:#fff;cursor:pointer}}
@media print{{button{{display:none}}}}
</style></head><body>
<button onclick="window.print()">🖨 Print / save as PDF</button>
<h1>Microtremor campaign report</h1>
<div class="muted">generated {time.strftime('%Y-%m-%d %H:%M')} ·
reference {a[0]:.5f}, {a[1]:.5f} · BasinInv3D Field Dashboard</div>
<h2>Datasets</h2>
<table><tr><th>name</th><th>type</th><th>points</th><th>status</th></tr>
{ds_rows}</table>
<div class="muted" style="margin-top:4px">{ncons} auxiliary depth
constraint(s) active.</div>
<h2>Inversion result</h2>
{res_html or "<div class='muted'>no inversion run yet</div>"}
<h2>Stations — QC and site parameters</h2>
<table><tr><th>id</th><th>data</th><th>f₀ (Hz)</th><th>A₀</th><th>win</th>
<th>SESAME</th><th>f₀ model</th><th>bedrock (m)</th><th>Vs30</th>
<th>EC8</th></tr>{st_rows}</table>
<div class="muted" style="margin-top:4px">Vs30 and site class are derived
from the inverted model (time-averaged Vs of the top 30 m; EC8: A&gt;800,
B 360–800, C 180–360, D&lt;180 m/s).</div>
<h2>Figures</h2>{charts}
</body></html>"""


# ------------------------------------------------------------- run history
RUNS_FILE = os.path.join(WORK, "runs.json")


def _load_runs():
    try:
        with open(RUNS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


RUNS = _load_runs()


def _log_run(cfg, result):
    RUNS.append({
        "ts": time.strftime("%Y-%m-%d %H:%M"),
        "misfit": result.get("misfit"),
        "f0res": result.get("f0_residual_pct"),
        "max_depth": result.get("max_depth"),
        "n_stations": result.get("n_stations"),
        "n_constraints": result.get("n_constraints"),
        "layers": "/".join(str(int(l["vs"])) for l in cfg["layers"]),
        "ncx": cfg["ncx"], "ensemble": cfg["ensemble"],
        "rms": result.get("rms_vs_truth"), "corr": result.get("corr_vs_truth")})
    del RUNS[:-20]
    try:
        with open(RUNS_FILE, "w") as f:
            json.dump(RUNS, f)
    except Exception:
        pass


def depth_csv():
    """Bedrock-depth grid of the final model as lat,lon,depth CSV."""
    rb = _rebuild_model()
    if not rb:
        return None
    model, params, origin = rb
    a = _anchor() or DEFAULT_ANCHOR
    depth = model.interface_depths(params)[-1]
    lats, _ = FIO.local_to_latlon(np.zeros_like(model.grid.y),
                                  origin[1] + model.grid.y, *a)
    _, lons = FIO.local_to_latlon(origin[0] + model.grid.x,
                                  np.zeros_like(model.grid.x), *a)
    lines = ["lat,lon,bedrock_depth_m"]
    for i in range(model.grid.nx):
        for j in range(model.grid.ny):
            lines.append(f"{lats[j]:.6f},{lons[i]:.6f},{depth[i, j]:.1f}")
    return "\n".join(lines).encode()


def run_inversion(cfg):
    try:
        t_all = time.time()
        set_job(t_start=t_all, stage="prepare", note="collecting project data",
                progress=0.02, misfit=[], f0res=[], overlays={}, result=None)
        freqs, xy, hv_obs, sig_obs, constraints, used = _collect_inputs(cfg)
        log(f"inversion inputs: {len(xy)} stations, "
            f"{len(constraints)} depth constraints")
        grid, origin, margin, max_thick = _grid_for(xy, cfg)
        nL = len(cfg["layers"])
        log(f"grid {grid.nx}×{grid.ny}×{grid.nz} @ {grid.dx:.1f} m, "
            f"max depth {max_thick:.0f} m, {nL} layers")

        model = H.MultiLayerBasin(grid, n_layers=nL, ncx=cfg["ncx"],
                                  ncy=cfg["ncx"], margin_cells=margin)
        spec = H.LayerSpec(vs=[l["vs"] for l in cfg["layers"]]
                           + [cfg["bedrock_vs"]],
                           vpvs=cfg["vpvs"], Qs=cfg["Qs"], Qp=cfg["Qp"])
        x0 = H.initial_guess(model, spec, freqs, hv_obs)
        free = np.ones(model.n_params, bool)
        free[model.n_thick + nL] = False
        for L, l in enumerate(cfg["layers"]):
            if l.get("fix"):
                free[model.n_thick + L] = False

        # local constraints in grid frame
        cons = [dict(c, x=c["x"] - origin[0], y=c["y"] - origin[1])
                for c in constraints]
        inv = H.HVSRInversion(model, spec, xy - origin, freqs, hv_obs,
                              free_mask=free,
                              smooth_weight=cfg["smooth_weight"],
                              peak_weight=cfg["peak_weight"],
                              data_weight=cfg["data_weight"],
                              depth_constraints=cons,
                              constraint_weight=cfg["constraint_weight"])
        obs_f0 = inv.fpk_obs
        maxiter = cfg["maxiter"]
        set_job(stage="invert", note="starting inversion")
        vlim = [0.0, max_thick]

        def push_overlay(x, ver):
            depth = model.interface_depths(x)[-1]
            vmax = max(float(depth.max()), 10.0)
            bounds = _overlay_png(grid, origin, depth,
                                  os.path.join(OVL_DIR, "depth.png"),
                                  DEPTH_CMAP, 0.0, vmax)
            with _LOCK:
                JOB["overlays"]["depth"] = {
                    "url": "/overlays/depth.png", "v": ver, "bounds": bounds,
                    "vmin": 0.0, "vmax": vmax, "label": "bedrock depth (m)"}
            return depth

        def on_eval(k, f, x):
            fpk = H.soft_peak(freqs, inv.predict(x))
            res = 100.0 * float(np.median(np.abs(np.log(fpk / obs_f0))))
            depth = push_overlay(x, k)
            with _LOCK:
                JOB["misfit"].append(float(f))
                JOB["f0res"].append(res)
            set_job(progress=min(0.98, k / (maxiter * 1.3)),
                    note=f"eval {k}: misfit {f:.3e}, f₀ residual {res:.1f}%",
                    vs3d=_vs3d_payload(model, x, stations=xy - origin))
            if k % 10 == 0:
                log(f"  eval {k}: misfit {f:.4e}, f0 residual {res:.1f}%")
            _check_stop()

        inv.on_eval = on_eval
        x_final, ilog = inv.run(x0, max_thick=max_thick, maxiter=maxiter)
        _check_stop()

        # ------------------------------------------------------- report
        set_job(stage="report", note="final results", progress=0.99)
        inv_depth = push_overlay(x_final, "final")
        _, vs_inv = model.thickness_grid(x_final)
        fpk = H.soft_peak(freqs, inv.predict(x_final))
        f0res = 100.0 * float(np.median(np.abs(np.log(fpk / obs_f0))))
        result = dict(
            misfit0=float(ilog.misfit[0]) if ilog.misfit else None,
            misfit=float(ilog.misfit[-1]) if ilog.misfit else None,
            f0_residual_pct=f0res, n_evals=len(ilog.misfit),
            n_stations=len(xy), n_constraints=len(constraints),
            vs_inv=[int(v) for v in vs_inv],
            max_depth=float(inv_depth.max()),
            elapsed=time.time() - t_all,
            grid=dict(nx=grid.nx, ny=grid.ny, dx=float(grid.dx)))
        # constraint residuals (how well external knowledge is honoured)
        if cons:
            th, _ = model.columns_at(x_final, [[c["x"], c["y"]] for c in cons])
            zs = np.cumsum(th[:, :-1], axis=1)
            resid = [float(zs[i, cons[i]["layer"] - 1] - cons[i]["depth"])
                     for i in range(len(cons))]
            result["constraint_residuals_m"] = [round(r, 1) for r in resid]
        true_depth = _truth_depth(grid, origin)
        if true_depth is not None:
            result["rms_vs_truth"] = float(np.sqrt(np.mean(
                (inv_depth - true_depth) ** 2)))
            result["corr_vs_truth"] = float(np.corrcoef(
                inv_depth.ravel(), true_depth.ravel())[0, 1])
            log(f"validation: RMS {result['rms_vs_truth']:.0f} m, "
                f"corr {result['corr_vs_truth']:.2f}")

        # -------------------------------------------- uncertainty ensemble
        n_ens = cfg["ensemble"]
        if n_ens > 1:
            log(f"uncertainty ensemble: {n_ens - 1} extra members")
            erng = np.random.default_rng(1234)
            jobs = []
            for mem in range(1, n_ens):
                hv_m = hv_obs * np.exp(erng.normal(0, 1, hv_obs.shape) * sig_obs)
                ncx_m = int(np.clip(cfg["ncx"] + erng.integers(-1, 2), 2, 5))
                model_m = H.MultiLayerBasin(grid, n_layers=nL, ncx=ncx_m,
                                            ncy=ncx_m, margin_cells=margin)
                x0m = H.initial_guess(model_m, spec, freqs, hv_m,
                                      frac=float(erng.uniform(0.7, 1.3)))
                free_m = np.ones(model_m.n_params, bool)
                free_m[model_m.n_thick + nL] = False
                for L, l in enumerate(cfg["layers"]):
                    if l.get("fix"):
                        free_m[model_m.n_thick + L] = False
                sw = cfg["smooth_weight"] * float(erng.uniform(0.4, 2.5))
                jobs.append((model_m, spec, xy - origin, freqs, hv_m, free_m,
                             x0m, max_thick, maxiter, sw, cons))
            depths = [inv_depth]
            with ProcessPoolExecutor(max_workers=min(3, len(jobs))) as ex:
                for k, dm in enumerate(ex.map(H.invert_member, jobs)):
                    depths.append(dm)
                    set_job(note=f"uncertainty ensemble ({k + 1}/{n_ens - 1})",
                            progress=0.99)
                    _check_stop()
            std_d = np.stack(depths).std(0)
            smax = max(float(std_d.max()), 1.0)
            bounds = _overlay_png(grid, origin, std_d,
                                  os.path.join(OVL_DIR, "sigma.png"),
                                  SIGMA_CMAP, 0.0, smax)
            with _LOCK:
                JOB["overlays"]["sigma"] = {
                    "url": "/overlays/sigma.png", "v": "final",
                    "bounds": bounds, "vmin": 0.0, "vmax": smax,
                    "label": "depth uncertainty ±1σ (m)"}
            result["ensemble"] = n_ens
            result["depth_std_median"] = float(np.median(std_d))
            log(f"ensemble: median depth σ {np.median(std_d):.0f} m")

        global MODEL_STATE
        MODEL_STATE = {"grid": dict(nx=grid.nx, ny=grid.ny, nz=grid.nz,
                                    dx=float(grid.dx)),
                       "ncx": cfg["ncx"], "n_layers": nL, "margin": margin,
                       "params": [float(v) for v in x_final],
                       "origin": [float(origin[0]), float(origin[1])],
                       "vs": [float(v) for v in vs_inv]
                             + [float(cfg["bedrock_vs"])]}
        _register_vs30_overlay()
        set_job(stage="done", note="", progress=1.0, result=result,
                vs3d=_vs3d_payload(model, x_final, stations=xy - origin))
        _persist_results()
        _log_run(cfg, result)
        log(f"DONE in {result['elapsed']:.0f}s — max depth "
            f"{result['max_depth']:.0f} m, f0 residual {f0res:.1f}%")
    except StopRequested:
        set_job(stage="stopped", note="")
        log("stopped by user")
    except Exception:
        err = traceback.format_exc()
        set_job(stage="error", error=err)
        log("ERROR:\n" + err)
    finally:
        set_job(running=False)


def _parse_invert_cfg(body):
    def num(k, d, lo, hi):
        try:
            return float(np.clip(float(body.get(k, d)), lo, hi))
        except (TypeError, ValueError):
            return d
    layers = body.get("layers") or [{"vs": 300, "fix": False},
                                    {"vs": 450, "fix": False},
                                    {"vs": 620, "fix": False}]
    layers = [{"vs": float(np.clip(float(l.get("vs", 300)), 80, 3000)),
               "fix": bool(l.get("fix"))} for l in layers[:4]] or \
             [{"vs": 300.0, "fix": False}]
    return dict(
        layers=layers,
        bedrock_vs=num("bedrock_vs", 1800.0, 500, 4500),
        vpvs=num("vpvs", 2.2, 1.4, 4.0),
        Qs=num("Qs", 25.0, 5, 200), Qp=num("Qp", 40.0, 5, 300),
        fmin=num("fmin", 0.3, 0.05, 5.0), fmax=num("fmax", 12.0, 1.0, 50.0),
        nfreq=int(num("nfreq", 120, 40, 300)),
        ncx=int(num("ncx", 3, 2, 5)),
        smooth_weight=num("smooth_weight", 4e-3, 0.0, 1.0),
        peak_weight=num("peak_weight", 4.0, 0.0, 20.0),
        data_weight=num("data_weight", 0.4, 0.0, 10.0),
        constraint_weight=num("constraint_weight", 2.0, 0.0, 50.0),
        use_constraints=bool(body.get("use_constraints", True)),
        only_reliable=bool(body.get("only_reliable", False)),
        maxiter=int(num("maxiter", 70, 5, 400)),
        max_depth=num("max_depth", 0.0, 0.0, 3000.0) or None,
        ensemble=int(num("ensemble", 1, 1, 8)))


# =============================================================== figures
def curve_png(name, sid):
    """Small light-theme H/V figure for the point panel."""
    curves = _curves(name)
    if sid not in curves:
        return None
    f, hvm, sg = curves[sid]
    fig, ax = plt.subplots(figsize=(4.4, 2.6), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.fill_between(f, hvm * np.exp(-sg), hvm * np.exp(sg),
                    color="#9ec5f4", alpha=0.55, lw=0)
    ax.semilogx(f, hvm, color="#2a78d6", lw=2.0)
    i0 = int(np.argmax(hvm))
    ax.axvline(f[i0], color="#eb6834", lw=1.3, ls="--")
    ax.annotate(f"f₀ ≈ {f[i0]:.2f} Hz", (f[i0], hvm[i0]),
                textcoords="offset points", xytext=(6, -2), fontsize=8.5,
                color="#52514e")
    ax.grid(alpha=0.35, which="both", color="#e1e0d9", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=8)
    ax.set_xlabel("frequency (Hz)", fontsize=8.5, color="#52514e")
    ax.set_ylabel("H/V", fontsize=8.5, color="#52514e")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, facecolor="#fcfcfb")
    plt.close(fig)
    return buf.getvalue()


def colorbar_png(cmap, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(2.6, 0.62), constrained_layout=True)
    fig.patch.set_alpha(0.0)
    g = np.linspace(0, 1, 256)[None, :]
    ax.imshow(g, aspect="auto", cmap=cmap,
              extent=[vmin, vmax, 0, 1])
    ax.set_yticks([])
    ax.tick_params(colors="#52514e", labelsize=7.5)
    ax.set_title(label, fontsize=8, color="#52514e", pad=3)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, transparent=True)
    plt.close(fig)
    return buf.getvalue()


# ================================================================== server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str))

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "static", "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            with _LOCK:
                job = {k: v for k, v in JOB.items() if k != "log"}
                job["log"] = JOB["log"][-120:]
                proj = json.loads(json.dumps(PROJ, default=str))
            # station sesame dicts are bulky; trim for the poll
            for ds in proj["datasets"].values():
                for s in ds.get("stations", []):
                    s.pop("sesame", None)
            self._json({"project": proj, "job": job, "runs": RUNS[-15:],
                        "anchor": proj.get("anchor") or list(DEFAULT_ANCHOR)})
        elif path == "/api/point":
            name = q.get("ds", [""])[0]
            pid = q.get("id", [""])[0]
            with _LOCK:
                ds = PROJ["datasets"].get(name, {})
                pts = ds.get("stations", []) + ds.get("points", [])
                p = next((x for x in pts
                          if x.get("sid", x.get("pid")) == pid), None)
            self._json(p or {"error": "not found"}, 200 if p else 404)
        elif path == "/api/profile":
            try:
                p = profile_at(float(q["lat"][0]), float(q["lon"][0]))
            except (KeyError, ValueError):
                p = None
            self._json(p or {"error": "no inversion result yet"},
                       200 if p else 404)
        elif path == "/api/export/depth.csv":
            data = depth_csv()
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition",
                                 "attachment; filename=bedrock_depth.csv")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"error": "no inversion result yet"}, 404)
        elif path == "/api/curve.png":
            png = curve_png(q.get("ds", [""])[0], q.get("id", [""])[0])
            if png:
                self._send(200, png, "image/png")
            else:
                self._send(404, b"{}")
        elif path == "/api/colorbar.png":
            key = q.get("k", ["depth"])[0]
            if key == "f0":
                try:
                    vmin = float(q["vmin"][0])
                    vmax = float(q["vmax"][0])
                except (KeyError, ValueError):
                    self._send(404, b"{}")
                    return
                self._send(200, colorbar_png(
                    F0_CMAP.reversed(), vmin, vmax, "measured f₀ (Hz)"),
                    "image/png")
                return
            with _LOCK:
                o = JOB["overlays"].get(key)
            if not o:
                self._send(404, b"{}")
                return
            cmap = DEPTH_CMAP if key == "depth" else SIGMA_CMAP
            self._send(200, colorbar_png(cmap, o["vmin"], o["vmax"],
                                         o["label"]), "image/png")
        elif path == "/api/f0overlay":
            o = f0_overlay()
            self._json(o or {"error": "need at least 3 processed stations"},
                       200 if o else 404)
        elif path == "/api/vs30overlay":
            with _LOCK:
                o = JOB["overlays"].get("vs30")
            o = o or _register_vs30_overlay()
            self._json(o or {"error": "no inversion result yet"},
                       200 if o else 404)
        elif path == "/api/chart.png":
            png = chart_png(q.get("k", [""])[0])
            if png:
                self._send(200, png, "image/png")
            else:
                self._send(404, b"{}")
        elif path == "/report":
            self._send(200, report_html().encode(),
                       "text/html; charset=utf-8")
        elif path == "/api/export/stations.csv":
            data = stations_csv()
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition",
                                 "attachment; filename=stations.csv")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"error": "no stations loaded"}, 404)
        elif path.startswith("/overlays/"):
            fp = os.path.join(OVL_DIR, os.path.basename(path))
            if os.path.isfile(fp) and fp.endswith(".png"):
                with open(fp, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"{}")
        else:
            self._send(404, b"{}")

    # --------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""

        if path == "/api/upload":
            name = _safe_name(q.get("dataset", ["dataset"])[0])
            fname = _safe_name(os.path.basename(q.get("name", ["file"])[0]))
            folder = os.path.join(DS_DIR, name)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, fname), "wb") as f:
                f.write(raw)
            self._json({"ok": True, "dataset": name, "file": fname})
            return

        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return

        if path == "/api/dataset/commit":
            name = _safe_name(body.get("name", ""))
            kind = body.get("kind", "aux")
            aux_type = body.get("type", "other")
            if not os.path.isdir(os.path.join(DS_DIR, name)):
                self._json({"error": f"no uploaded data for '{name}'"}, 400)
                return
            with _LOCK:
                if kind == "hvsr" and JOB["running"]:
                    self._json({"error": "another job is running"}, 409)
                    return
                PROJ["datasets"][name] = {"kind": kind, "type": aux_type,
                                          "status": "processing", "n": 0}
            try:
                if kind == "hvsr":
                    with _LOCK:
                        JOB.update(_fresh_job())
                        JOB.update(running=True, kind="process")
                    threading.Thread(target=commit_hvsr_dataset, args=(name,),
                                     daemon=True).start()
                else:
                    commit_aux_dataset(name, aux_type)
                self._json({"ok": True})
            except Exception as exc:
                with _LOCK:
                    PROJ["datasets"].pop(name, None)
                self._json({"error": str(exc)}, 400)
        elif path == "/api/dataset/delete":
            name = body.get("name", "")
            with _LOCK:
                PROJ["datasets"].pop(name, None)
            shutil.rmtree(os.path.join(DS_DIR, name), ignore_errors=True)
            for ext in (".npz",):
                p = os.path.join(PROC_DIR, name + ext)
                if os.path.isfile(p):
                    os.remove(p)
            _set_anchor_from_stations()
            _relocalize_all()
            _save_project()
            log(f"dataset '{name}' removed")
            self._json({"ok": True})
        elif path == "/api/point/config":
            name, pid = body.get("ds", ""), body.get("id", "")
            with _LOCK:
                ds = PROJ["datasets"].get(name, {})
                pts = ds.get("stations", []) + ds.get("points", [])
                p = next((x for x in pts
                          if x.get("sid", x.get("pid")) == pid), None)
                if p is not None:
                    p.setdefault("cfg", {}).update(body.get("cfg", {}))
                    if "attrs" in body and "attrs" in p:
                        for k, v in body["attrs"].items():
                            if v is None:
                                p["attrs"].pop(k, None)
                            else:
                                try:
                                    p["attrs"][k] = float(v)
                                except (TypeError, ValueError):
                                    p["attrs"][k] = v
            if p is None:
                self._json({"error": "point not found"}, 404)
                return
            _save_project()
            self._json({"ok": True, "point": p})
        elif path == "/api/point/add":
            name = _safe_name(body.get("ds") or "manual_points")
            with _LOCK:
                ds = PROJ["datasets"].setdefault(
                    name, {"kind": "aux", "type": "manual", "status": "ready",
                           "points": [], "n": 0})
                pid = body.get("id") or f"M{len(ds['points']) + 1}"
                attrs = body.get("attrs") or {}
                for k, v in list(attrs.items()):
                    try:
                        attrs[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                p = dict(pid=pid, lat=float(body["lat"]),
                         lon=float(body["lon"]), attrs=attrs,
                         cfg=_default_point_cfg(attrs))
                ds["points"].append(p)
                ds["n"] = len(ds["points"])
            _localize(p)
            _save_project()
            log(f"manual point {pid} added at "
                f"({body['lat']:.5f}, {body['lon']:.5f})")
            self._json({"ok": True, "point": p})
        elif path == "/api/point/delete":
            name, pid = body.get("ds", ""), body.get("id", "")
            with _LOCK:
                ds = PROJ["datasets"].get(name, {})
                pts = ds.get("points")
                if pts is not None:
                    ds["points"] = [p for p in pts if p["pid"] != pid]
                    ds["n"] = len(ds["points"])
            _save_project()
            self._json({"ok": True})
        elif path == "/api/invert":
            with _LOCK:
                if JOB["running"]:
                    self._json({"error": "a job is already running"}, 409)
                    return
                cfg = _parse_invert_cfg(body)
                JOB.update(_fresh_job())
                JOB.update(running=True, kind="invert")
            log(f"starting inversion: {len(cfg['layers'])} layers, "
                f"constraints={'on' if cfg['use_constraints'] else 'off'}")
            threading.Thread(target=run_inversion, args=(cfg,),
                             daemon=True).start()
            self._json({"ok": True})
        elif path == "/api/section":
            try:
                png = section_png(float(body["lat1"]), float(body["lon1"]),
                                  float(body["lat2"]), float(body["lon2"]))
            except (KeyError, ValueError):
                png = None
            if png:
                self._send(200, png, "image/png")
            else:
                self._json({"error": "no inversion result yet"}, 404)
        elif path == "/api/stop":
            set_job(stop=True)
            log("stop requested…")
            self._json({"ok": True})
        elif path == "/api/project/clear":
            with _LOCK:
                if JOB["running"]:
                    self._json({"error": "a job is running"}, 409)
                    return
                PROJ.clear()
                PROJ.update(_empty_project())
                JOB.update(_fresh_job())
            for d in (DS_DIR, PROC_DIR, OVL_DIR):
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)
            _save_project()
            log("project cleared")
            self._json({"ok": True})
        elif path == "/api/demo":
            with _LOCK:
                if JOB["running"]:
                    self._json({"error": "a job is already running"}, 409)
                    return
                JOB.update(_fresh_job())
                JOB.update(running=True, kind="process")
            threading.Thread(target=_demo_job,
                             args=(int(body.get("seed", 2) or 2),),
                             daemon=True).start()
            self._json({"ok": True})
        else:
            self._send(404, b"{}")


def _demo_job(seed):
    """Generate the validation demo (hvsr campaign + boreholes + resistivity)
    and ingest it like uploaded data."""
    try:
        set_job(stage="processing", note="synthesising demo campaign…",
                progress=0.01)
        name = "demo_hvsr"
        folder = os.path.join(DS_DIR, name)
        shutil.rmtree(folder, ignore_errors=True)
        FIO.make_demo_campaign(folder, seed=seed, coords="latlon",
                               anchor=DEFAULT_ANCHOR)
        with _LOCK:
            PROJ["datasets"][name] = {"kind": "hvsr", "type": "hvsr",
                                      "status": "processing", "n": 0}
        for fname, aux_type in (("boreholes.csv", "borehole"),
                                ("resistivity.csv", "resistivity")):
            aname = _safe_name("demo_" + fname.split(".")[0])
            afold = os.path.join(DS_DIR, aname)
            shutil.rmtree(afold, ignore_errors=True)
            os.makedirs(afold, exist_ok=True)
            shutil.copy(os.path.join(folder, fname),
                        os.path.join(afold, fname))
            with _LOCK:
                PROJ["datasets"][aname] = {"kind": "aux", "type": aux_type,
                                           "status": "processing", "n": 0}
            commit_aux_dataset(aname, aux_type)
        commit_hvsr_dataset(name)      # runs in this thread; sets job idle
    except Exception:
        err = traceback.format_exc()
        set_job(stage="error", error=err, running=False)
        log("ERROR:\n" + err)


_restore_results()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8644)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Field Dashboard at http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
