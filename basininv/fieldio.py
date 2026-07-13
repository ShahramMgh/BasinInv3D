"""Field-campaign folder I/O for the Microtremor Field Assistant.

A *campaign* is a plain folder that mirrors how microtremor field data is
actually delivered:

    campaign/
      stations.csv      id,x,y[,elev]    local metric coordinates
      ST01.npz          raw 3-C record   keys: data (3, nt) as [N, E, Z], dt
      ST07.csv          raw 3-C record   "# dt=0.005" header, columns N,E,Z
      ST09.hv           processed H/V    columns: freq  hv  [sigma]
      ST12.mseed        raw 3-C record   read via obspy when installed
      truth.npz         OPTIONAL         synthetic ground truth, used only to
                                         score a validation run — never read
                                         by the inversion itself

Stations may freely mix raw records and already-processed ``.hv`` curves
(e.g. points recycled from an earlier survey).  ``scan_campaign`` gives a
cheap inventory for the dashboard; ``load_station`` returns the payload.

``make_demo_campaign`` writes an *imaginary* campaign in exactly this format:
records synthesised from a hidden layered basin, saved to disk and then
treated as real data — the drop-in path for actual field folders later.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

from .basin import GridSpec
from . import hvsr as H

RECORD_EXT = (".npz", ".csv", ".mseed", ".msd", ".sac")
HV_EXT = (".hv", ".hv.csv")


@dataclass
class StationEntry:
    sid: str
    x: float
    y: float
    kind: str          # "record" | "hv" | "missing" | "unsupported"
    path: str = ""
    meta: dict = None
    lat: float = None  # WGS84, when the coordinate file provides lat/lon
    lon: float = None


# --------------------------------------------------- geographic coordinates

M_PER_DEG_LAT = 110540.0


def latlon_to_local(lat, lon, lat0, lon0):
    """Equirectangular WGS84 -> local metres around (lat0, lon0).  Accurate to
    well under a metre over the few-km extent of a microtremor survey."""
    mx = 111320.0 * np.cos(np.radians(lat0))
    return ((np.asarray(lon, float) - lon0) * mx,
            (np.asarray(lat, float) - lat0) * M_PER_DEG_LAT)


def local_to_latlon(x, y, lat0, lon0):
    mx = 111320.0 * np.cos(np.radians(lat0))
    return (lat0 + np.asarray(y, float) / M_PER_DEG_LAT,
            lon0 + np.asarray(x, float) / mx)


def _parse_coord_row(row):
    """Extract (lat, lon, x, y) from a CSV row; any pair may be absent."""
    def g(*keys):
        for k in keys:
            if row.get(k) not in (None, ""):
                return float(row[k])
        return None
    return (g("lat", "latitude"), g("lon", "lng", "longitude"),
            g("x", "easting"), g("y", "northing"))


# ------------------------------------------------------------------ reading


def _find_station_file(folder, sid):
    for ext in HV_EXT:
        p = os.path.join(folder, sid + ext)
        if os.path.isfile(p):
            return p, "hv"
    for ext in RECORD_EXT:
        p = os.path.join(folder, sid + ext)
        if os.path.isfile(p):
            if ext in (".mseed", ".msd", ".sac"):
                try:
                    import obspy  # noqa: F401
                except ImportError:
                    return p, "unsupported"
            return p, "record"
    return "", "missing"


def _record_meta(path):
    """Cheap metadata (dt, duration) without holding the data."""
    try:
        if path.endswith(".npz"):
            with np.load(path) as z:
                dt = float(z["dt"])
                nt = int(z["data"].shape[-1])
            return {"dt": dt, "duration": nt * dt}
        if path.endswith(".csv"):
            dt, n = None, 0
            with open(path) as f:
                for line in f:
                    if line.startswith("#"):
                        if "dt=" in line:
                            dt = float(line.split("dt=")[1].split(",")[0])
                    elif line.strip():
                        n += 1
            if dt:
                return {"dt": dt, "duration": n * dt}
    except Exception:
        pass
    return {}


def scan_campaign(folder):
    """Inventory of a campaign folder: stations, data kinds, bbox, truth.

    The coordinate file (``stations.csv``) may give WGS84 ``lat, lon`` columns
    (preferred for real surveys) or local metric ``x, y``.  With lat/lon,
    local metres are derived around the mean station position (returned as
    ``anchor``) so the inversion machinery always sees metres.
    """
    st_file = os.path.join(folder, "stations.csv")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"campaign folder not found: {folder}")
    if not os.path.isfile(st_file):
        raise FileNotFoundError(f"no stations.csv in {folder}")
    entries = []
    with open(st_file) as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v for k, v in row.items() if k}
            sid = row["id"].strip()
            lat, lon, x, y = _parse_coord_row(row)
            if lat is None and x is None:
                raise ValueError(f"station {sid}: needs lat/lon or x/y")
            path, kind = _find_station_file(folder, sid)
            meta = _record_meta(path) if kind == "record" else {}
            entries.append(StationEntry(sid=sid, x=x, y=y, kind=kind,
                                        path=path, meta=meta, lat=lat, lon=lon))
    anchor = None
    lls = [(e.lat, e.lon) for e in entries if e.lat is not None]
    if lls:
        anchor = (float(np.mean([p[0] for p in lls])),
                  float(np.mean([p[1] for p in lls])))
        for e in entries:
            if e.lat is not None:
                ex, ey = latlon_to_local(e.lat, e.lon, *anchor)
                e.x, e.y = float(ex), float(ey)
    for e in entries:
        if e.x is None:
            raise ValueError(f"station {e.sid}: no usable coordinates")
    xs = [e.x for e in entries]
    ys = [e.y for e in entries]
    return {"folder": os.path.abspath(folder), "stations": entries,
            "bbox": [min(xs), max(xs), min(ys), max(ys)], "anchor": anchor,
            "has_truth": os.path.isfile(os.path.join(folder, "truth.npz"))}


def read_points_csv(path):
    """Generic auxiliary point dataset (boreholes, resistivity soundings, GPR
    lines' picks, geological observations…): a CSV with ``id`` plus ``lat,
    lon`` (or ``x, y``) columns; every other column becomes an attribute
    (numeric where possible).  Returns a list of dicts."""
    pts = []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            row = {k.strip().lower(): (v or "").strip()
                   for k, v in row.items() if k}
            lat, lon, x, y = _parse_coord_row(row)
            attrs = {}
            for k, v in row.items():
                if k in ("id", "lat", "latitude", "lon", "lng", "longitude",
                         "x", "y", "easting", "northing") or v == "":
                    continue
                try:
                    attrs[k] = float(v)
                except ValueError:
                    attrs[k] = v
            pts.append(dict(pid=row.get("id") or f"P{i + 1}", lat=lat,
                            lon=lon, x=x, y=y, attrs=attrs))
    return pts


def _read_record_csv(path):
    dt = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if "dt=" in line:
                    dt = float(line.split("dt=")[1].split(",")[0])
                continue
            s = line.strip()
            if s:
                rows.append([float(v) for v in s.replace(",", " ").split()])
    a = np.asarray(rows, float)
    if a.shape[1] >= 4:                    # explicit time column
        dt = float(np.median(np.diff(a[:, 0])))
        a = a[:, 1:4]
    if dt is None:
        raise ValueError(f"{path}: no '# dt=' header and no time column")
    return a[:, :3].T.copy(), dt           # (3=N,E,Z, nt)


def _read_record_obspy(path):
    from obspy import read
    st = read(path)
    comps = {}
    for tr in st:
        c = (tr.stats.channel or "Z")[-1].upper()
        comps.setdefault(c, tr)
    order = [comps.get("N") or comps.get("1"), comps.get("E") or comps.get("2"),
             comps.get("Z")]
    if any(t is None for t in order):
        order = list(st[:3])
    nt = min(len(t.data) for t in order)
    dt = float(order[0].stats.delta)
    return np.stack([np.asarray(t.data[:nt], float) for t in order]), dt


def _read_hv(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(v) for v in s.replace(",", " ").split()])
    a = np.asarray(rows, float)
    sigma = a[:, 2] if a.shape[1] > 2 else None
    return a[:, 0], a[:, 1], sigma


def load_station(entry: StationEntry):
    """Load one station's payload.

    Returns ("record", data (3, nt), dt) or ("hv", freqs, hv, sigma|None).
    """
    p = entry.path
    if entry.kind == "hv":
        return ("hv",) + _read_hv(p)
    if entry.kind != "record":
        raise ValueError(f"station {entry.sid}: no usable data ({entry.kind})")
    if p.endswith(".npz"):
        with np.load(p) as z:
            return "record", np.asarray(z["data"], float), float(z["dt"])
    if p.endswith(".csv"):
        return ("record",) + _read_record_csv(p)
    return ("record",) + _read_record_obspy(p)


def load_truth(folder):
    """Optional synthetic ground truth for validation scoring, or None."""
    p = os.path.join(folder, "truth.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        out = {"true_maps": np.asarray(z["true_maps"], float),
               "vs": np.asarray(z["vs"], float),
               "dx": float(z["dx"]), "origin": np.asarray(z["origin"], float)}
        if "lat0" in z:                    # geographic anchor of grid (0, 0)
            out["lat0"] = float(z["lat0"])
            out["lon0"] = float(z["lon0"])
        return out


# ------------------------------------------------------------------ writing


def make_demo_campaign(folder, seed=2, n_layers=3, n_side=5, dx=15.0,
                       nx=44, ny=44, nz=24, max_total_depth=140.0,
                       duration=328.0, dt=0.01, noise_pct=5.0,
                       origin=(3200.0, 5100.0), vs=None, coords="xy",
                       anchor=(35.7000, 51.4000)):
    """Write an imaginary field campaign FOR VALIDATION: a hidden layered
    basin is built, a 3-component ambient record is synthesised at every
    station and saved to disk in the real formats (mostly .npz, some .csv, a
    few processed .hv), plus stations.csv and a truth.npz for scoring.

    coords="xy" writes local metric coordinates (offset by `origin`);
    coords="latlon" writes WGS84 lat/lon around `anchor` — the format of a
    real GPS-positioned survey — and adds two auxiliary point files in the
    same folder: boreholes.csv (bedrock_depth from the truth, small error)
    and resistivity.csv (an interpreted bedrock depth), for exercising the
    constraint workflow.
    """
    os.makedirs(folder, exist_ok=True)
    grid = GridSpec(nx=nx, ny=ny, nz=nz, dx=dx)
    true_maps, vs_true = H.make_true_basin(
        grid, n_layers=n_layers, seed=seed, vs=vs,
        max_total_depth=max_total_depth)
    xy = H.station_lattice(grid, n_side=n_side, margin_cells=7,
                           jitter=0.35, seed=seed + 3)
    th = H.sample_true_columns(grid, true_maps, xy)
    th = np.hstack([th, np.full((len(xy), 1), 1.0e4)])
    freqs = np.geomspace(0.3, 12.0, 120)
    nt = int(round(duration / dt))
    origin = np.asarray(origin, float)

    rows, kinds = [], []
    for i in range(len(xy)):
        sid = f"ST{i + 1:02d}"
        gx, gy = xy[i]                      # grid-local coords
        fx, fy = origin + (gx, gy)          # "field" coords written to disk
        if i % 9 == 4:                      # a few pre-processed H/V points
            hv = H.observed_hvsr(vs_true, th[i:i + 1], freqs,
                                 noise_pct=noise_pct, seed=seed * 100 + i)[0]
            sig = np.full_like(freqs, noise_pct / 100.0)
            with open(os.path.join(folder, sid + ".hv"), "w") as f:
                f.write("# processed H/V curve (from an earlier survey)\n"
                        "# freq(Hz)  hv  sigma(log)\n")
                for a, b, c in zip(freqs, hv, sig):
                    f.write(f"{a:.5f}  {b:.5f}  {c:.4f}\n")
            kind = "hv"
        else:
            data, _ = H.synth_microtremor(vs_true, th[i], freqs, dt=dt,
                                          nt=nt, seed=seed * 1000 + i)
            if i % 5 == 2:                  # some records delivered as CSV
                with open(os.path.join(folder, sid + ".csv"), "w") as f:
                    f.write(f"# ambient 3-component record, dt={dt}\n"
                            "# columns: N, E, Z\n")
                    np.savetxt(f, data.T, fmt="%.6e", delimiter=",")
            else:
                np.savez_compressed(os.path.join(folder, sid + ".npz"),
                                    data=data.astype(np.float32), dt=dt)
            kind = "record"
        rows.append((sid, fx, fy))
        kinds.append(kind)

    rng = np.random.default_rng(seed + 55)
    depth_t = np.cumsum(true_maps, axis=0)
    if coords == "latlon":
        # grid-local (0,0) sits at `anchor`; station latlon from grid coords
        with open(os.path.join(folder, "stations.csv"), "w") as f:
            f.write("id,lat,lon\n")
            for (sid, _, _), (gx, gy) in zip(rows, xy):
                la, lo = local_to_latlon(gx, gy, *anchor)
                f.write(f"{sid},{la:.6f},{lo:.6f}\n")
        # auxiliary constraint files (bedrock depth from truth ± error)
        from scipy.interpolate import RectBivariateSpline
        spl = RectBivariateSpline(grid.x, grid.y, depth_t[-1])
        n_bh = min(3, len(xy))
        picks = rng.choice(len(xy), size=n_bh, replace=False)
        with open(os.path.join(folder, "boreholes.csv"), "w") as f:
            f.write("id,lat,lon,bedrock_depth,drilled_by\n")
            for b, i in enumerate(picks):
                bx = xy[i, 0] + rng.uniform(-25, 25)
                by = xy[i, 1] + rng.uniform(-25, 25)
                d = float(spl(bx, by)[0, 0]) * rng.uniform(0.97, 1.03)
                la, lo = local_to_latlon(bx, by, *anchor)
                f.write(f"BH{b+1},{la:.6f},{lo:.6f},{d:.1f},demo drilling\n")
        with open(os.path.join(folder, "resistivity.csv"), "w") as f:
            f.write("id,lat,lon,interp_bedrock_depth,array\n")
            for r in range(2):
                bx = rng.uniform(0.3, 0.7) * grid.x[-1]
                by = rng.uniform(0.3, 0.7) * grid.y[-1]
                d = float(spl(bx, by)[0, 0]) * rng.uniform(0.90, 1.10)
                la, lo = local_to_latlon(bx, by, *anchor)
                f.write(f"VES{r+1},{la:.6f},{lo:.6f},{d:.1f},Schlumberger\n")
        np.savez_compressed(os.path.join(folder, "truth.npz"),
                            true_maps=true_maps, vs=vs_true, dx=dx,
                            origin=(0.0, 0.0), lat0=anchor[0], lon0=anchor[1])
    else:
        with open(os.path.join(folder, "stations.csv"), "w") as f:
            f.write("id,x,y\n")
            for sid, fx, fy in rows:
                f.write(f"{sid},{fx:.1f},{fy:.1f}\n")
        np.savez_compressed(os.path.join(folder, "truth.npz"),
                            true_maps=true_maps, vs=vs_true, dx=dx,
                            origin=origin)
    with open(os.path.join(folder, "README.txt"), "w") as f:
        f.write(
            "Imaginary microtremor field campaign (BasinInv3D demo).\n"
            f"{len(rows)} stations over a hidden {n_layers}-layer basin; "
            f"records {duration:.0f} s @ dt={dt} s.\n\n"
            "Format (drop real data in the same way):\n"
            "  stations.csv   id,x,y  (local metric coordinates)\n"
            "  <id>.npz       raw record: data (3, nt) as [N, E, Z], dt\n"
            "  <id>.csv       raw record: '# dt=...' header, columns N,E,Z\n"
            "  <id>.hv        processed H/V: freq  hv  [sigma]\n"
            "  <id>.mseed     miniSEED (needs obspy)\n"
            "  truth.npz      demo-only ground truth for scoring\n")
    return {"folder": os.path.abspath(folder), "n_stations": len(rows),
            "kinds": kinds, "vs_true": [float(v) for v in vs_true],
            "max_depth": float(np.cumsum(true_maps, 0)[-1].max())}
