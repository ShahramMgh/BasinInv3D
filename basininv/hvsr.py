"""Ambient-microtremor (HVSR) path: invert station H/V curves for a 3D
multi-layer sediment Vs structure.

This is the field-data-oriented companion to the active-source elastic FWI in
:mod:`basininv.inversion`.  Instead of one soft layer over bedrock recovered
from full waveforms, here we recover a **stack of sediment layers** whose
interface depths vary across the basin, from the **H/V spectral ratios** of
microtremor recordings at a scattered set of surface stations.

Forward model
-------------
Under each station the earth is a 1-D layered column (layer velocities are
laterally constant in this phase; only the interface depths vary across x, y).
The HVSR is modelled as the ratio of the surface amplification of horizontally
polarised shear waves to that of vertically incident P waves:

    HVSR(f) = A_SH(f; Vs profile) / A_P(f; Vp profile)

Each amplification is the classic 1-D vertical-incidence transfer function of a
damped layered medium (Kramer 1996, propagator recursion) — a standard,
inexpensive HVSR proxy whose fundamental peak reproduces f0 = Vs/4H for a single
layer.  Rayleigh-wave ellipticity / diffuse-field HVSR are the accuracy upgrade
path.  Because one forward is only a few matrix recursions over frequency, the
whole survey costs milliseconds, so finite-difference gradients over a dense
parameterization are entirely practical.

Parameterization
----------------
Layer *thicknesses* (not absolute depths) live on coarse control-node grids and
are bicubic-interpolated to any (x, y); thicknesses are non-negative, so the
interfaces can never cross.  Layer Vs are global scalars.  Any subset of the
parameters can be *fixed* (held at a supplied value) — e.g. a known bedrock
interface, a fixed Vs contrast, or a known layer depth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from .basin import GridSpec

# ------------------------------------------------------------------ forward


def _transfer_1d(vel, rho, thick, freqs, Q):
    """Surface amplification |1/A_bottom| of a 1-D damped layered column at
    vertical incidence, for a wave with the given velocity profile.

    vel, rho, thick : arrays over N sediment layers (thick in metres).
    A halfspace with the last given (vel, rho) is assumed below the stack; its
    thickness entry is ignored.  Returns amplification over `freqs`.
    """
    vel = np.asarray(vel, float)
    rho = np.asarray(rho, float)
    thick = np.asarray(thick, float)
    w = 2.0 * np.pi * np.asarray(freqs, float)
    # complex velocity with hysteretic damping
    vc = vel * (1.0 + 0.5j / Q)
    Z = rho * vc                                   # (nlayer,)
    A = np.ones_like(w, dtype=complex)             # surface: A1 = B1 = 1
    B = np.ones_like(w, dtype=complex)
    nlay = len(vel)
    for i in range(nlay - 1):                      # propagate through layers
        k = w / vc[i]
        e = np.exp(1j * k * thick[i])
        alpha = Z[i] / Z[i + 1]
        A2 = 0.5 * A * (1 + alpha) * e + 0.5 * B * (1 - alpha) / e
        B2 = 0.5 * A * (1 - alpha) * e + 0.5 * B * (1 + alpha) / e
        A, B = A2, B2
    return 1.0 / np.maximum(np.abs(A), 1e-30)      # surface vs bedrock-outcrop


def _density(vp):
    """Simple Gardner-style density from P velocity (kg/m^3)."""
    return 310.0 * np.power(np.maximum(np.asarray(vp, float), 300.0), 0.25) * 3.0


def hvsr_curve(vs, thick, freqs, vp=None, rho=None, vpvs=2.2,
               Qs=25.0, Qp=40.0):
    """Modelled H/V curve of one layered column.

    vs, thick : per-layer shear velocity (m/s) and thickness (m); the last
                entry is the bedrock halfspace (its thickness is ignored).
    """
    vs = np.asarray(vs, float)
    if vp is None:
        vp = vpvs * vs
    if rho is None:
        rho = _density(vp)
    Ah = _transfer_1d(vs, rho, thick, freqs, Qs)
    Av = _transfer_1d(np.asarray(vp, float), rho, thick, freqs, Qp)
    return Ah / np.maximum(Av, 1e-30)


def fundamental_frequency(vs, thick):
    """Quarter-wavelength fundamental resonance of the sediment stack (Hz)."""
    vs = np.asarray(vs, float)[:-1]
    thick = np.asarray(thick, float)[:-1]
    if len(thick) == 0 or thick.sum() <= 0:
        return np.nan
    travel = np.sum(thick / vs)                    # one-way SH traveltime
    return 1.0 / (4.0 * travel)


def soft_peak(freqs, hv, q=16.0):
    """Differentiable estimate of the HVSR peak frequency: a curvefit-free
    weighted average that concentrates on the largest values.  `hv` may be 1-D
    (nfreq) or 2-D (nstation, nfreq)."""
    f = np.asarray(freqs, float)
    hv = np.asarray(hv, float)
    w = np.maximum(hv, 1e-6) ** q
    return np.sum(w * f, axis=-1) / np.sum(w, axis=-1)


# ------------------------------------------------------- parameterization


@dataclass
class LayerSpec:
    """Static description of the sediment stack (velocities are the *initial*
    guess; which quantities are actually inverted is set by the fix flags)."""
    vs: list                     # per sediment layer + bedrock (len = nL+1)
    vpvs: float = 2.2
    Qs: float = 25.0
    Qp: float = 40.0

    @property
    def n_layers(self):          # sediment layers (excludes bedrock halfspace)
        return len(self.vs) - 1


class MultiLayerBasin:
    """Map a parameter vector <-> (per-layer thickness maps, layer Vs).

    Parameters = [thickness nodes: nL x ncx x ncy] + [Vs: nL+1].
    Thicknesses are metres and non-negative; bedrock Vs (last) is usually fixed.
    """

    def __init__(self, grid: GridSpec, n_layers, ncx=3, ncy=3, margin_cells=12):
        self.grid = grid
        self.n_layers = n_layers
        self.ncx, self.ncy = ncx, ncy
        self.margin = margin_cells
        pad = margin_cells * grid.dx
        self.node_x = np.linspace(pad, grid.x[-1] - pad, ncx)
        self.node_y = np.linspace(pad, grid.y[-1] - pad, ncy)
        self.n_thick = n_layers * ncx * ncy

    @property
    def n_params(self):
        return self.n_thick + (self.n_layers + 1)

    # ---- packing -------------------------------------------------------
    def pack(self, thick_nodes, vs):
        return np.concatenate([np.asarray(thick_nodes, float).ravel(),
                               np.asarray(vs, float).ravel()])

    def unpack(self, params):
        params = np.asarray(params, float)
        tn = params[:self.n_thick].reshape(self.n_layers, self.ncx, self.ncy)
        vs = params[self.n_thick:]
        return tn, vs

    def _spl(self, node_vals):
        kx = min(3, self.ncx - 1)
        ky = min(3, self.ncy - 1)
        return RectBivariateSpline(self.node_x, self.node_y, node_vals,
                                   kx=kx, ky=ky)

    # ---- evaluation at scattered stations ------------------------------
    def columns_at(self, params, xy):
        """Return (thick, vs) for every station.

        thick : (nstation, n_layers+1) with a large halfspace thickness last.
        vs    : (n_layers+1,) global layer velocities.
        """
        tn, vs = self.unpack(params)
        xy = np.atleast_2d(np.asarray(xy, float))
        th = np.empty((len(xy), self.n_layers + 1))
        for L in range(self.n_layers):
            spl = self._spl(tn[L])
            th[:, L] = np.clip(spl(xy[:, 0], xy[:, 1], grid=False), 0.0, None)
        th[:, -1] = 1.0e4                          # bedrock halfspace
        return th, vs

    # ---- evaluation on the full grid (for volumes / figures) ----------
    def thickness_grid(self, params):
        tn, vs = self.unpack(params)
        maps = np.stack([np.clip(self._spl(tn[L])(self.grid.x, self.grid.y),
                                 0.0, None) for L in range(self.n_layers)])
        return maps, vs                            # (nL, nx, ny), (nL+1,)

    def interface_depths(self, params):
        """Cumulative interface depths z_1..z_nL on the grid (metres)."""
        maps, _ = self.thickness_grid(params)
        return np.cumsum(maps, axis=0)             # (nL, nx, ny)

    def vs_volume(self, params):
        """Vs(x, y, z) sampled on the grid, for cross-sections / 3-D views."""
        maps, vs = self.thickness_grid(params)
        z = self.grid.z                            # (nz,)
        depth = np.cumsum(maps, axis=0)            # interface depths
        vol = np.full((self.grid.nx, self.grid.ny, self.grid.nz),
                      float(vs[-1]))               # bedrock everywhere first
        # fill from the bottom sediment layer upward so shallow layers win
        for L in range(self.n_layers - 1, -1, -1):
            top = depth[L - 1] if L > 0 else np.zeros_like(maps[0])
            mask = z[None, None, :] < depth[L][:, :, None]
            mask &= z[None, None, :] >= top[:, :, None]
            vol = np.where(mask, vs[L], vol)
        return vol

    # ---- bounds --------------------------------------------------------
    def bounds(self, max_thick, vs_range=(120.0, 2500.0)):
        b = [(0.0, max_thick)] * self.n_thick
        b += [vs_range] * (self.n_layers + 1)
        return b


# ------------------------------------------------------- synthetic data


def gaussian_layer_thickness(grid: GridSpec, bumps, base=0.0, margin_cells=12):
    """A smooth thickness map: constant `base` plus anisotropic Gaussians."""
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    th = np.full_like(X, float(base))
    for b in bumps:
        c, s = np.cos(b.get("theta", 0.0)), np.sin(b.get("theta", 0.0))
        xr = c * (X - b["x0"]) + s * (Y - b["y0"])
        yr = -s * (X - b["x0"]) + c * (Y - b["y0"])
        th += b["amp"] * np.exp(-0.5 * ((xr / b["sx"]) ** 2 + (yr / b["sy"]) ** 2))
    return np.clip(th, 0.0, None)


def synth_microtremor(vs, thick, freqs, dt=0.01, nt=4096, seed=0,
                      vpvs=2.2, Qs=25.0, Qp=40.0):
    """A 3-component ambient-noise time series whose H/V matches the layered
    column — white input shaped by the H and V transfer functions.  Lets the
    web app run the genuine 'time series -> HVSR extraction -> inversion' path.
    """
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(nt, dt)
    fp = np.clip(f, freqs[0] * 0.25, None)
    vs = np.asarray(vs, float)
    vp = vpvs * vs
    rho = _density(vp)
    Ah = _transfer_1d(vs, rho, thick, fp, Qs)
    Av = _transfer_1d(vp, rho, thick, fp, Qp)
    band = ((f >= freqs[0] * 0.5) & (f <= freqs[-1] * 1.5)).astype(float)
    band = np.convolve(band, np.hanning(9) / np.hanning(9).sum(), "same")

    def comp(A):
        ph = np.exp(2j * np.pi * rng.random(len(f)))
        spec = A * band * ph * rng.normal(1.0, 0.15, len(f))
        return np.fft.irfft(spec, n=nt).astype(np.float32)

    return np.stack([comp(Ah), comp(Ah), comp(Av)]), dt   # (3, nt)


def make_true_basin(grid: GridSpec, n_layers=3, seed=1, vs=None,
                    margin_cells=12, max_total_depth=140.0):
    """Build an imaginary layered basin appropriate for microtremor HVSR:
    increasing Vs with depth and smooth, non-crossing layer thicknesses that
    thicken toward the basin centre.

    `max_total_depth` (m) is kept modest so the fundamental resonance stays
    inside a measurable microtremor band (~0.3-10 Hz): f0 ~ Vs/4H, so a
    140 m / Vs~300 basin resonates near 0.5 Hz.  Deeper basins push f0 below
    the band and cannot be resolved by HVSR — that is a physical limit, not a
    code limit.
    """
    rng = np.random.default_rng(seed)
    if vs is None:
        vs = list(np.linspace(280, 620, n_layers)) + [1800.0]
    x0 = rng.uniform(0.42, 0.58) * grid.x[-1]
    y0 = rng.uniform(0.42, 0.58) * grid.y[-1]
    th = rng.uniform(-1.0, 1.0)
    maps = []
    for L in range(n_layers):
        amp = (max_total_depth / n_layers) * rng.uniform(0.75, 1.25)
        maps.append(gaussian_layer_thickness(
            grid, [dict(amp=amp, x0=x0, y0=y0,
                        sx=rng.uniform(0.18, 0.28) * grid.x[-1],
                        sy=rng.uniform(0.18, 0.28) * grid.y[-1], theta=th)],
            base=0.10 * max_total_depth / n_layers, margin_cells=margin_cells))
    return np.stack(maps), np.asarray(vs, float)


def station_lattice(grid: GridSpec, n_side=6, margin_cells=12, jitter=0.0,
                    seed=0):
    """A (jittered) grid of surface stations, returned as (x, y) metres."""
    rng = np.random.default_rng(seed)
    pad = margin_cells * grid.dx
    xs = np.linspace(pad, grid.x[-1] - pad, n_side)
    ys = np.linspace(pad, grid.y[-1] - pad, n_side)
    xy = np.array([(x, y) for x in xs for y in ys], float)
    if jitter:
        xy += rng.uniform(-jitter, jitter, xy.shape) * grid.dx
    return xy


def observed_hvsr(vs_true, th_true, freqs, noise_pct=5.0, seed=0):
    """Synthetic 'observed' HVSR per station: theoretical curve of the true
    column plus realistic multiplicative log-noise and a small peak jitter."""
    rng = np.random.default_rng(seed)
    hv = np.array([hvsr_curve(vs_true, th_true[i], freqs)
                   for i in range(len(th_true))])
    if noise_pct > 0:
        sig = noise_pct / 100.0
        hv = hv * np.exp(rng.normal(0, sig, hv.shape))
        # mild spectral smoothing, as real HVSR processing applies
        k = np.hanning(7); k /= k.sum()
        hv = np.array([np.convolve(h, k, "same") for h in hv])
    return np.maximum(hv, 1e-3)


def initial_guess(model: MultiLayerBasin, spec: LayerSpec, freqs, hv_obs,
                  frac=1.0):
    """Flat starting model whose fundamental matches the *median observed peak*
    — the key to avoiding HVSR cycle-skipping.  Splits the required one-way
    traveltime equally across the sediment layers."""
    fpk = np.median(soft_peak(freqs, hv_obs))
    fpk = float(np.clip(fpk, freqs[0] * 1.2, freqs[-1] * 0.9))
    travel = 1.0 / (4.0 * fpk)                          # target one-way time
    vs = np.asarray(spec.vs, float)
    nL = model.n_layers
    t_each = travel / nL * frac
    thick = np.array([t_each * vs[L] for L in range(nL)])
    tn = np.stack([np.full((model.ncx, model.ncy), thick[L]) for L in range(nL)])
    return model.pack(tn, spec.vs)


def sample_true_columns(grid, true_maps, xy):
    """Layer thicknesses of the true model interpolated at station (x, y)."""
    xy = np.atleast_2d(xy)
    out = np.empty((len(xy), true_maps.shape[0]))
    for L in range(true_maps.shape[0]):
        spl = RectBivariateSpline(grid.x, grid.y, true_maps[L])
        out[:, L] = np.clip(spl(xy[:, 0], xy[:, 1], grid=False), 0.0, None)
    return out


# --------------------------------------------------------------- inversion


@dataclass
class HVSRLog:
    misfit: list = field(default_factory=list)
    params: list = field(default_factory=list)


class HVSRInversion:
    """Fit modelled station HVSR curves to observed ones for the layer
    thickness maps (and, optionally, layer Vs).  Any parameter can be fixed."""

    def __init__(self, model: MultiLayerBasin, spec: LayerSpec, station_xy,
                 freqs, hv_obs, free_mask=None, smooth_weight=3e-3,
                 order_weight=0.0, peak_weight=4.0, data_weight=0.4):
        self.model = model
        self.spec = spec
        self.xy = np.atleast_2d(np.asarray(station_xy, float))
        self.freqs = np.asarray(freqs, float)
        self.hv_obs = np.asarray(hv_obs, float)          # (nstation, nfreq)
        self.logobs = np.log(np.maximum(self.hv_obs, 1e-6))
        self.fpk_obs = soft_peak(self.freqs, self.hv_obs)
        self.smooth_weight = smooth_weight
        self.order_weight = order_weight
        self.peak_weight = peak_weight
        self.data_weight = data_weight
        n = model.n_params
        self.free = np.ones(n, bool) if free_mask is None else np.asarray(free_mask, bool)
        self.on_eval = None
        self.log = HVSRLog()

    # ---- forward for all stations -------------------------------------
    def predict(self, params):
        th, vs = self.model.columns_at(params, self.xy)
        vp = self.spec.vpvs * vs
        rho = _density(vp)
        out = np.empty((len(self.xy), len(self.freqs)))
        for i in range(len(self.xy)):
            Ah = _transfer_1d(vs, rho, th[i], self.freqs, self.spec.Qs)
            Av = _transfer_1d(vp, rho, th[i], self.freqs, self.spec.Qp)
            out[i] = Ah / np.maximum(Av, 1e-30)
        return out

    # ---- objective -----------------------------------------------------
    def _reg(self, params):
        tn, vs = self.model.unpack(params)
        r = 0.0
        for L in range(self.model.n_layers):
            nd = tn[L]
            r += np.sum(np.diff(nd, 1, axis=0) ** 2) + np.sum(np.diff(nd, 1, axis=1) ** 2)
        scale = (self.model.grid.nz * self.model.grid.dx) ** 2
        reg = self.smooth_weight * r / scale
        if self.order_weight > 0:                  # encourage Vs increasing
            dv = np.diff(vs)
            reg += self.order_weight * np.sum(np.minimum(dv, 0.0) ** 2)
        return reg

    def misfit(self, params):
        pred = self.predict(params)
        logpred = np.log(np.maximum(pred, 1e-6))
        data = self.data_weight * 0.5 * float(np.mean((logpred - self.logobs) ** 2))
        # peak-frequency term: smooth, monotonic in sediment traveltime, so it
        # pulls the fundamental toward the data instead of cycle-skipping
        peak = 0.0
        if self.peak_weight > 0:
            fpk = soft_peak(self.freqs, pred)
            peak = self.peak_weight * float(np.mean(
                (np.log(fpk) - np.log(self.fpk_obs)) ** 2))
        return data + peak + self._reg(params)

    def _grad(self, x_free, x_full, step):
        x_full = x_full.copy()
        x_full[self.free] = x_free
        f0 = self.misfit(x_full)
        g = np.zeros(np.count_nonzero(self.free))
        idx = np.where(self.free)[0]
        for j, i in enumerate(idx):
            xp = x_full.copy()
            xp[i] += step[i]
            g[j] = (self.misfit(xp) - f0) / step[i]
        return f0, g

    def run(self, x0, max_thick, maxiter=60, verbose=False):
        x0 = np.asarray(x0, float)
        bounds_full = self.model.bounds(max_thick)
        step = np.full(self.model.n_params, 0.5 * self.model.grid.dx)
        step[self.model.n_thick:] = 8.0            # m/s for Vs
        idx = np.where(self.free)[0]

        # variable scaling: L-BFGS-B expects O(1) variables, but thicknesses are
        # tens–hundreds of metres and Vs hundreds of m/s.  Optimise y = x / sc.
        sc = np.full(self.model.n_params, max(0.3 * max_thick, 40.0))
        sc[self.model.n_thick:] = 250.0
        scf = sc[idx]
        y0 = x0[idx] / scf
        ybounds = [(bounds_full[i][0] / sc[i], bounds_full[i][1] / sc[i])
                   for i in idx]

        def fun(yf):
            xf = yf * scf
            f, g = self._grad(xf, x0, step)        # g wrt x (free)
            full = x0.copy()
            full[self.free] = xf
            self.log.misfit.append(f)
            self.log.params.append(full.copy())
            if verbose:
                print(f"  eval {len(self.log.misfit):3d}  misfit {f:.4e}", flush=True)
            if self.on_eval is not None:
                self.on_eval(len(self.log.misfit), f, full)
            return f, g * scf                      # chain rule: dF/dy = dF/dx * sc

        res = minimize(fun, y0, jac=True, method="L-BFGS-B", bounds=ybounds,
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-10, "maxls": 25})
        xf = x0.copy()
        xf[self.free] = res.x * scf
        return xf, self.log
