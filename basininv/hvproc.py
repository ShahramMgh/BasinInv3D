"""Field-grade HVSR processing: 3-component ambient-noise records -> H/V
curves with uncertainties and SESAME quality checks.

This is the acquisition-side companion to :mod:`basininv.hvsr` (which models
and inverts the curves).  It implements the standard microtremor processing
chain used on real recordings:

  1. split the record into overlapping windows with a cosine taper
  2. reject windows contaminated by transients (STA/LTA anti-trigger)
  3. FFT each component, smooth spectra with the Konno-Ohmachi window
  4. merge horizontals (geometric or quadratic mean) and form H/V per window
  5. average log H/V over windows -> mean curve and a multiplicative sigma
  6. pick f0 per window (parabolic refinement) -> f0 and its scatter
  7. evaluate the SESAME (2004) reliability and clear-peak criteria

Every step is controlled by :class:`ProcConfig`, so the web dashboard can
expose the full set of field-analysis parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------- configuration


@dataclass
class ProcConfig:
    """All parameters of the record -> H/V processing chain."""
    win_len: float = 40.0        # window length (s)
    overlap: float = 0.50        # fraction of window overlap
    taper: float = 0.10          # cosine taper fraction per window end
    detrend: bool = True         # remove mean + linear trend per window
    fmin: float = 0.3            # output band (Hz)
    fmax: float = 12.0
    nfreq: int = 120             # log-spaced output frequencies
    ko_b: float = 40.0           # Konno-Ohmachi bandwidth coefficient
    horizontal: str = "geometric"  # "geometric" | "quadratic"
    reject: bool = True          # STA/LTA anti-trigger window rejection
    sta: float = 1.0             # short-term average (s)
    lta: float = 30.0            # long-term average (s)
    min_ratio: float = 0.30      # keep windows with STA/LTA in [min, max]
    max_ratio: float = 2.5
    min_windows: int = 4         # fewer kept windows -> station flagged

    def out_freqs(self):
        return np.geomspace(self.fmin, self.fmax, self.nfreq)


@dataclass
class HVResult:
    """Processed H/V of one station."""
    freqs: np.ndarray            # (nfreq,) output frequencies
    hv: np.ndarray               # (nfreq,) log-mean H/V curve
    sigma: np.ndarray            # (nfreq,) std of log H/V over windows
    f0: float                    # mean window peak frequency (Hz)
    f0_sigma: float              # std of window peaks (Hz)
    a0: float                    # H/V amplitude at the mean-curve peak
    n_win: int                   # windows kept
    n_rej: int                   # windows rejected by the anti-trigger
    duration: float              # record length (s)
    sesame: dict = field(default_factory=dict)   # criteria -> bool
    hv_windows: np.ndarray | None = None         # (n_win, nfreq), optional

    @property
    def reliable(self):
        s = self.sesame
        return bool(s.get("c1_cycles") and s.get("c2_windows") and s.get("c3_scatter"))

    @property
    def clear_peak(self):
        s = self.sesame
        keys = [k for k in s if k.startswith("p")]
        return sum(bool(s[k]) for k in keys) >= 5


# ------------------------------------------------------------------ helpers


def konno_ohmachi_matrix(f_in, f_out, b=40.0):
    """Row-normalised Konno-Ohmachi smoothing matrix W (nout, nin):
    smoothed(f_out) = W @ raw(f_in)."""
    f_in = np.asarray(f_in, float)
    f_out = np.asarray(f_out, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = b * np.log10(np.maximum(f_in[None, :], 1e-12) /
                         np.maximum(f_out[:, None], 1e-12))
        w = (np.sin(x) / x) ** 4
    w[np.abs(x) < 1e-7] = 1.0
    w[:, f_in <= 0] = 0.0
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    return w


def _cos_taper(n, frac):
    w = np.ones(n)
    m = max(1, int(frac * n))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w


def _running_mean(x, n):
    n = max(1, int(n))
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.empty_like(x)
    lo = np.maximum(np.arange(len(x)) - n + 1, 0)
    hi = np.arange(len(x)) + 1
    out = (c[hi] - c[lo]) / (hi - lo)
    return out


def sta_lta(data3, dt, sta_s, lta_s):
    """STA/LTA characteristic function of the 3-component envelope."""
    env = np.mean(np.abs(np.asarray(data3, float)), axis=0)
    sta = _running_mean(env, round(sta_s / dt))
    lta = _running_mean(env, round(lta_s / dt))
    return sta / np.maximum(lta, 1e-30)


def _detrend(x):
    n = len(x)
    t = np.arange(n, dtype=float)
    p = np.polyfit(t, x, 1)
    return x - (p[0] * t + p[1])


def _refine_peak(f, a, i):
    """Parabolic peak refinement in log-frequency."""
    if i <= 0 or i >= len(f) - 1:
        return float(f[i]), float(a[i])
    lf = np.log(f[i - 1:i + 2])
    y = np.log(np.maximum(a[i - 1:i + 2], 1e-12))
    d = (y[0] - 2 * y[1] + y[2])
    if d >= -1e-12:
        return float(f[i]), float(a[i])
    x0 = np.clip(0.5 * (y[0] - y[2]) / d, -1.0, 1.0)
    return float(np.exp(lf[1] + x0 * (lf[2] - lf[1]))), float(a[i])


# -------------------------------------------------------------- main chain


def split_windows(nt, dt, cfg: ProcConfig):
    """Start indices and length of the analysis windows."""
    n = int(round(cfg.win_len / dt))
    if n < 16 or n > nt:
        n = min(nt, max(16, n))
    step = max(1, int(n * (1.0 - cfg.overlap)))
    starts = list(range(0, nt - n + 1, step))
    return starts, n


def reject_windows(data3, dt, starts, n, cfg: ProcConfig):
    """Anti-trigger: keep windows whose STA/LTA stays inside the band."""
    if not cfg.reject:
        return [True] * len(starts), None
    cf = sta_lta(data3, dt, cfg.sta, cfg.lta)
    skip = int(round(cfg.lta / dt))            # LTA warm-up is meaningless
    keep = []
    for s in starts:
        seg = cf[max(s, skip):s + n]
        if len(seg) == 0:
            keep.append(True)
            continue
        keep.append(bool(seg.max() <= cfg.max_ratio and seg.min() >= cfg.min_ratio))
    if not any(keep):                          # never reject everything
        keep = [True] * len(starts)
    return keep, cf


def window_hv(data3, dt, s, n, W, band_idx, cfg: ProcConfig):
    """H/V curve of one window on the output frequency grid."""
    taper = _cos_taper(n, cfg.taper)
    spec = []
    for c in range(3):
        x = np.asarray(data3[c][s:s + n], float)
        if cfg.detrend:
            x = _detrend(x)
        A = np.abs(np.fft.rfft(x * taper))[band_idx]
        spec.append(W @ A)                     # Konno-Ohmachi smoothed
    nn, ee, zz = spec[0], spec[1], spec[2]
    if cfg.horizontal == "quadratic":
        h = np.sqrt(0.5 * (nn ** 2 + ee ** 2))
    else:
        h = np.sqrt(np.maximum(nn * ee, 1e-30))
    return h / np.maximum(zz, 1e-30)


def process_record(data3, dt, cfg: ProcConfig | None = None,
                   keep_windows=False):
    """Full chain: 3-component record (3, nt) -> :class:`HVResult`."""
    cfg = cfg or ProcConfig()
    data3 = np.asarray(data3, float)
    nt = data3.shape[-1]
    freqs = cfg.out_freqs()

    starts, n = split_windows(nt, dt, cfg)
    keep, _ = reject_windows(data3, dt, starts, n, cfg)
    n_rej = int(len(keep) - sum(keep))

    f_fft = np.fft.rfftfreq(n, dt)
    band_idx = np.where((f_fft >= cfg.fmin / 1.6) & (f_fft <= cfg.fmax * 1.4))[0]
    W = konno_ohmachi_matrix(f_fft[band_idx], freqs, cfg.ko_b)

    curves, peaks = [], []
    for s, ok in zip(starts, keep):
        if not ok:
            continue
        hv = window_hv(data3, dt, s, n, W, band_idx, cfg)
        curves.append(hv)
        fpk, _ = _refine_peak(freqs, hv, int(np.argmax(hv)))
        peaks.append(fpk)
    curves = np.asarray(curves)
    logc = np.log(np.maximum(curves, 1e-12))
    hv_mean = np.exp(logc.mean(axis=0))
    sigma = logc.std(axis=0)
    f0_mean = float(np.exp(np.mean(np.log(peaks))))
    f0_sigma = float(np.std(peaks))
    ipk = int(np.argmax(hv_mean))
    f0_curve, a0 = _refine_peak(freqs, hv_mean, ipk)

    res = HVResult(freqs=freqs, hv=hv_mean, sigma=sigma, f0=f0_mean,
                   f0_sigma=f0_sigma, a0=float(a0), n_win=len(curves),
                   n_rej=n_rej, duration=nt * dt,
                   hv_windows=curves if keep_windows else None)
    res.sesame = sesame_criteria(res, cfg)
    return res


# ------------------------------------------------------------------ SESAME


def _sesame_thresholds(f0):
    """(epsilon, log-theta) window-scatter thresholds vs f0 (SESAME 2004)."""
    edges = [0.2, 0.5, 1.0, 2.0]
    eps = [0.25, 0.20, 0.15, 0.10, 0.05]           # sigma_f0 < eps * f0
    logtheta = [0.48, 0.40, 0.30, 0.25, 0.20]      # sigma_logA(f0) < log(theta)
    i = int(np.searchsorted(edges, f0))
    return eps[i] * f0, logtheta[i]


def sesame_criteria(res: HVResult, cfg: ProcConfig):
    """SESAME (2004) checks: 3 curve-reliability criteria (c*) and the 6
    clear-peak criteria (p*, need >= 5).  All computed on the log-mean curve
    and the window scatter."""
    f, hv, sig = res.freqs, res.hv, res.sigma
    f0, a0 = res.f0, res.a0
    lw = cfg.win_len
    out = {}
    # --- reliability
    out["c1_cycles"] = bool(f0 > 10.0 / lw)
    out["c2_windows"] = bool(lw * res.n_win * f0 > 200.0)
    band = (f > 0.5 * f0) & (f < 2.0 * f0)
    lim = np.log(2.0) if f0 > 0.5 else np.log(3.0)
    out["c3_scatter"] = bool(band.any() and float(sig[band].max()) < lim)
    # --- clear peak
    left = (f >= f0 / 4.0) & (f < f0)
    right = (f > f0) & (f <= 4.0 * f0)
    out["p1_trough_lo"] = bool(left.any() and hv[left].min() < a0 / 2.0)
    out["p2_trough_hi"] = bool(right.any() and hv[right].min() < a0 / 2.0)
    out["p3_amp"] = bool(a0 > 2.0)
    ipk = int(np.argmax(hv))
    f_lo = f[int(np.argmax(hv * np.exp(-sig)))]
    f_hi = f[int(np.argmax(hv * np.exp(+sig)))]
    out["p4_stable"] = bool(abs(f_lo - f[ipk]) <= 0.05 * f0 and
                            abs(f_hi - f[ipk]) <= 0.05 * f0)
    eps, ltheta = _sesame_thresholds(f0)
    out["p5_f0_scatter"] = bool(res.f0_sigma < eps)
    i0 = int(np.argmin(np.abs(f - f0)))
    out["p6_amp_scatter"] = bool(float(sig[i0]) < ltheta)
    return out


def interp_hv(freqs_out, freqs_in, hv_in, sigma_in=None):
    """Log-log interpolate an H/V curve (and optionally its sigma) onto the
    inversion frequency grid, clamping outside the measured band."""
    lo = np.log(np.asarray(freqs_out, float))
    li = np.log(np.asarray(freqs_in, float))
    hv = np.exp(np.interp(lo, li, np.log(np.maximum(hv_in, 1e-12))))
    if sigma_in is None:
        return hv
    return hv, np.interp(lo, li, np.asarray(sigma_in, float))
