# Method

Numerical details of BasinInv3D and the reasoning behind each choice.

## 1. Forward problem: 3D elastic wave propagation

**Equations.** Isotropic linear elastodynamics in velocity–stress form:
9 fields (vx, vy, vz, σxx, σyy, σzz, σxy, σxz, σyz) on a standard staggered
grid (Virieux 1986; Levander 1988): normal stresses at cell centers,
velocities and shear stresses at the usual half-offset positions.

**Discretization.** 4th-order centered differences in space
(coefficients 9/8, −1/24), 2nd-order leapfrog in time. Time step from the
3D 4th-order CFL bound, `dt = 0.45 dx / vp_max`. Everything is float32
NumPy; a 66³-class model steps in tens of milliseconds on a laptop core.

**Free surface.** z = 0 is a stress-imaging free surface (Graves 1996):
σzz = 0 on the surface plane, antisymmetric images for σxz/σyz across it,
and on the surface plane the σxx/σyy updates use the analytic substitution
∂vz/∂z = −λ/(λ+2μ) (∂vx/∂x + ∂vy/∂y). z-derivatives drop to 2nd order in
the two planes nearest the surface. The free surface is essential — it
carries the surface waves and basin resonance that the inversion feeds on.

**Absorbing edges.** Cerjan exponential sponges (15 cells) on the four
sides and the bottom. Simple and robust; CPML is the planned upgrade for
weaker edge reflections.

**Sources / receivers.** Vertical point force with a Ricker wavelet
(delay 1.2/f0) one cell below the surface; receivers record 3-component
velocity on the surface plane. Grid resolution is chosen to keep
≥ 5–6 points per minimum shear wavelength `vs_sed / (f_max)`.

**Material blending.** The sediment/bedrock contrast is blended with a
sigmoid over ~one grid cell in depth. This is deliberate: it makes the
waveforms *continuously differentiable with respect to the interface
depth*, which finite-difference gradients require. A sharp staircase
interface makes the misfit piecewise-constant in the parameters and the
gradient useless.

## 2. Model parameterization

The unknown is the interface depth map `z_b(x, y)` plus the sediment shear
velocity `vs_sed` (vp and density follow fixed scalings; bedrock is assumed
known). The depth map is represented by an `n×n` grid of control-node
depths (default 3×3) spanning the interior of the domain,
bicubic-interpolated (RectBivariateSpline) to the full surface, clipped to
≥ 0 and cosine-tapered to zero near the edges so sediments never reach the
absorbing zones.

Why nodes and not voxels: with the interface + one velocity, ~10 parameters
capture "can we recover the geometry?" directly, keep the FD-gradient cost
practical, and sidestep the massive null space of voxel FWI at these
frequencies.

## 3. Inverse problem

**Misfit.**

    Φ(m) = ½ Σ (d_syn(m) − d_obs)² / Σ d_obs²  +  R(m)

summed over shots × receivers × 3 components × time. The normalization
makes Φ ≈ 0.5 for "no signal predicted" and 0 for a perfect fit.

**Regularization — the checkerboard lesson.** The first benchmark run used
only a weak second-difference (curvature) penalty on the node grid. The
optimizer happily drove the waveform misfit down while the node depths went
to an alternating deep/shallow checkerboard, and the *geometric* error got
worse. Surface waveforms at these frequencies cannot discriminate against
node-scale oscillation — that pattern lives in the data null space, so the
regularization has to remove it. The fix:

    R(m) = w [ Σ (Δ¹ nodes)² + ½ Σ (Δ² nodes)² ] / (nz·dx)²,   w = 10⁻²

First differences are the term that penalizes checkerboards (they are huge
for alternating patterns, moderate for genuinely deep smooth basins);
curvature alone does not separate the two. The weight is a bias–variance
dial: 10⁻² suppressed the artifact and let the benchmark recover the basin
smoothly, at the cost of some depth deficit at the basin center — which the
multiscale scheme below removes.

**Multiscale refinement (`MultiscaleInversion`).** A single coarse node grid
leaves the basin centre too shallow (regularization/resolution trade-off);
a single fine grid re-introduces the checkerboard null space. The fix is
coarse-to-fine: solve on a schedule of node grids (e.g. 2×2 → 3×3 → 4×4 →
5×5), **warm-starting** each scale from the previous solution
(`resample_params`: the coarse interpolated depth map re-sampled at the finer
nodes) and **relaxing the smoothing weight geometrically** as resolution grows
(strong at coarse scales to lock the basin shape, weak at fine scales to add
detail). Each scale is an independent L-BFGS-B solve; the live hooks report
the active parameterization so callers need not track the schedule.

**Gradients.** Forward finite differences: one full multi-shot simulation
per parameter (steps: dx/2 in depth, 10 m/s in vs), all n+1 evaluations
running concurrently on a persistent process pool whose workers receive the
static problem data (survey, parameterization, observed records) once at
startup. For ~10 parameters this costs about 11 forward problems per
gradient — practical at demo grid sizes and completely general. The
adjoint-state method (one forward + one reverse run per shot, independent
of parameter count) is the upgrade path for dense parameterizations.

**Optimizer.** SciPy L-BFGS-B with bounds (depths in [0, 0.8·Lz], vs in
[200, 900] m/s), typically 8–15 iterations.

**Live hooks.** The solver exposes a per-time-step callback and the
inversion exposes per-forward and per-evaluation callbacks; the web studio
uses them to stream wavefield frames, the current basin, and the misfit
curve while the run is in progress.

## 4. Ambient-noise generation

`basininv/noise.py` scatters tens of random force sources (random position,
component, and Ricker-burst time series in a 0.5–8 Hz band) over the
surface, runs one long simulation, and computes smoothed H/V spectral
ratios per station from the 3-component records. Verified behavior: strong
low-frequency H/V amplification over deep sediments, with the peak moving
to higher frequency as sediments thin toward the basin edge. Records of
30–60 s are recommended to resolve sub-0.5 Hz resonances.

## 5. Microtremor (HVSR) inversion — `basininv/hvsr.py`

The field-data-oriented path (Microtremor Studio, `webapp_mt/`): recover a
**3D multi-layer sediment Vs structure** from the **H/V spectral ratios** of a
scattered station set.

**Forward.** Under each station the earth is a 1-D layered column. Its HVSR is
modelled as the ratio of the SH to the P vertical-incidence surface
amplification of a damped layered medium, each computed by the Kramer (1996)
propagator recursion with complex (hysteretic-Q) velocities:

    HVSR(f) = A_SH(f; Vs profile) / A_P(f; Vp profile)

The fundamental reproduces `f0 = Vs/4H` (validated to ~1–2 % for a single
layer). This is a standard, inexpensive *transfer-function HVSR proxy* — one
forward is a few complex-matrix recursions over frequency, i.e. milliseconds —
so finite-difference gradients over a dense parameterization are cheap.
Rayleigh-wave ellipticity / diffuse-field HVSR is the physics upgrade path.

**Parameterization (`MultiLayerBasin`).** Sediment layers are described by
per-layer **thickness** control-node grids (bicubic-interpolated, non-negative)
plus per-layer Vs. Parameterizing thickness rather than absolute depth
guarantees the interfaces never cross. **Any parameter can be fixed** via a
free/fixed mask — a known bedrock interface (borehole), a fixed Vs jump, a
known layer depth — pinned into the parameter vector before optimization.

**Misfit.**

    Φ(m) = w_d·½⟨(log H_syn − log H_obs)²⟩  +  w_p·⟨(log f0_syn − log f0_obs)²⟩  +  R(m)

over stations × frequency. The log-curve term alone is non-unique and
noise-sensitive; the **peak-frequency term** — a differentiable soft-argmax
`f0` estimate (weighted mean with a high exponent, so it tracks the resonance,
not a band centroid) — is what makes the objective's minimum sit at the truth
and avoids HVSR cycle-skipping. It is weighted above the curve term
(w_p ≈ 4, w_d ≈ 0.4). `R` is a first-difference roughness penalty on each
thickness node grid. The flat **initial guess** is chosen so its fundamental
matches the *median observed peak* — the other half of avoiding cycle-skipping.

**Optimizer.** SciPy L-BFGS-B on the free parameters with **variable scaling**
(thicknesses are tens–hundreds of metres, Vs hundreds of m/s; without dividing
each variable by a characteristic scale the line search fails immediately).

**Two hard limits, by design.** (1) *Band-clipping* — HVSR resolves only depths
whose fundamental stays inside the measurable band (~0.3–10 Hz); once `f0`
drops below the band, the peak-finder locks onto the first overtone and a deep
basin masquerades as a shallow one. The synthetic basins are therefore capped
at ≈140 m so `f0` stays in band; this is physics, not a code limit. (2)
*Multi-layer non-uniqueness* — with several free layers of different Vs, many
thickness splits give the same `f0`; recovering the individual layers needs the
curve shape, known Vs, or a fixed bedrock (hence the fix-parameters feature).

**Uncertainty ensemble.** Optionally re-run the inversion across members that
vary the noise realization, starting model, smoothing strength **and node
resolution**, and report a per-cell bedrock-depth ±σ map with a ±2σ coverage
check. Varying node resolution is essential: in well-constrained cases the
residual is *representation bias* (a coarse grid cannot fit the sharp centre),
not variance, so a noise-only ensemble is badly over-confident (σ ≈ 1 m vs
≈10 m error). Members run in a process pool.

## 6. Field assistant: real recordings → basin structure

The field-facing layer on top of §5 (`basininv/hvproc.py`,
`basininv/fieldio.py`, Field-assistant mode of `webapp_mt/`): instead of
synthesising observations internally, the studio reads a **campaign folder**
of recordings and inverts whatever it finds.

**Campaign format (`fieldio`).** `stations.csv` (id, x, y in local metric
coordinates) plus one file per station: a raw 3-component record (`.npz` with
`data (3, nt)` as [N, E, Z] and `dt`; `.csv` with a `# dt=` header; miniSEED/
SAC via obspy when installed) *or* an already-processed `.hv` curve
(`freq hv [sigma]`). Kinds mix freely — recycled curves from earlier surveys
sit next to new recordings. `make_demo_campaign` writes an *imaginary*
campaign in exactly this format (records synthesised from a hidden layered
basin, §5's `synth_microtremor`), so the identical code path runs with or
without real data; its `truth.npz` is read only to score validation runs.

**Processing chain (`hvproc`).** The standard microtremor reduction, fully
parameterised: overlapping cosine-tapered windows (default 40 s / 50 %);
STA/LTA **anti-trigger** rejection of transient-contaminated windows;
per-window FFT + **Konno–Ohmachi** smoothing (b = 40) onto a log frequency
grid; geometric (or quadratic) horizontal merge; H/V per window; log-mean
curve with a per-frequency multiplicative σ; f₀ per window by parabolic
refinement in log f, giving f₀ ± σ(f₀). The **SESAME (2004)** criteria are
evaluated per station: 3 curve-reliability conditions (enough cycles, enough
windows, bounded scatter) and the 6 clear-peak conditions (troughs on both
sides, amplitude > 2, peak stability under ±σ, f₀ and amplitude scatter below
the frequency-dependent thresholds). Failing stations are flagged in the QC
panel and can be excluded from the inversion.

**Inversion adaptation.** The inversion grid is built from the station
bounding box (44×44 cells + 7-cell margin; depth extent auto-estimated from
the lowest measured f₀ via H ≈ Vs/4f₀, overridable). Curves are log-log
interpolated onto the inversion band and passed to §5's `HVSRInversion`
unchanged; convergence is reported as the **median |log f₀ residual|**, since
no truth exists. The uncertainty ensemble replaces synthetic noise
re-realisations with perturbations of each curve **within its measured window
scatter** (per-station, per-frequency σ from processing), on top of the
varied start model, smoothing and node resolution of §5.

**Depth constraints from other geophysics.** Any point with an externally
known or interpreted interface depth — borehole logs, resistivity soundings,
GPR picks, mapped outcrops — adds a term to the misfit:

    C(m) = w_c · ⟨ w_i · ((z_Li(x_i, y_i; m) − d_i) / max(d_i, 10))² ⟩

over constraint points i, where `z_L` is the modelled depth of the target
interface (any sediment interface or the bedrock) at the point.  The relative
form makes a 5 m miss at a 20 m borehole count like a 25 m miss at 100 m.
Constraints enter the ensemble members too.  How exactly a constraint is
honoured is limited by the thickness-node resolution — a coarse 3×3 grid
cannot bend to a single borehole on a steep flank (residuals of tens of
metres there are representation, not weighting); more nodes and weaker
smoothing tighten it.  Constraints are strong medicine both ways: a wrong
depth (validated deliberately with a fake 5 m "outcrop" over the basin
centre) visibly degrades the whole model, which is why the dashboard shows
per-constraint residuals and lets each point be toggled.

**Field Dashboard (`webapp_field/`).** The map-oriented front end on top of
all of the above: datasets are uploaded from the browser into a persistent
project workspace; stations and auxiliary points live on an OSM map (WGS84
lat/lon ↔ local metres via an equirectangular projection around the survey
anchor — sub-metre over survey extents); per-point configuration and manual
enrichment (attributes, fixed depths, exclusions, new points) feed directly
into `_collect_inputs`; the recovered depth / ±σ rasters are rendered as
georeferenced RGBA overlays with live updates each optimizer evaluation.
The final model is persisted (`results.json`) and rebuildable, powering an
in-browser 3-D basin view (stacked interfaces + stations, live during the
run), map-drawn Vs cross-sections, click-anywhere 1-D layer columns, and a
`lat,lon,depth` grid export for GIS.

**Verified end-to-end** (demo campaign: 25 stations, 328 s records at
dt = 0.01 written to disk, 3 layers, Vs fixed): records → H/V (curves match
the theoretical transfer-function H/V to ~5 % median in-band; all stations
pass SESAME) → inversion recovers the hidden bedrock with RMS ≈ 15 m,
correlation 0.95–0.96, final median f₀ residual ≈ 5 %.

## 7. Validation summary

- **Solver smoke test**: stability (no growth over the run), free-surface
  sanity, and a 98% relative record difference between basin and halfspace
  models (strong signal content).
- **Elastic FWI benchmark** (50×50×30, two-bump basin 519 m deep, 4 shots,
  36 receivers, 10 unknowns, flat 101 m / vs=550 start): misfit
  0.52 → 0.021, vs 395 m/s recovered vs 400 true, RMS depth error
  120 → 91 m, smooth artifact-free geometry. The residual central depth
  deficit of a single 3×3 grid is what the multiscale schedule (§3) targets.
- **HVSR forward**: single-layer HVSR peak within ~1–2 % of `f0 = Vs/4H`
  across a range of thicknesses.
- **HVSR inversion benchmark** (3-layer basin, bedrock ≤ 140 m, 25 stations,
  Vs fixed at truth, 5 % HVSR noise): bedrock-depth RMS 32 → 13 m, depth
  correlation 0.97; layer Vs recovered within a few % when left free.
- **Uncertainty ensemble** (same case): resolution-varying members give a
  bedrock σ ≈ 7 m with the truth inside ±2σ over ~50–68 % of the area, versus
  a falsely-confident σ ≈ 1 m for a noise-only ensemble.
- **Field-assistant end-to-end** (demo campaign read from disk, §6): raw
  records → SESAME-checked H/V → inversion; bedrock RMS ≈ 15 m,
  correlation 0.95, median f₀ residual ≈ 5 %.

## References

- Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
  velocity-stress finite-difference method. *Geophysics*, 51(4).
- Levander, A. R. (1988). Fourth-order finite-difference P-SV seismograms.
  *Geophysics*, 53(11).
- Graves, R. W. (1996). Simulating seismic wave propagation in 3D elastic
  media using staggered-grid finite differences. *BSSA*, 86(4).
- Cerjan, C. et al. (1985). A nonreflecting boundary condition for discrete
  acoustic and elastic wave equations. *Geophysics*, 50(4).
- Kramer, S. L. (1996). *Geotechnical Earthquake Engineering* — 1-D layered
  site-response transfer function (propagator recursion). Prentice Hall.
- SESAME (2004). Guidelines for the implementation of the H/V spectral ratio
  technique on ambient vibrations.
