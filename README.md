# BasinInv3D

3D elastic waveform inversion of sediment-filled basin geometry — a synthetic
proof-of-concept for going beyond 1D HVSR inversion in sedimentary basins.

## Idea

1. **Forward**: build an imaginary 3D sediment-filled valley (soft sediments
   embedded in bedrock, irregular interface) and simulate full elastic wave
   propagation, recording 3-component velocity on a surface receiver grid.
2. **Inverse**: pretending the basin is unknown, invert those surface records
   for the 3D bedrock–sediment interface `z_b(x, y)` *and* the sediment shear
   velocity, starting from a wrong flat guess.
3. **Score**: compare recovered vs. true geometry (RMS depth error, maps,
   3D surfaces).

## Components

| file | contents |
|---|---|
| `basininv/solver.py` | 3D isotropic elastic velocity–stress staggered-grid FD: 4th-order space, 2nd-order time, Graves stress-imaging free surface, Cerjan absorbing edges |
| `basininv/basin.py` | true basin (sum of Gaussians) and inversion parameterization (coarse control-node depth grid → bicubic surface, + sediment vs); sigmoid-blended interface so the misfit is smooth in the parameters |
| `basininv/survey.py` | shots (vertical-force Ricker at the surface), receiver grid, parallel multi-shot forward modeling |
| `basininv/inversion.py` | normalized least-squares waveform misfit + Tikhonov node smoothing, parallel finite-difference gradients, L-BFGS-B |
| `basininv/noise.py` | ambient-noise mode: distributed random sources, H/V spectral-ratio extraction (alternative data type, same inversion machinery) |
| `basininv/viz.py` | depth maps, 3D interfaces, waveform fits, convergence, live wavefield/inversion frames |
| `webapp/app.py` | **BasinInv3D Studio** — live web dashboard running the whole pipeline (stdlib http.server) |

## Run

### Web studio (recommended)

```bash
/usr/bin/python3 webapp/app.py            # then open http://127.0.0.1:8642
```

Pick a preset (fast ≈ 10–20 min, standard ≈ 1–2 h), press **Run experiment**,
and watch live: the random true basin, the elastic wavefield propagating and
reverberating in the valley (surface + cross-section view), then the inverted
basin / difference map / misfit curve updating after every gradient
evaluation, and a final report with score tiles. Stage flow, per-forward
progress, log console and Stop button included. All state is served by a
background pipeline thread; forward simulations for the FD gradient run in a
persistent process pool.

### Command line

```bash
/usr/bin/python3 scripts/smoke_test.py        # ~10 s sanity check
/usr/bin/python3 scripts/run_demo.py --quick  # small end-to-end inversion
/usr/bin/python3 scripts/run_demo.py          # fuller run (hours)
/usr/bin/python3 scripts/run_noise_demo.py    # ambient-noise H/V across the basin
```

Outputs (data + figures) land in `outputs/` (CLI) or `webapp/run/` (studio).

Use `/usr/bin/python3` (has NumPy/SciPy/matplotlib); the anaconda env lacks
matplotlib.

## Notes / upgrade path

- FD gradients cost one multi-shot forward per parameter; fine for ~10–30
  parameters. For dense parameterizations, implement the adjoint-state
  gradient in `solver.py`.
- Cerjan sponges are simple but imperfect absorbers; CPML is the upgrade.
- Real-data path: replace synthetic `d_obs` with field records, switch to the
  ambient-noise/HVSR misfit (`noise.py`), add source-wavelet estimation and
  time-window/frequency-band selection (multiscale continuation).
