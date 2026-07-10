"""Basin geometry and material-model construction.

Two parameterizations of the sediment/bedrock interface depth map z_b(x, y):

* `gaussian_basin`   — the "true" model: a sum of smooth anisotropic Gaussian
  depressions.  This is what generates the synthetic observed data.
* `BasinParameterization` — the inversion unknowns: depths at a coarse grid of
  control nodes (bicubic-interpolated to the full surface) plus the sediment
  shear velocity.  The inversion never sees the Gaussian description.

The interface is blended over ~one grid cell (sigmoid in depth) so the misfit
is smooth with respect to the depth parameters — essential for
finite-difference gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int
    dx: float           # cubic cells, meters

    @property
    def x(self):
        return np.arange(self.nx) * self.dx

    @property
    def y(self):
        return np.arange(self.ny) * self.dx

    @property
    def z(self):
        return np.arange(self.nz) * self.dx


@dataclass
class Materials:
    """Bedrock is fixed and assumed known; sediments are (partly) unknown."""
    vp_rock: float = 3200.0
    vs_rock: float = 1800.0
    rho_rock: float = 2400.0
    vs_sed: float = 400.0       # inversion unknown
    vpvs_sed: float = 2.2       # fixed scaling vp = vpvs * vs
    rho_sed: float = 1900.0

    @property
    def vp_sed(self):
        return self.vpvs_sed * self.vs_sed


def edge_taper(grid: GridSpec, margin_cells: int):
    """Cosine mask forcing the basin to zero depth near the domain edges,
    keeping sediments away from the absorbing zones."""
    def ramp(n):
        r = np.ones(n)
        m = margin_cells
        t = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        r[:m] = t
        r[-m:] = t[::-1]
        return r

    return ramp(grid.nx)[:, None] * ramp(grid.ny)[None, :]


def gaussian_basin(grid: GridSpec, bumps, margin_cells=20):
    """True basin depth map: sum of rotated anisotropic Gaussians.

    bumps: list of dicts with keys amp (m), x0, y0 (m), sx, sy (m), theta (rad).
    """
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    zb = np.zeros_like(X)
    for b in bumps:
        c, s = np.cos(b.get("theta", 0.0)), np.sin(b.get("theta", 0.0))
        xr = c * (X - b["x0"]) + s * (Y - b["y0"])
        yr = -s * (X - b["x0"]) + c * (Y - b["y0"])
        zb += b["amp"] * np.exp(-0.5 * ((xr / b["sx"]) ** 2 + (yr / b["sy"]) ** 2))
    return zb * edge_taper(grid, margin_cells)


class BasinParameterization:
    """Inversion parameter vector <-> (depth map, sediment vs).

    Parameters are [node_depths (ncx*ncy, meters), vs_sed (m/s)].
    Control nodes span the interior of the domain; the interpolated map is
    clipped to >= 0 and cosine-tapered to zero at the edges.
    """

    def __init__(self, grid: GridSpec, ncx=3, ncy=3, margin_cells=20,
                 invert_vs=True):
        self.grid = grid
        self.ncx, self.ncy = ncx, ncy
        self.invert_vs = invert_vs
        self.margin = margin_cells
        pad = margin_cells * grid.dx
        self.node_x = np.linspace(pad, grid.x[-1] - pad, ncx)
        self.node_y = np.linspace(pad, grid.y[-1] - pad, ncy)
        self._taper = edge_taper(grid, margin_cells)

    @property
    def n_params(self):
        return self.ncx * self.ncy + (1 if self.invert_vs else 0)

    def pack(self, node_depths, vs_sed):
        v = list(np.asarray(node_depths).ravel())
        if self.invert_vs:
            v.append(vs_sed)
        return np.array(v, float)

    def unpack(self, params):
        nd = np.asarray(params[: self.ncx * self.ncy]).reshape(self.ncx, self.ncy)
        vs = float(params[-1]) if self.invert_vs else None
        return nd, vs

    def depth_map(self, params):
        nd, _ = self.unpack(params)
        kx = min(3, self.ncx - 1)
        ky = min(3, self.ncy - 1)
        spl = RectBivariateSpline(self.node_x, self.node_y, nd, kx=kx, ky=ky)
        zb = spl(self.grid.x, self.grid.y)
        return np.clip(zb, 0.0, None) * self._taper

    def bounds(self, max_depth, vs_range=(200.0, 900.0)):
        b = [(0.0, max_depth)] * (self.ncx * self.ncy)
        if self.invert_vs:
            b.append(vs_range)
        return b


def build_model(grid: GridSpec, zb, mat: Materials, blend_cells=1.0):
    """(vp, vs, rho) volumes from a depth map, with a sigmoid-blended
    interface so the model varies smoothly with zb."""
    z = grid.z[None, None, :]
    h = blend_cells * grid.dx
    # weight -> 1 inside sediments (z < zb), 0 in bedrock
    w = 1.0 / (1.0 + np.exp(np.clip((z - zb[:, :, None]) / (0.25 * h), -40, 40)))
    w = w.astype(np.float32)
    vp = mat.vp_rock + w * (mat.vp_sed - mat.vp_rock)
    vs = mat.vs_rock + w * (mat.vs_sed - mat.vs_rock)
    rho = mat.rho_rock + w * (mat.rho_sed - mat.rho_rock)
    return vp.astype(np.float32), vs.astype(np.float32), rho.astype(np.float32)


def fit_nodes_to_map(param: BasinParameterization, zb):
    """Sample a full depth map at the control nodes (for building a reference
    'best achievable' representation of the true basin)."""
    ix = np.clip(np.round(param.node_x / param.grid.dx).astype(int), 0, param.grid.nx - 1)
    iy = np.clip(np.round(param.node_y / param.grid.dx).astype(int), 0, param.grid.ny - 1)
    return zb[np.ix_(ix, iy)]
