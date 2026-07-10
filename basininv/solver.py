"""3D isotropic elastic wave solver.

Velocity-stress staggered-grid finite differences (Virieux/Levander style):
4th-order in space, 2nd-order leapfrog in time, Graves (1996) stress-imaging
free surface at z=0 (order reduced to 2nd in the two planes nearest the
surface), Cerjan exponential damping zones on the four sides and the bottom.

Grid convention: arrays have shape (nx, ny, nz); k=0 is the free surface and
k increases downward.  All nine fields are stored on same-shape arrays with
the usual implied half-cell offsets:

    vx  at (i+1/2, j,     k)        sxx,syy,szz at (i, j, k)
    vy  at (i,     j+1/2, k)        sxy at (i+1/2, j+1/2, k)
    vz  at (i,     j,     k+1/2)    sxz at (i+1/2, j,     k+1/2)
                                    syz at (i,     j+1/2, k+1/2)
"""
from __future__ import annotations

import numpy as np

C1 = np.float32(9.0 / 8.0)
C2 = np.float32(-1.0 / 24.0)

DTYPE = np.float32


def _df(a, axis):
    """4th-order forward staggered derivative (result at i+1/2), grid units."""
    out = np.zeros_like(a)
    n = a.shape[axis]

    def sl(lo, hi):
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(lo, hi if hi != 0 else None)
        return tuple(idx)

    out[sl(1, -2)] = (C1 * (a[sl(2, -1)] - a[sl(1, -2)])
                      + C2 * (a[sl(3, 0)] - a[sl(0, -3)]))
    # 2nd-order fill at edges (edges lie inside the damping zones)
    i0 = [slice(None)] * a.ndim
    i0[axis] = 0
    i1 = [slice(None)] * a.ndim
    i1[axis] = 1
    out[tuple(i0)] = a[tuple(i1)] - a[tuple(i0)]
    im2 = [slice(None)] * a.ndim
    im2[axis] = n - 2
    im1 = [slice(None)] * a.ndim
    im1[axis] = n - 1
    out[tuple(im2)] = a[tuple(im1)] - a[tuple(im2)]
    return out


def _db(a, axis):
    """4th-order backward staggered derivative (result at integer i)."""
    out = np.zeros_like(a)
    n = a.shape[axis]

    def sl(lo, hi):
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(lo, hi if hi != 0 else None)
        return tuple(idx)

    out[sl(2, -1)] = (C1 * (a[sl(2, -1)] - a[sl(1, -2)])
                      + C2 * (a[sl(3, 0)] - a[sl(0, -3)]))
    i1 = [slice(None)] * a.ndim
    i1[axis] = 1
    i0 = [slice(None)] * a.ndim
    i0[axis] = 0
    out[tuple(i1)] = a[tuple(i1)] - a[tuple(i0)]
    im1 = [slice(None)] * a.ndim
    im1[axis] = n - 1
    im2 = [slice(None)] * a.ndim
    im2[axis] = n - 2
    out[tuple(im1)] = a[tuple(im1)] - a[tuple(im2)]
    return out


class ElasticSolver3D:
    """One forward simulation on a fixed material model."""

    def __init__(self, vp, vs, rho, dx, cfl=0.45, damp_width=15, damp_alpha=0.10):
        vp = np.asarray(vp, DTYPE)
        vs = np.asarray(vs, DTYPE)
        rho = np.asarray(rho, DTYPE)
        self.nx, self.ny, self.nz = vp.shape
        self.dx = float(dx)
        self.dt = float(cfl * dx / vp.max())

        mu = rho * vs ** 2
        lam = rho * vp ** 2 - 2.0 * mu
        self.lam = lam
        self.mu = mu
        self.l2m = lam + 2.0 * mu

        # buoyancy averaged onto velocity points
        bx = np.empty_like(rho)
        bx[:-1] = 2.0 / (rho[:-1] + rho[1:])
        bx[-1] = 1.0 / rho[-1]
        by = np.empty_like(rho)
        by[:, :-1] = 2.0 / (rho[:, :-1] + rho[:, 1:])
        by[:, -1] = 1.0 / rho[:, -1]
        bz = np.empty_like(rho)
        bz[:, :, :-1] = 2.0 / (rho[:, :, :-1] + rho[:, :, 1:])
        bz[:, :, -1] = 1.0 / rho[:, :, -1]
        self.bx, self.by, self.bz = bx, by, bz

        # shear modulus averaged onto edge points (arithmetic; prototype-grade)
        def avg2(a, ax1, ax2):
            out = a.copy()
            s1 = [slice(None)] * 3
            s1[ax1] = slice(0, -1)
            s2 = [slice(None)] * 3
            s2[ax1] = slice(1, None)
            out[tuple(s1)] = 0.5 * (out[tuple(s1)] + out[tuple(s2)])
            s1 = [slice(None)] * 3
            s1[ax2] = slice(0, -1)
            s2 = [slice(None)] * 3
            s2[ax2] = slice(1, None)
            out[tuple(s1)] = 0.5 * (out[tuple(s1)] + out[tuple(s2)])
            return out

        self.mu_xy = avg2(mu, 0, 1)
        self.mu_xz = avg2(mu, 0, 2)
        self.mu_yz = avg2(mu, 1, 2)

        # ratio used by the free-surface condition on sxx/syy at k=0
        self.fs_ratio = (lam[:, :, 0] / self.l2m[:, :, 0]).astype(DTYPE)

        self._build_damping(damp_width, damp_alpha)
        self.reset()

    def _build_damping(self, w, alpha):
        """Cerjan taper: sides in x,y and bottom in z (free surface stays open)."""
        a = alpha / w
        damp = np.exp(-(a * (w - np.arange(w))) ** 2).astype(DTYPE)

        def profile(n, both):
            g = np.ones(n, DTYPE)
            if both:
                g[:w] = damp
            g[-w:] = damp[::-1]
            return g

        gx = profile(self.nx, both=True)
        gy = profile(self.ny, both=True)
        gz = profile(self.nz, both=False)   # only bottom
        self.damp = (gx[:, None, None] * gy[None, :, None] * gz[None, None, :]).astype(DTYPE)

    def reset(self):
        shp = (self.nx, self.ny, self.nz)
        self.vx = np.zeros(shp, DTYPE)
        self.vy = np.zeros(shp, DTYPE)
        self.vz = np.zeros(shp, DTYPE)
        self.sxx = np.zeros(shp, DTYPE)
        self.syy = np.zeros(shp, DTYPE)
        self.szz = np.zeros(shp, DTYPE)
        self.sxy = np.zeros(shp, DTYPE)
        self.sxz = np.zeros(shp, DTYPE)
        self.syz = np.zeros(shp, DTYPE)

    # ------------------------------------------------------------------ step
    def step(self):
        dtdx = DTYPE(self.dt / self.dx)
        vx, vy, vz = self.vx, self.vy, self.vz
        sxx, syy, szz = self.sxx, self.syy, self.szz
        sxy, sxz, syz = self.sxy, self.sxz, self.syz

        # ---- velocities -------------------------------------------------
        dsxz_dz = _db(sxz, 2)
        # free surface: sxz(-1/2) = -sxz(+1/2)  ->  2nd order at k=0,1
        dsxz_dz[:, :, 0] = 2.0 * sxz[:, :, 0]
        dsxz_dz[:, :, 1] = sxz[:, :, 1] - sxz[:, :, 0]
        vx += dtdx * self.bx * (_df(sxx, 0) + _db(sxy, 1) + dsxz_dz)

        dsyz_dz = _db(syz, 2)
        dsyz_dz[:, :, 0] = 2.0 * syz[:, :, 0]
        dsyz_dz[:, :, 1] = syz[:, :, 1] - syz[:, :, 0]
        vy += dtdx * self.by * (_db(sxy, 0) + _df(syy, 1) + dsyz_dz)

        dszz_dz = _df(szz, 2)
        dszz_dz[:, :, 0] = szz[:, :, 1] - szz[:, :, 0]   # szz[...,0] == 0
        vz += dtdx * self.bz * (_db(sxz, 0) + _db(syz, 1) + dszz_dz)

        # ---- stresses ----------------------------------------------------
        exx = _db(vx, 0)
        eyy = _db(vy, 1)
        ezz = _db(vz, 2)
        ezz[:, :, 1] = vz[:, :, 1] - vz[:, :, 0]         # 2nd order near surface
        ezz[:, :, 0] = -self.fs_ratio * (exx[:, :, 0] + eyy[:, :, 0])

        sxx += dtdx * (self.l2m * exx + self.lam * (eyy + ezz))
        syy += dtdx * (self.l2m * eyy + self.lam * (exx + ezz))
        szz += dtdx * (self.l2m * ezz + self.lam * (exx + eyy))
        szz[:, :, 0] = 0.0

        dvx_dz = _df(vx, 2)
        dvx_dz[:, :, 0] = vx[:, :, 1] - vx[:, :, 0]
        sxz += dtdx * self.mu_xz * (dvx_dz + _df(vz, 0))

        dvy_dz = _df(vy, 2)
        dvy_dz[:, :, 0] = vy[:, :, 1] - vy[:, :, 0]
        syz += dtdx * self.mu_yz * (dvy_dz + _df(vz, 1))

        sxy += dtdx * self.mu_xy * (_df(vx, 1) + _df(vy, 0))

        # ---- absorbing edges ---------------------------------------------
        d = self.damp
        for f in (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz):
            f *= d

    # ------------------------------------------------------------------ run
    def run(self, nt, sources=(), receivers=None, snapshot_every=0):
        """Time-march nt steps.

        sources   : list of (ix, iy, iz, component, wavelet[nt]) tuples;
                    component in {"fx","fy","fz"} (body force) or "explosion".
        receivers : integer array (nrec, 2) of surface (ix, iy) positions.
        Returns (seis, snaps): seis is (nrec, 3, nt) of (vx,vy,vz) at z=0.
        """
        self.reset()
        nrec = 0 if receivers is None else len(receivers)
        seis = np.zeros((nrec, 3, nt), DTYPE)
        snaps = []
        for it in range(nt):
            self.step()
            for (ix, iy, iz, comp, w) in sources:
                a = DTYPE(w[it] * self.dt)
                if comp == "fz":
                    self.vz[ix, iy, iz] += a * self.bz[ix, iy, iz]
                elif comp == "fx":
                    self.vx[ix, iy, iz] += a * self.bx[ix, iy, iz]
                elif comp == "fy":
                    self.vy[ix, iy, iz] += a * self.by[ix, iy, iz]
                else:  # explosion
                    self.sxx[ix, iy, iz] += a
                    self.syy[ix, iy, iz] += a
                    self.szz[ix, iy, iz] += a
            if nrec:
                rx, ry = receivers[:, 0], receivers[:, 1]
                seis[:, 0, it] = self.vx[rx, ry, 0]
                seis[:, 1, it] = self.vy[rx, ry, 0]
                seis[:, 2, it] = self.vz[rx, ry, 0]
            if snapshot_every and it % snapshot_every == 0:
                snaps.append(self.vz[:, :, 0].copy())
        return seis, snaps


def ricker(f0, nt, dt, t0=None):
    """Ricker wavelet sampled at the solver time step."""
    if t0 is None:
        t0 = 1.5 / f0
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(DTYPE)
