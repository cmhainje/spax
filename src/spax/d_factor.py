import jax
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from .sphgeo import angular_separation, offset_l_b

DFAC_DEGREE = 40
DFAC_x, DFAC_w = tuple(map(jnp.asarray, leggauss(DFAC_DEGREE)))
DFAC_tcut = 40  # corresponds to a cut at s_max/R (or |s_0/R|) < 1e17


def density_NFW(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


def D_factor(density_fn, psi, dist, s_max: float = 1e5):
    s_0 = dist * jnp.cos(psi)
    R = dist * jnp.sin(psi)
    R = jnp.maximum(R, 1e-10)

    t_lo = jnp.maximum(-DFAC_tcut, jnp.arcsinh((0.0 - s_0) / R))
    t_hi = jnp.minimum(+DFAC_tcut, jnp.arcsinh((s_max - s_0) / R))

    mid = 0.5 * (t_hi + t_lo)
    half = 0.5 * (t_hi - t_lo)

    t_i = half[..., None] * DFAC_x + mid[..., None]
    w_i = half[..., None] * DFAC_w
    r_i = R[..., None] * jnp.cosh(t_i)

    return jnp.sum(w_i * r_i * density_fn(r_i), axis=-1)


def D_factor_factory(
    l0,
    b0,
    l_halo,
    b_halo,
    dist,
    rho_s,
    r_s,
):
    """Makes a JAX jitted and vmapped wrapper of `D_factor` that automatically
    performs the coordinate conversions for arbitrary (l, b) into the frame
    centered at (l0, b0).

    Arguments:
        l0: float [rad]
            origin longitude of offset frame
        b0: float [rad]
            origin latitude of offset frame
        l_halo: float [rad]
            longitude of halo center
        b_halo: float [rad]
            latitude of halo center
        dist: float [kpc]
            distance (from Sun) to halo center
        rho_s: float [M_sun kpc^-3]
            NFW scale density of halo
        r_s: float [kpc]
            NFW scale radius of halo

    Return:
        function with the following signature
        Arguments:
            l: float/array [rad]
                longitude
            b: float/array [rad]
                latitude
        Return:
            D_factor: float/array [M_sun kpc^-2]
                halo column density along line of sight
    """
    def rho(r):
        return density_NFW(r, rho_s, r_s)

    dl_halo, db_halo = offset_l_b(l0, b0, l_halo, b_halo)

    def D_fac(dl, db):
        psi = angular_separation(dl, db, dl_halo, db_halo)
        return D_factor(rho, psi, dist)

    return jax.jit(jax.vmap(D_fac))
