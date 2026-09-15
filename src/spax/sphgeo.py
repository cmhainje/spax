import jax
import jax.numpy as jnp


def _clip(x):
    return jnp.clip(x, -1.0, +1.0)


def angular_separation(l1, b1, l2, b2):
    """Compute angular separation between two lon/lat coordinates.

    Arguments:
        l1: float/array [rad]
            longitude of point 1
        b1: float/array [rad]
            latitude of point 1
        l2: float/array [rad]
            longitude of point 2
        b2: float/array [rad]
            latitude of point 2

    Return:
        angular separation [rad]
    """
    arg = jnp.sin(b1) * jnp.sin(b2) + jnp.cos(b1) * jnp.cos(b2) * jnp.cos(l1 - l2)
    return jnp.arccos(_clip(arg))


def offset_l_b(l0, b0, l, b):
    """Return the lon/lat of point (l,b) in the frame of offsets about (l0,b0).

    Arguments:
        l0: float/array [rad]
            origin longitude of offset frame
        b0: float/array [rad]
            origin latitude of offset frame
        l: float/array [rad]
            longitude of target point
        b: float/array [rad]
            latitude of target point

    Return:
        dl: float/array [rad]
            offset longitude
        db: float/array [rad]
            offset latitude
    """
    cos_b = jnp.cos(b)
    sin_b = jnp.sin(b)
    cos_b0 = jnp.cos(b0)
    sin_b0 = jnp.sin(b0)
    cos_dl = jnp.cos(l - l0)
    sin_dl = jnp.sin(l - l0)

    l_off = jnp.atan2(
        _clip(cos_b * sin_dl),
        _clip(sin_b0 * sin_b + cos_b0 * cos_b * cos_dl),
    )
    b_off = jnp.arcsin(_clip(cos_b0 * sin_b - sin_b0 * cos_b * cos_dl))
    return l_off, b_off


def build_offset_grid(
    theta_max,
    spacing=3e-5,
    gnomonic=True,
    circle_cut=True,
):
    """Build a square grid (of `spacing`) of offset lon/lats.

    By default, performs the gnomonic tangent plane projection and clips the resulting
    grid to a circle.

    Arguments:
        theta_max:  float [rad]
            radius (or half side-length) of resulting grid
        spacing:    float [rad]
            grid spacing. defaults to 3e-5 (~6.18 arcsec)
        gnomonic:   bool
            whether to perform tangent plane projection. (raw offsets if not)
        circle_cut: bool
            whether to cut resulting grid down to circle. (square if not)

    Return:
        dl: float/array [rad]
            grid offsets in longitude
        db: float/array [rad]
            grid offsets in latitude
    """
    if gnomonic:
        n = int(jnp.ceil(jnp.tan(theta_max) / spacing))
    else:
        n = int(jnp.ceil(theta_max / spacing))

    edges = jnp.arange(-n, +n + 1) * spacing
    xi, eta = jnp.meshgrid(edges, edges, indexing="ij")

    if gnomonic:
        lon = jnp.arctan(xi)
        lat = jnp.arctan(eta * jnp.cos(lon))
    else:
        lon = xi
        lat = eta

    if circle_cut:
        psi = jnp.arccos(_clip(jnp.cos(lat) * jnp.cos(lon)))
        mask = psi <= theta_max
        lon = lon[mask]
        lat = lat[mask]

    return lon, lat
