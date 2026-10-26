"""
compute constaints using the two-sided statistic
"""

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import healpy
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from numpy.lib.stride_tricks import sliding_window_view
from talltable.constants import HP_HIGH_LEVEL
from talltable.partition import find_partitions_disc
from talltable.paths import PIXEL_DB_PATH, WAVES_DB_PATH
from talltable.waveid import rowcoldet_to_waveid, waveid_to_rowcoldet

from spax.d_factor import D_factor_factory


def read_halo(halo_name):
    with open(Path(__file__).parent.parent / "data/halos.csv", "r") as f:
        for row in csv.DictReader(f.readlines()):
            if row["name"] == halo_name:
                halo = {k.strip(): float(v) for k, v in row.items() if k != "name"}

                coord = SkyCoord(l=halo["l"], b=halo["b"], unit="deg", frame="galactic")
                coord = coord.transform_to("icrs")

                halo["ra"] = coord.ra.to("rad").value  # type: ignore
                halo["dec"] = coord.dec.to("rad").value  # type: ignore
                return halo

    raise RuntimeError(f"no halo found with name '{halo_name}'")


def load_wavelengths():
    f = pq.ParquetFile(WAVES_DB_PATH)
    table = f.read(["waveid", "wavelength", "bandwidth"])
    return (
        table["waveid"].to_numpy(),
        table["wavelength"].to_numpy(),
        table["bandwidth"].to_numpy(),
    )


def load_spec_channels():
    from astropy.io import fits

    with fits.open("data/spectral_channels_spx_cal-sch-v2-2026-229.fits") as f:
        return f[2].data


HBAR_EV_S = 6.582119569e-16  # https://en.wikipedia.org/wiki/Planck_constant
SPEED_OF_LIGHT_CGS = 2.99792458e10  # https://en.wikipedia.org/wiki/Speed_of_light
M_SUN_G = 1.988409870698051e33  # IAU 2015 nominal solar mass
KPC_CM = 3.0856775814913673e21  # IAU 2012/2015
MJY_CGS = 1e-17  # 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1 (definition)

# I_nu = Gamma/(4 pi) * c^2 * D * phi_nu: each decay releases the full rest energy m_a c^2.
# with D [M_sun kpc^-2] and phi_nu [Hz^-1], this factor gives I_nu [MJy sr^-1] per Gamma [s^-1]
INTENSITY_FACTOR = M_SUN_G / KPC_CM**2 * SPEED_OF_LIGHT_CGS**2 / (4 * np.pi) / MJY_CGS


def g_agg_from_gamma(gamma, m_a):
    """invert Gamma = g^2 m_a^3 / (64 pi) (natural units)

    Arguments:
        gamma: float/array [s^-1]
        m_a: float/array [eV]

    Return:
        g_agg: float/array [GeV^-1]
    """
    return 1e9 * np.sqrt(64 * np.pi * HBAR_EV_S * gamma / m_a**3)


HALO_NAME = "MW"
HALO = read_halo(HALO_NAME)
DFAC = D_factor_factory(
    0.0, 0.0, HALO["ra"], HALO["dec"], HALO["dist"], HALO["rho_s"], HALO["r_s"]
)
SIGMA_V = 160  # km/s
WAVEIDS, WAVELENGTHS, BANDWIDTHS = load_wavelengths()
SPEC_CHANNELS = load_spec_channels()

OUTLIER_BITFLAG = 1 << 19
KNOWN_SRC_BITFLAG = 1 << 21
ALLOWED_BITS = 0  # OUTLIER_BITFLAG | KNOWN_SRC_BITFLAG
BIN_LEVEL = 15
POLYNOMIAL_DEGREE = 4
WINDOW_K = 5

C1_WINDOW = 9
C1_DDOF = 1


def waveid_to_wave_index(wid):
    # det in [1, 6]; row, col in [0, 2039]
    # index in (sorted) WAVEIDs therefore given by
    row, col, det = waveid_to_rowcoldet(wid)
    return (det - 1) * 2040 * 2040 + row * 2040 + col


def wave_index_to_waveid(w_idx):
    col = w_idx % 2040
    row = (w_idx // 2040) % 2040
    det = w_idx // (2040 * 2040) + 1
    return rowcoldet_to_waveid(row, col, det)


assert np.array_equal(WAVEIDS, wave_index_to_waveid(np.arange(len(WAVEIDS))))


def resolution(lam_a, det=None):
    """returns the average resolution of spectral channels overlapping this wavelength
    (optionally filtered to a single detector)"""
    mask = SPEC_CHANNELS["WL_MIN"] < lam_a
    mask &= SPEC_CHANNELS["WL_MAX"] > lam_a
    if det is not None:
        mask &= SPEC_CHANNELS["DETECTOR"] == det
    if np.count_nonzero(mask) == 0:
        raise RuntimeError(
            f"wavelength {lam_a} matched no known channels"
            + (f" in det {det}" if det is not None else "")
        )
    return np.mean(SPEC_CHANNELS["R"][mask])


def read_partition(part_num):
    """read one partition into memory, keeping only the per-row quantities needed for
    every mass. rows are sorted by bin, so any subset of rows stays grouped by bin."""
    part_path = PIXEL_DB_PATH / f"part={part_num}/compacted.parquet"
    f = pq.ParquetFile(part_path, pre_buffer=True)
    columns = ["hphigh", "flux", "variance", "zodi", "waveid"]

    # restrict only to allowed bits, positive flux and variance, finite values
    # (comparisons already drop NaN and null, but not inf)
    expr = pc.bit_wise_and(pc.field("flags"), ~ALLOWED_BITS) == 0
    expr &= pc.field("flux") > 0
    expr &= pc.field("variance") > 0
    for c in ["flux", "variance", "zodi"]:
        expr &= pc.is_finite(pc.field(c))
    table = f.read(columns + ["flags"]).filter(expr).drop_columns("flags")
    hphigh, flux, variance, zodi, waveid = (table.column(c).to_numpy() for c in columns)
    del table

    # sort rows by bin
    hpbin = hphigh >> (2 * (HP_HIGH_LEVEL - BIN_LEVEL))
    order = np.argsort(hpbin, kind="stable")
    hpbin = hpbin[order]
    waveid = waveid[order]
    wave_index = waveid_to_wave_index(waveid)

    # one D-factor evaluation per bin
    # DFAC returns a float32 jax array; cast so everything downstream is float64 numpy
    hpbin_u, inverse = np.unique(hpbin, return_inverse=True)
    lon, lat = healpy.pix2ang(2**BIN_LEVEL, hpbin_u, nest=True, lonlat=True)
    D_bin = np.asarray(DFAC(np.radians(lon), np.radians(lat)), dtype=np.float64)

    return {
        "hpbin": hpbin,
        "D": D_bin[inverse],
        "d": flux[order].astype(np.float64) - zodi[order],
        "variance": variance[order].astype(np.float64),
        "wavelength": WAVELENGTHS[wave_index],
        "bandwidth": BANDWIDTHS[wave_index],
        "det": waveid_to_rowcoldet(waveid)[2],
    }


def process_partition_single_mass(rows, lam_a, det=None):
    """compute A, B in every bin of a partition (from `read_partition`) for one mass

    Return:
        dict of per-bin arrays, over the bins with at least one row in the window
            hpbin: bin index (nested, BIN_LEVEL)
            n_rows: number of rows in the window
            rank: numerical rank of the (weighted) continuum design matrix
            A, B: quadratic coefficients of the profiled chi^2
            chi2_0: chi^2 of the continuum-only fit, so chi^2(Gamma) = chi2_0 - 2 Gamma B
                + Gamma^2 A. its dof is n_rows - rank
    """
    m_a = (4 * np.pi) / lam_a * 1e4 * SPEED_OF_LIGHT_CGS * HBAR_EV_S  # eV
    window_hw = WINDOW_K * lam_a / resolution(lam_a, det=det)

    # select rows in the window (and detector); flatnonzero keeps them sorted by bin
    sel = np.abs(rows["wavelength"] - lam_a) <= window_hw
    if det is not None:
        sel &= rows["det"] == det
    sel = np.flatnonzero(sel)
    if len(sel) == 0:
        return {
            "hpbin": rows["hpbin"][:0],
            "n_rows": np.empty(0, dtype=np.int64),
            "rank": np.empty(0, dtype=np.int64),
            "A": np.empty(0),
            "B": np.empty(0),
            "chi2_0": np.empty(0),
        }

    hpbin = rows["hpbin"][sel]
    lam = rows["wavelength"][sel]
    d = rows["d"][sel]
    variance = rows["variance"][sel]
    transmission = pixel_response(m_a, SIGMA_V, lam, rows["bandwidth"][sel])
    a = INTENSITY_FACTOR * rows["D"][sel] * transmission  # MJy/sr per (1/s)

    # each bin is a contiguous slice [lo, hi)
    starts = np.flatnonzero(np.r_[True, hpbin[1:] != hpbin[:-1]])
    bounds = np.r_[starts, len(hpbin)]

    A = np.empty(len(starts))
    B = np.empty(len(starts))
    chi2_0 = np.empty(len(starts))
    rank = np.empty(len(starts), dtype=np.int64)
    for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        A[k], B[k], chi2_0[k], rank[k] = process_bin(
            lam_a, window_hw, a[lo:hi], d[lo:hi], variance[lo:hi], lam[lo:hi]
        )

    return {
        "hpbin": hpbin[starts],
        "n_rows": np.diff(bounds),
        "rank": rank,
        "A": A,
        "B": B,
        "chi2_0": chi2_0,
    }


def process_bin(lam_a, window_hw, a, d, variance, lam_j):
    x_j = (lam_j - lam_a) / window_hw

    M = np.stack([x_j**p for p in range(POLYNOMIAL_DEGREE + 1)], axis=-1)

    # very direct version (bad idea for numerics)
    # w = 1.0 / variance
    # F = M.T @ (w[:, None] * M)
    # cont = np.linalg.solve(F, M.T @ (w * a))
    # a_proj = a - M @ cont
    # A = a_proj.T @ (w * a_proj)
    # B = a_proj.T @ (w * d)

    # CORRECTION 1: fit continuum, then rescale variances to local scatter if necessary
    sw = np.sqrt(1.0 / variance)
    U, s, _ = np.linalg.svd(sw[:, None] * M, full_matrices=False)
    U = U[:, s > s[0] * 1e-10]  # numerical rank, not a physics cut
    d_t = sw * d
    r = (d_t - U @ (U.T @ d_t)) / sw  # residual of the continuum-only fit
    if len(r) >= C1_WINDOW:
        sort_idx = np.argsort(lam_j)
        sort_inv = np.argsort(sort_idx)
        rsig = np.zeros_like(r)
        hw = C1_WINDOW // 2
        rsig[hw:-hw] = sliding_window_view(r[sort_idx], C1_WINDOW).std(
            axis=1, ddof=C1_DDOF
        )
        rsig[:hw] = rsig[hw]  # assign edges the nearest full window's std
        rsig[-hw:] = rsig[-hw - 1]
        with np.errstate(divide="ignore"):
            sw = np.minimum(sw, 1.0 / rsig[sort_inv])

    # now do the fit and calculate the chi-square quadratic coefficients
    U, s, _ = np.linalg.svd(sw[:, None] * M, full_matrices=False)
    U = U[:, s > s[0] * 1e-10]  # numerical rank, not a physics cut
    a_t, d_t = sw * a, sw * d
    a_proj = a_t - U @ (U.T @ a_t)
    d_proj = d_t - U @ (U.T @ d_t)  # residual of the continuum-only fit
    A = a_proj @ a_proj
    B = a_proj @ d_t
    chi2_0 = d_proj @ d_proj

    return A, B, chi2_0, U.shape[1]


def gaussian(x, mu, sigma_sq):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma_sq) / np.sqrt(2 * np.pi * sigma_sq)


def pixel_response(m_a, sigma_v, wavelength, bandwidth):
    """
    Arguments:
        m_a: float [eV]
        sigma_v: float [km/s]
        wavelength: array [um], center of each pixel's response
        bandwidth: array [um], width of each pixel's response

    @TODO: use real transmission functions instead of Gaussian approx
    """
    nu_a = m_a / (4 * np.pi) / HBAR_EV_S  # Hz
    sigma_a = sigma_v * 1e5 / SPEED_OF_LIGHT_CGS * nu_a  # Hz

    c_um_Hz = SPEED_OF_LIGHT_CGS * 1e4
    sigma_j = c_um_Hz * bandwidth / np.sqrt(2 * np.pi) / wavelength**2  # Hz
    return gaussian(c_um_Hz / wavelength, nu_a, sigma_a**2 + sigma_j**2)


def process_partition(part_num, masses, out_path):
    """compute every mass in one partition and write the per-bin results to `out_path`
    (one row per bin, mass, and detector). returns the number of rows read."""
    rows = read_partition(part_num)
    n_rows = len(rows["hpbin"])

    tables = []
    for lam_idx, wl, det in masses:
        res = process_partition_single_mass(rows, wl, det)
        n = len(res["hpbin"])
        tables.append(
            pa.table(
                {
                    "lam_idx": np.full(n, lam_idx, dtype=np.int16),
                    "det": np.full(n, det, dtype=np.int8),
                    **res,
                }
            )
        )
    del rows

    # write to a temporary file, then rename: a file at out_path is always complete
    tmp_path = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    pq.write_table(pa.concat_tables(tables), tmp_path)
    os.replace(tmp_path, out_path)
    return n_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    # striding: task i of n handles partitions[i::n]. under `srun`, slurm sets these
    parser.add_argument(
        "--task-id", type=int, default=int(os.environ.get("SLURM_PROCID", "0"))
    )
    parser.add_argument(
        "--num-tasks", type=int, default=int(os.environ.get("SLURM_NTASKS", "1"))
    )
    args = parser.parse_args()
    assert 0 <= args.task_id < args.num_tasks

    # 1 degree north ecliptic cap (center of north deep field)
    radius_deg = 1.0
    nep = SkyCoord(lon=0, lat=+90, unit="deg", frame="geocentricmeanecliptic")
    nep = nep.transform_to("icrs")
    nep_ra = nep.ra.to("deg").value  # type: ignore
    nep_dec = nep.dec.to("deg").value  # type: ignore
    # talltable may return numpy ints, which json can't serialize
    partitions = sorted(
        int(p) for p in find_partitions_disc(nep_ra, nep_dec, radius_deg)
    )

    # every (channel wavelength, detector covering it) pair
    masses = []
    for i, wl in enumerate(SPEC_CHANNELS["WAVELENGTH"]):
        for det in range(1, 7):
            mask = SPEC_CHANNELS["DETECTOR"] == det
            wl_min = SPEC_CHANNELS["WL_MIN"][mask].min()
            wl_max = SPEC_CHANNELS["WL_MAX"][mask].max()
            if wl_min > wl or wl_max < wl:
                continue
            masses.append((i, float(wl), det))

    # everything the per-bin results depend on. a run directory holds one config only:
    # resuming with a different config would silently mix results
    config = {
        "center_ra_deg": nep_ra,
        "center_dec_deg": nep_dec,
        "radius_deg": radius_deg,
        "partitions": partitions,
        "halo_name": HALO_NAME,
        "halo": HALO,
        "sigma_v_kms": SIGMA_V,
        "allowed_bits": ALLOWED_BITS,
        "bin_level": BIN_LEVEL,
        "polynomial_degree": POLYNOMIAL_DEGREE,
        "window_k": WINDOW_K,
        "masses": [
            {"lam_idx": i, "wavelength_um": wl, "det": det} for i, wl, det in masses
        ],
        "pinetti_c1": {
            "window": C1_WINDOW,
            "ddof": C1_DDOF,
            "edges": "nearest_full_window",
        },
    }
    config = json.loads(json.dumps(config))  # normalize to what a reload gives back

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"{config_path} differs from this run's config")
    else:
        tmp_path = config_path.with_name(f"config.json.tmp{os.getpid()}")
        tmp_path.write_text(json.dumps(config, indent=2))
        os.replace(tmp_path, config_path)

    # checkpointing: skip partitions whose output already exists
    my_parts = partitions[args.task_id :: args.num_tasks]
    todo = [p for p in my_parts if not (args.out_dir / f"part={p}.parquet").exists()]
    print(
        f"task {args.task_id}/{args.num_tasks}: {len(my_parts)} partitions, "
        f"{len(my_parts) - len(todo)} already done",
        flush=True,
    )

    t_start = time.time()
    for k, part_num in enumerate(todo):
        t0 = time.time()
        out_path = args.out_dir / f"part={part_num}.parquet"
        n_rows = process_partition(part_num, masses, out_path)

        elapsed = time.time() - t_start
        eta = elapsed / (k + 1) * (len(todo) - k - 1)
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_rss_gb = max_rss / (
            2**30 if sys.platform == "darwin" else 2**20
        )  # B vs KiB
        print(
            f"part={part_num}: {n_rows} rows in {time.time() - t0:.0f} s, "
            f"peak memory {max_rss_gb:.1f} GB ({k + 1}/{len(todo)}, ETA {eta / 60:.0f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
