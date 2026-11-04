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
from scipy.interpolate import BSpline
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


SRF_DIR = Path(__file__).parent.parent / "data/external/spherex-srf-202607"
SRF_SUBCHANNELS = 68


def load_transmission_curves():
    """fiducial middle-pixel ("Cent") spectral response functions from the SSDC R7
    spectral-calibration delivery (SRF_DIR/SPHERExSpectralFiles_202607.pdf), one per
    (detector, sub-channel), and which curve applies to each pixel. per that document
    (sec. 3), the response of a pixel off the channel's central strip is the channel's
    curve shifted in wavelength.

    Return:
        curves: list of (wavelength [um], peak-normalized transmission, band center [um])
        pixel_curve: array over wave index, index into `curves`
    """
    from astropy.io import fits

    n = SRF_SUBCHANNELS
    curves = []
    for det in range(1, 7):
        for ch in range(1, n + 1):
            path = (
                SRF_DIR / f"MiddlePixel/SubChannel{n}/Band{det}"
                f"/FiducialSpec_Band{det}_SubCh{ch:02d}_{n}_Cent.csv"
            )
            lam, T = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
            T = np.maximum(T, 0.0)  # lab noise dips slightly negative (>= -1.2% of peak)
            # band center: where the cumulative integral reaches half (Hui et al. 2026
            # eq. 3), the same definition as the per-pixel CWAVE in WAVELENGTHS
            cum = np.r_[0.0, np.cumsum(0.5 * (T[1:] + T[:-1]) * np.diff(lam))]
            curves.append((lam, T, np.interp(0.5 * cum[-1], cum, lam)))

    # one geometric channel map for all detectors, in detector coordinates. these are
    # flipped in x relative to waveid (row, col) for detectors 4-6 (checked against the
    # delivery's band-center maps). excess regions (0 and n+1) get the nearest channel
    chan = fits.getdata(SRF_DIR / f"ChannelMaps/spectral_channels_{n}ch_2040.fits")
    chan = np.clip(chan.astype(np.int64), 1, n) - 1
    pixel_curve = np.stack(
        [(chan[:, ::-1] if det >= 4 else chan) + (det - 1) * n for det in range(1, 7)]
    )
    return curves, pixel_curve.ravel().astype(np.int16)


def load_sub_masses():
    """the masses: SUBS_PER_CHANNEL per spectral channel, at the band centers of the
    channel's sub-channels in the same SSDC delivery (FidSpecSummary, middle-pixel case)

    Return:
        array [n_channels, SUBS_PER_CHANNEL] of wavelengths [um], rows in SPEC_CHANNELS order
    """
    out = np.empty((len(SPEC_CHANNELS), SUBS_PER_CHANNEL))
    for i, (det, ch) in enumerate(zip(SPEC_CHANNELS["DETECTOR"], SPEC_CHANNELS["SUBCHAN"])):
        path = (
            SRF_DIR / f"FidSpecSummary/SubChannel{SRF_SUBCHANNELS}/Band{det}"
            f"/FidSpecSummary_Band{det}_SpecCh{SRF_SUBCHANNELS}.csv"
        )
        centers = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
        ch = int(ch)
        out[i] = centers[(ch - 1) * SUBS_PER_CHANNEL : ch * SUBS_PER_CHANNEL]
    return out


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
CURVES, PIXEL_CURVE = load_transmission_curves()
# masses: 4 per channel (the 68 sub-channels), sharing the channel's window
SUBS_PER_CHANNEL = SRF_SUBCHANNELS // 17
SUB_MASSES = load_sub_masses()

OUTLIER_BITFLAG = 1 << 19
KNOWN_SRC_BITFLAG = 1 << 21
ALLOWED_BITS = 0  # OUTLIER_BITFLAG | KNOWN_SRC_BITFLAG
BIN_LEVEL = 15
POLYNOMIAL_DEGREE = 4
WINDOW_K = 5
SPLINE_DEGREE = 3  # cubic
# the continuum model in each window: ("poly", degree) or ("spline", knot spacing in
# resolution elements lam / R). see `continuum_basis`
CONTINUUM = ("poly", POLYNOMIAL_DEGREE)

C1_WINDOW = 9
C1_DDOF = 1

# A, B are written as float32 in units of this Gamma (A ~ 1e50 s^2 otherwise overflows)
GAMMA_UNIT = 1e-25  # s^-1

# quadrature for the line-response overlap (pixel_response)
LINE_NODES = 161
SHIFT_STEPS_PER_SIGMA = 16


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
        "curve": PIXEL_CURVE[wave_index],
        "det": waveid_to_rowcoldet(waveid)[2],
    }


def process_partition_window(rows, lam_c, det, lams, continuum=CONTINUUM):
    """compute A, B in every bin of a partition (from `read_partition`) for several masses
    sharing one window: the continuum fit and Correction 1 depend only on the window's
    rows, so they are done once per bin for all masses

    Arguments:
        lam_c: window center [um], a spectral channel's center; half-width
            WINDOW_K * lam_c / R
        det: only rows from this detector
        lams: mass wavelengths [um]
        continuum: the continuum model, see `continuum_basis`

    Return:
        dict of per-bin arrays, over the bins with at least one row in the window
            hpbin: bin index (nested, BIN_LEVEL)
            n_rows: number of rows in the window
            rank: numerical rank of the (weighted) continuum design matrix
            chi2_0: chi^2 of the continuum-only fit, so chi^2(Gamma) = chi2_0 - 2 Gamma B
                + Gamma^2 A. its dof is n_rows - rank
            A, B: [n_bins, len(lams)], quadratic coefficients of the profiled chi^2
    """
    window_hw = WINDOW_K * lam_c / resolution(lam_c, det=det)

    # select rows in the window and detector; flatnonzero keeps them sorted by bin
    sel = (np.abs(rows["wavelength"] - lam_c) <= window_hw) & (rows["det"] == det)
    sel = np.flatnonzero(sel)

    hpbin = rows["hpbin"][sel]
    lam = rows["wavelength"][sel]
    d = rows["d"][sel]
    variance = rows["variance"][sel]
    a = np.stack(
        [
            INTENSITY_FACTOR
            * rows["D"][sel]
            * pixel_response(
                (4 * np.pi) / lam_a * 1e4 * SPEED_OF_LIGHT_CGS * HBAR_EV_S,  # eV
                SIGMA_V,
                lam,
                rows["curve"][sel],
            )
            for lam_a in lams
        ],
        axis=-1,
    )  # MJy/sr per (1/s)

    # each bin is a contiguous slice [lo, hi). a window can have no rows at all (e.g. a
    # partition at the field's edge that a detector doesn't cover): then there are no bins
    starts = np.zeros(0, dtype=np.int64)
    if len(hpbin):
        starts = np.flatnonzero(np.r_[True, hpbin[1:] != hpbin[:-1]])
    bounds = np.r_[starts, len(hpbin)]

    A = np.empty((len(starts), len(lams)))
    B = np.empty((len(starts), len(lams)))
    chi2_0 = np.empty(len(starts))
    rank = np.empty(len(starts), dtype=np.int64)
    for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        A[k], B[k], chi2_0[k], rank[k] = process_bin(
            lam_c, window_hw, a[lo:hi], d[lo:hi], variance[lo:hi], lam[lo:hi], continuum
        )

    return {
        "hpbin": hpbin[starts],
        "n_rows": np.diff(bounds),
        "rank": rank,
        "chi2_0": chi2_0,
        "A": A,
        "B": B,
    }


def process_partition_single_mass(rows, lam_a, det, continuum=CONTINUUM):
    """`process_partition_window` for one mass at the center of its own window"""
    res = process_partition_window(rows, lam_a, det, [lam_a], continuum)
    return {**res, "A": res["A"][:, 0], "B": res["B"][:, 0]}


def spline_knots(knot_spacing):
    """knot vector, in window coordinates x in [-1, 1], of a uniform cubic spline with
    knots as close to `knot_spacing` resolution elements apart as fits the window (whose
    full width is 2 WINDOW_K resolution elements) in a whole number of intervals"""
    n_intervals = max(1, round(2 * WINDOW_K / knot_spacing))
    return np.r_[
        np.full(SPLINE_DEGREE, -1.0),
        np.linspace(-1.0, 1.0, n_intervals + 1),
        np.full(SPLINE_DEGREE, 1.0),
    ]


def continuum_basis(x, continuum):
    """design matrix [len(x), n_params] of the continuum model at window coordinates
    x = (lam - lam_c) / window_hw in [-1, 1]

    Arguments:
        continuum: ("poly", degree): monomials x^p, p = 0..degree; or
            ("spline", knot spacing [resolution elements]): cubic B-splines on
            `spline_knots`. either way the continuum is linear in its parameters, so the
            profiling is unchanged
    """
    kind, param = continuum
    if kind == "poly":
        return np.stack([x**p for p in range(param + 1)], axis=-1)
    if kind == "spline":
        return BSpline.design_matrix(x, spline_knots(param), SPLINE_DEGREE).toarray()
    raise ValueError(f"unknown continuum model {continuum}")


def process_bin(lam_a, window_hw, a, d, variance, lam_j, continuum=CONTINUUM):
    x_j = (lam_j - lam_a) / window_hw

    M = continuum_basis(x_j, continuum)

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
    a_t, d_t = (sw * a.T).T, sw * d  # a: [n] or [n, n_masses]
    a_proj = a_t - U @ (U.T @ a_t)
    d_proj = d_t - U @ (U.T @ d_t)  # residual of the continuum-only fit
    A = np.sum(a_proj**2, axis=0)
    B = a_proj.T @ d_t
    chi2_0 = d_proj @ d_proj

    return A, B, chi2_0, U.shape[1]


def gaussian(x, mu, sigma_sq):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma_sq) / np.sqrt(2 * np.pi * sigma_sq)


def pixel_response(m_a, sigma_v, wavelength, curve):
    """overlap of each pixel's spectral response T_j with the line,
        int dlam T_j(lam) phi_nu(c / lam) / int dlam T_j(lam)
    i.e. the bandpass average of I_nu as SPHEREx defines it (Explanatory Supplement
    v2.0, eq. 22). phi_nu is the Gaussian line profile in frequency (unit integral over
    nu), and T_j(lam) = T_c(lam - wavelength_j + center_c) for the pixel's curve c.

    Arguments:
        m_a: float [eV]
        sigma_v: float [km/s]
        wavelength: array [um], band center (CWAVE) of each pixel's response
        curve: array, index into CURVES of each pixel's response

    Return:
        array [Hz^-1]
    """
    nu_a = m_a / (4 * np.pi) / HBAR_EV_S  # Hz
    sigma_a = sigma_v * 1e5 / SPEED_OF_LIGHT_CGS * nu_a  # Hz

    # the line in wavelength: phi_nu(c / lam) on a grid spanning +-8 of its widths
    c_um_Hz = SPEED_OF_LIGHT_CGS * 1e4
    lam_a = c_um_Hz / nu_a  # um
    sigma_lam = lam_a * sigma_a / nu_a  # um, to first order in sigma_v / c
    mu = lam_a + sigma_lam * np.linspace(-8, 8, LINE_NODES)
    w = np.full(LINE_NODES, mu[1] - mu[0])
    w[[0, -1]] /= 2  # trapezoid weights
    w *= gaussian(c_um_Hz / mu, nu_a, sigma_a**2)

    # for curve c, the overlap depends on the pixel only through its shift
    # x = wavelength_j - center_c. tabulate int dlam T_c(lam - x) phi_nu(c / lam) on a
    # grid in x much finer than the line, then interpolate in x for each pixel
    out = np.zeros(len(wavelength))
    for c in np.unique(curve):
        lam_c, T_c, center_c = CURVES[c]
        sel = curve == c
        x = wavelength[sel] - center_c
        n_x = int(np.ceil((x.max() - x.min()) / (sigma_lam / SHIFT_STEPS_PER_SIGMA))) + 2
        x_grid = np.linspace(x.min(), x.max(), n_x)
        T = np.interp(mu[None, :] - x_grid[:, None], lam_c, T_c, left=0.0, right=0.0)
        norm = np.sum(0.5 * (T_c[1:] + T_c[:-1]) * np.diff(lam_c))  # um
        out[sel] = np.interp(x, x_grid, T @ w) / norm
    return out


def process_partition(part_num, windows, out_path, windows_path, continuum=CONTINUUM):
    """compute every window (and its SUBS_PER_CHANNEL masses) in one partition. writes
    the per-bin results to `out_path` (one row per bin, mass, and detector; A, B in units
    of GAMMA_UNIT) and the per-bin continuum fits to `windows_path` (one row per bin,
    window, and detector). returns the number of rows read."""
    rows = read_partition(part_num)
    n_rows = len(rows["hpbin"])

    mass_tables, window_tables = [], []
    for lam_idx, wl, det in windows:
        res = process_partition_window(rows, wl, det, SUB_MASSES[lam_idx], continuum)
        n = len(res["hpbin"])
        window_tables.append(
            pa.table(
                {
                    "lam_idx": np.full(n, lam_idx, dtype=np.int16),
                    "det": np.full(n, det, dtype=np.int8),
                    "hpbin": res["hpbin"],
                    "n_rows": res["n_rows"].astype(np.int32),
                    "rank": res["rank"].astype(np.int8),
                    "chi2_0": res["chi2_0"].astype(np.float32),
                }
            )
        )
        for sub in range(SUBS_PER_CHANNEL):
            mass_tables.append(
                pa.table(
                    {
                        "lam_idx": np.full(n, lam_idx, dtype=np.int16),
                        "sub": np.full(n, sub, dtype=np.int8),
                        "det": np.full(n, det, dtype=np.int8),
                        "hpbin": res["hpbin"],
                        "A": (res["A"][:, sub] * GAMMA_UNIT**2).astype(np.float32),
                        "B": (res["B"][:, sub] * GAMMA_UNIT).astype(np.float32),
                    }
                )
            )
    del rows

    # write to temporary files, then rename: a file at out_path is always complete, and
    # its windows file was written before it
    for path, tables in [(windows_path, window_tables), (out_path, mass_tables)]:
        tmp_path = path.with_name(
            path.name + f".tmp.{os.uname().nodename}.{os.getpid()}"
        )
        pq.write_table(pa.concat_tables(tables), tmp_path)
        os.replace(tmp_path, path)
    return n_rows


def rounded(x, digits=12):
    """a json-like structure with every float rounded to `digits` significant digits"""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, digits) for v in x]
    return x


# deep field centers, (ecliptic lon, lat) in deg. south: Bock et al., "The SPHEREx
# Satellite Mission" (the arXiv versions have the sign of the longitude wrong)
DEEP_FIELDS = {
    "ndf": (0.0, +90.0),
    "sdf": (44.8, -82.0),
}


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
    # the disc to analyze: a deep field by name, or any center
    center = parser.add_mutually_exclusive_group(required=True)
    center.add_argument("--field", choices=DEEP_FIELDS)
    center.add_argument(
        "--center", type=float, nargs=2, metavar=("RA_DEG", "DEC_DEG")
    )
    # default: 100 deg^2, where the north deep field's depth drops off sharply
    parser.add_argument("--radius", type=float, default=5.64, metavar="DEG")
    # the continuum model (default: Pinetti's polynomial of degree POLYNOMIAL_DEGREE)
    cont = parser.add_mutually_exclusive_group()
    cont.add_argument("--poly-degree", type=int, default=POLYNOMIAL_DEGREE)
    cont.add_argument("--knot-spacing", type=float, metavar="RES_ELEMENTS")
    args = parser.parse_args()
    assert 0 <= args.task_id < args.num_tasks
    if args.knot_spacing is not None:
        continuum = ("spline", args.knot_spacing)
    else:
        continuum = ("poly", args.poly_degree)

    if args.field is not None:
        lon, lat = DEEP_FIELDS[args.field]
        coord = SkyCoord(lon=lon, lat=lat, unit="deg", frame="geocentricmeanecliptic")
        coord = coord.transform_to("icrs")
        center_ra = coord.ra.to("deg").value  # type: ignore
        center_dec = coord.dec.to("deg").value  # type: ignore
    else:
        center_ra, center_dec = args.center
    # talltable may return numpy ints, which json can't serialize
    partitions = sorted(
        int(p) for p in find_partitions_disc(center_ra, center_dec, args.radius)
    )
    print(
        f"field {args.field}: center (RA, Dec) = ({center_ra:.4f}, {center_dec:.4f}) deg, "
        f"radius {args.radius} deg, {len(partitions)} partitions",
        flush=True,
    )

    # windows: every (channel wavelength, detector covering it) pair. each holds the
    # channel's SUBS_PER_CHANNEL masses (SUB_MASSES[lam_idx])
    windows = []
    for i, wl in enumerate(SPEC_CHANNELS["WAVELENGTH"]):
        for det in range(1, 7):
            mask = SPEC_CHANNELS["DETECTOR"] == det
            wl_min = SPEC_CHANNELS["WL_MIN"][mask].min()
            wl_max = SPEC_CHANNELS["WL_MAX"][mask].max()
            if wl_min > wl or wl_max < wl:
                continue
            windows.append((i, float(wl), det))

    # everything the per-bin results depend on. a run directory holds one config only:
    # resuming with a different config would silently mix results
    config = {
        "field": args.field,
        "center_ra_deg": center_ra,
        "center_dec_deg": center_dec,
        "radius_deg": args.radius,
        "partitions": partitions,
        "halo_name": HALO_NAME,
        "halo": HALO,
        "sigma_v_kms": SIGMA_V,
        "allowed_bits": ALLOWED_BITS,
        "bin_level": BIN_LEVEL,
        "window_k": WINDOW_K,
        "windows": [
            {
                "lam_idx": i,
                "det": det,
                "center_um": wl,
                "masses_um": SUB_MASSES[i].tolist(),
            }
            for i, wl, det in windows
        ],
        "mass_grid": f"FidSpecSummary SubChannel{SRF_SUBCHANNELS} BandCenter_um (Cent)",
        "window": "shared by the masses of a channel: centered on the channel, "
        "half-width window_k * lambda / R",
        "gamma_unit_per_s": GAMMA_UNIT,
        "transmission": {
            "product": "SPHERExSpectralFiles_202607",
            "case": "MiddlePixel (Cent), shifted to per-pixel CWAVE",
            "subchannels": SRF_SUBCHANNELS,
            "overlap": "int dlam T phi_nu / int dlam T",
            "line_nodes": LINE_NODES,
            "shift_steps_per_sigma": SHIFT_STEPS_PER_SIGMA,
        },
        "pinetti_c1": {
            "window": C1_WINDOW,
            "ddof": C1_DDOF,
            "edges": "nearest_full_window",
        },
    }
    if continuum[0] == "poly":
        # the key runs before splines had, so they still resume
        config["polynomial_degree"] = continuum[1]
    else:
        knots = spline_knots(continuum[1])
        config["continuum"] = {
            "basis": f"B-spline of degree {SPLINE_DEGREE}, uniform knots in the window",
            "knot_spacing_requested": continuum[1],
            "knot_spacing": float(knots[SPLINE_DEGREE + 1] - knots[SPLINE_DEGREE]) * WINDOW_K,
            "n_params": len(knots) - SPLINE_DEGREE - 1,
        }
    config = json.loads(json.dumps(config))  # normalize to what a reload gives back

    (args.out_dir / "windows").mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        # to 12 digits: tasks on different nodes (CPUs) may differ in the last bit of the
        # coordinate transforms
        if rounded(json.loads(config_path.read_text())) != rounded(config):
            raise RuntimeError(f"{config_path} differs from this run's config")
    else:
        # unique across nodes: tasks on different nodes can share a pid
        tmp_path = config_path.with_name(
            f"config.json.tmp.{os.uname().nodename}.{os.getpid()}"
        )
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
        n_rows = process_partition(
            part_num,
            windows,
            out_path,
            args.out_dir / f"windows/part={part_num}.parquet",
            continuum,
        )

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
