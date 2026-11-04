"""
sensitivity forecast from the measured noise:
        chi^2 = sum_rows a_i^2 / var_i
(as Regis+2025, but with the pipeline's per-row variance instead of the pre-launch table)
uses the same rows and footprint as the two-sided runs.

a_i = F * D_i * R(m_a; pixel), where R depends on the row only through its detector pixel, so one
pass stores W[pixel] = sum_rows D_i^2 / var_i and any mass follows afterwards from
sum_pixels F^2 R^2 W. NO WAVELENGTH IS BINNED.

    sbatch scripts/measured_forecast.sbatch   # task i handles chunks[i::n]; resumable
    srun -n 1 -c 8 --mem=96G uv run python -u scripts/measured_forecast.py --combine
the combine step writes data/measured-forecast.npz: A per mass at the current data volume.
    A scales with time, so the forecast at T months is g(A * T / t_now_months)
"""

import json
import os
import sys
import time
from pathlib import Path

import healpy
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import two_sided as ts
from talltable.constants import HP_HIGH_LEVEL
from talltable.paths import PIXEL_DB_PATH

BASE = Path("/mnt/sdceph/users/chainje/spax")
OUT = BASE / "measured-forecast-2026-11-04"
RUNS = {
    "ndf": BASE / "two-sided-2026-10-26-ndf",
    "sdf": BASE / "two-sided-2026-10-26-sdf",
}
N_CHUNKS = 160  # per field
N_PIXELS = 6 * 2040 * 2040
N_MONTHS = 17
MONTH_DAYS = 30.4375


def config(field):
    return json.loads((RUNS[field] / "config.json").read_text())


def load_images():
    t = pq.read_table(
        PIXEL_DB_PATH.parent / "image.parquet", columns=["imageid", "t_beg"]
    )
    ids, t_beg = t["imageid"].to_numpy(), t["t_beg"].to_numpy()
    o = np.argsort(ids)
    return ids[o], t_beg[o], t_beg.min(), t_beg.max()


def add_partition(part, W, monthly, images):
    """add one partition's rows (the cuts of two_sided.read_partition) to W and to monthly, the
    same sum per (detector, month). returns the number of rows used"""
    f = pq.ParquetFile(
        PIXEL_DB_PATH / f"part={part}/compacted.parquet", pre_buffer=True
    )
    expr = pc.bit_wise_and(pc.field("flags"), ~ts.ALLOWED_BITS) == 0
    expr &= pc.field("flux") > 0
    expr &= pc.field("variance") > 0
    for c in ["flux", "variance", "zodi"]:
        expr &= pc.is_finite(pc.field(c))
    table = f.read(
        ["hphigh", "flags", "flux", "variance", "zodi", "waveid", "imageid"]
    ).filter(expr)
    hphigh, variance, waveid, imageid = (
        table[c].to_numpy() for c in ["hphigh", "variance", "waveid", "imageid"]
    )
    del table

    hpbin_u, inverse = np.unique(
        hphigh >> (2 * (HP_HIGH_LEVEL - ts.BIN_LEVEL)), return_inverse=True
    )
    lon, lat = healpy.pix2ang(2**ts.BIN_LEVEL, hpbin_u, nest=True, lonlat=True)
    D_bin = np.asarray(ts.DFAC(np.radians(lon), np.radians(lat)), dtype=np.float64)
    weight = D_bin[inverse] ** 2 / variance.astype(np.float64)

    pixel = ts.waveid_to_wave_index(waveid)
    W += np.bincount(pixel, weight, minlength=N_PIXELS)

    ids, t_beg, t0, _ = images
    month = ((t_beg[np.searchsorted(ids, imageid)] - t0) / MONTH_DAYS).astype(np.int64)
    det = pixel // (2040 * 2040)
    monthly += np.bincount(
        det * N_MONTHS + month, weight, minlength=6 * N_MONTHS
    ).reshape(6, N_MONTHS)
    return len(weight)


def forecast_A(W, windows):
    """A [GAMMA_UNIT^-2] per (window, mass), from W; same selection of rows as two_sided"""
    index = np.flatnonzero(W > 0)
    det, lam, curve = (
        index // (2040 * 2040) + 1,
        ts.WAVELENGTHS[index],
        ts.PIXEL_CURVE[index],
    )
    W = W[index]
    A = np.zeros((len(windows), ts.SUBS_PER_CHANNEL))
    for w, win in enumerate(windows):
        lam_c = win["center_um"]
        hw = ts.WINDOW_K * lam_c / ts.resolution(lam_c, det=win["det"])
        sel = np.flatnonzero((np.abs(lam - lam_c) <= hw) & (det == win["det"]))
        for s, lam_a in enumerate(win["masses_um"]):
            m_a = (4 * np.pi) / lam_a * 1e4 * ts.SPEED_OF_LIGHT_CGS * ts.HBAR_EV_S
            R = ts.pixel_response(m_a, ts.SIGMA_V, lam[sel], curve[sel])
            A[w, s] = np.sum((ts.INTENSITY_FACTOR * R) ** 2 * W[sel]) * ts.GAMMA_UNIT**2
    return A


def chunks():
    out = []
    for field in RUNS:
        parts = config(field)["partitions"]
        out += [(field, i, parts[i::N_CHUNKS]) for i in range(N_CHUNKS)]
    return out


def scan():
    task, n_tasks = (
        int(os.environ.get("SLURM_PROCID", 0)),
        int(os.environ.get("SLURM_NTASKS", 1)),
    )
    OUT.mkdir(exist_ok=True)
    images = load_images()
    for field, i, parts in chunks()[task::n_tasks]:
        path = OUT / f"chunk-{field}-{i:03d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        W, monthly, n_rows = np.zeros(N_PIXELS), np.zeros((6, N_MONTHS)), 0
        for part in parts:
            n_rows += add_partition(part, W, monthly, images)
        tmp = path.with_name(f"{path.stem}.tmp.{os.uname().nodename}.{os.getpid()}.npz")
        np.savez(tmp, W=W, monthly=monthly, n_rows=n_rows, n_parts=len(parts))
        os.replace(tmp, path)
        print(
            f"task {task}: {path.name}, {len(parts)} partitions, {n_rows} rows, {time.time() - t0:.0f} s",
            flush=True,
        )


def combine():
    images = load_images()
    t_now = (images[3] - images[2]) / MONTH_DAYS
    out = {"t_now_months": t_now}
    for field in RUNS:
        cfg = config(field)
        paths = [OUT / f"chunk-{field}-{i:03d}.npz" for i in range(N_CHUNKS)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} chunks missing, e.g. {missing[0]}")
        W, monthly, n_rows, n_parts = np.zeros(N_PIXELS), np.zeros((6, N_MONTHS)), 0, 0
        for p in paths:
            z = np.load(p)
            W += z["W"]
            monthly += z["monthly"]
            n_rows += int(z["n_rows"])
            n_parts += int(z["n_parts"])
        if n_parts != len(cfg["partitions"]):
            raise RuntimeError(
                f"{field}: {n_parts} partitions in chunks, {len(cfg['partitions'])} in the run"
            )
        np.save(OUT / f"W-{field}.npy", W)

        # per mass, summing over the detectors of a window, as the analysis does
        A_win = forecast_A(W, cfg["windows"])
        masses = {}
        for w, win in enumerate(cfg["windows"]):
            for s, lam_a in enumerate(win["masses_um"]):
                k = (win["lam_idx"], s)
                masses[k] = (lam_a, masses.get(k, (0, 0.0))[1] + A_win[w, s])
        keys = sorted(masses)
        out["lam_idx"], out["sub"] = np.array(keys).T
        out["lam_um"] = np.array([masses[k][0] for k in keys])
        out[f"A_{field}"] = np.array([masses[k][1] for k in keys])
        out[f"monthly_{field}"] = monthly
        out[f"n_rows_{field}"] = n_rows
        print(f"{field}: {n_parts} partitions, {n_rows} rows", flush=True)
    np.savez(Path(__file__).parents[1] / "data/measured-forecast.npz", **out)


if __name__ == "__main__":
    combine() if "--combine" in sys.argv else scan()
