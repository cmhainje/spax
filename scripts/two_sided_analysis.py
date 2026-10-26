"""
analyze two_sided.py output
"""

import os

# popeye's login node caps threads per user (pids.max = 256): keep BLAS and duckdb small
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import functools
import hashlib
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from astropy.io import fits
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.special import ndtr

REFRESH = "--refresh" in sys.argv
duckdb.execute("SET threads = 8")

HBAR_EV_S = 6.582119569e-16  # https://en.wikipedia.org/wiki/Planck_constant
SPEED_OF_LIGHT_CGS = 2.99792458e10  # https://en.wikipedia.org/wiki/Speed_of_light

# usage: two_sided_analysis.py [run name, default two-sided-2026-10-23] [--refresh]
_run_name = next(
    (a for a in sys.argv[1:] if not a.startswith("--")), "two-sided-2026-10-23"
)
RUN_DIR = Path("/mnt/sdceph/users/chainje/spax") / _run_name
FIG_DIR = Path(__file__).parent.parent / "plots" / _run_name
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RUN_DIR / "cache"
parts = list(RUN_DIR.glob("*.parquet"))

with fits.open(
    "/mnt/sdceph/users/spherex/spherex_data_qr3/spectral_channels/cal-sch-v2-2026-229/spectral_channels_spx_cal-sch-v2-2026-229.fits"
) as f:
    WAVES = f[2].data["WAVELENGTH"]


def cached(fn):
    """cache a reduction to npz"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = inspect.getsource(fn) + repr(args) + repr(sorted(kwargs.items()))
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        path = CACHE_DIR / f"{fn.__name__}-{h}.npz"
        if path.exists() and not REFRESH:
            print(f"using cached result for {fn.__name__}")
            return dict(np.load(path))
        print(f"running {fn.__name__}")
        out = fn(*args, **kwargs)
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(path, **out)
        return out

    return wrapper


@cached
def per_lambda():
    return duckdb.sql(f"""
    SELECT
        sum(A) AS A,
        sum(B) AS B,
        lam_idx
    FROM read_parquet('{RUN_DIR}/*.parquet')
    GROUP BY lam_idx
    ORDER BY lam_idx ASC
    """).fetchnumpy()


def m_a_from_lam_a(lam_a):
    return (4 * np.pi) / lam_a * 1e4 * SPEED_OF_LIGHT_CGS * HBAR_EV_S  # eV


def g_agg_from_gamma(gamma, m_a):
    return 1e9 * np.sqrt(64 * np.pi * HBAR_EV_S * gamma / m_a**3)


def gamma_95(A, B, pcl=True):
    """95% upper limit on Gamma from Delta chi^2 = 3.84 relative to max(0, B/A)

    with pcl, power-constrained (Cowan+ 1105.3166) at the 16% threshold of Janish & Pinetti
    2310.15395 Eq. 14: the limit is monotonic in z = B / sqrt(A) ~ N(0, 1) under background
    only, so its 15.87% quantile is the limit at z = -1, i.e. (sqrt(1 + 3.84) - 1) / sqrt(A)
    """
    limit = (B + np.sqrt(3.84 * A + np.minimum(B, 0.0) ** 2)) / A
    if pcl:
        limit = np.maximum(limit, (np.sqrt(1 + 3.84) - 1) / np.sqrt(A))
    return limit


WAVE_TICKS = [0.7, 1, 2, 3, 4, 5]  # um
MASS_TICKS = [0.5, 1, 2, 3]  # eV


def set_ticks(axis, ticks):
    """label only `ticks`, as plain numbers (minor ticks stay, unlabeled)"""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axis.set_minor_formatter(NullFormatter())


def mass_axis(ax):
    """log m_a x-axis with wavelength on top (lambda = 4 pi hbar c / m_a is its own inverse)"""
    ax.set_xscale("log")
    ax.set_xlabel(r"$m_a$ [eV]")

    def convert(x):
        with np.errstate(divide="ignore"):  # matplotlib probes x = 0
            return m_a_from_lam_a(x)

    top = ax.secondary_xaxis("top", functions=(convert, convert))
    top.set_xlabel(r"$\lambda$ [um]")
    set_ticks(ax.xaxis, MASS_TICKS)
    set_ticks(top.xaxis, WAVE_TICKS)


result = per_lambda()
_A = result["A"]
_B = result["B"]
_lam = WAVES[result["lam_idx"]]

pull_z = _B / np.sqrt(_A)

plt.plot(_lam, np.abs(pull_z))
plt.xlabel(r"$\lambda$ [um]")
plt.ylabel(r"$|z|$")
plt.xscale("log")
plt.yscale("log")
set_ticks(plt.gca().xaxis, WAVE_TICKS)
plt.savefig(FIG_DIR / "pull.png")
plt.close()


@cached
def pull_stats():
    return duckdb.sql(f"""
    SELECT
        lam_idx,
        det,
        median(B / sqrt(A)) AS median_z,
        1.4826 * mad(B / sqrt(A)) AS robust_width,
        countif(abs(B / sqrt(A)) > 3.0) / count() AS frac_g3,
        median(sqrt(chi2_0 / (n_rows - rank))) FILTER (WHERE n_rows > rank) AS median_s_chi2,
    FROM read_parquet('{RUN_DIR}/*.parquet')
    WHERE A > 0  -- z is 0/0 = NaN otherwise, which duckdb sorts above everything
    GROUP BY lam_idx, det
    ORDER BY lam_idx ASC
    """).fetchnumpy()


result = pull_stats()

fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharex=True)

for d in np.unique(result["det"]):
    mask = result["det"] == d
    kw = dict(color=f"C{d - 1}", label=f"D{d}")
    axs[0].plot(WAVES[result["lam_idx"][mask]], result["median_z"][mask], **kw)
    axs[1].plot(WAVES[result["lam_idx"][mask]], result["robust_width"][mask], **kw)
    axs[2].plot(WAVES[result["lam_idx"][mask]], result["frac_g3"][mask], **kw)
    axs[1].plot(
        WAVES[result["lam_idx"][mask]],
        result["median_s_chi2"][mask],
        ls="--",
        color=f"C{d - 1}",
    )
    axs[3].plot(
        WAVES[result["lam_idx"][mask]],
        result["robust_width"][mask] / result["median_s_chi2"][mask],
        **kw,
    )

# expectation for N(0, 1) pulls and chi^2-distributed chi2_0
axs[0].axhline(0, color="k", lw=0.5)
axs[1].axhline(1, color="k", lw=0.5)
axs[2].axhline(0.0027, color="k", lw=0.5)
axs[3].axhline(1, color="k", lw=0.5)
axs[0].set_yscale("symlog", linthresh=1)
axs[1].set_yscale("log")
axs[2].set_yscale("log")
axs[3].set_yscale("log")
axs[0].legend(fontsize="small", ncols=2)

for ax in axs.flat:
    ax.set_xlabel(r"$\lambda$ [um]")
    ax.set_xscale("log")
    set_ticks(ax.xaxis, WAVE_TICKS)

axs[0].set_ylabel(r"median $z$")
axs[1].set_ylabel(r"width of $z$ (solid), $\sqrt{\chi^2_0 / \nu}$ (dashed)")
axs[2].set_ylabel(r"frac $|z| > 3$")
axs[3].set_ylabel(r"width of $z$ / $\sqrt{\chi^2_0 / \nu}$")

fig.tight_layout()
fig.savefig(FIG_DIR / "pull-stats.png")
plt.close()


@cached
def pooled_hist(lo=-10.0, hi=10.0, width=0.1):
    """per-detector histogram of z over all lam, bins. bin -1 / n_bins are under/overflow"""
    n_bins = round((hi - lo) / width)
    return duckdb.sql(f"""
    SELECT
        det,
        greatest(least(floor((B / sqrt(A) - {lo}) / {width}), {n_bins}), -1)::INTEGER AS bin,
        count() AS n,
    FROM read_parquet('{RUN_DIR}/*.parquet')
    WHERE A > 0
    GROUP BY det, bin
    ORDER BY det, bin
    """).fetchnumpy()


@cached
def pooled_quantiles(z_max=4.5, n=181):
    """per-detector quantiles of z at the N(0, 1) quantiles z_ref"""
    z_ref = np.linspace(-z_max, z_max, n)
    probs = ", ".join(repr(float(p)) for p in ndtr(z_ref))
    out = duckdb.sql(f"""
    WITH t AS (
        SELECT det, quantile_cont(B / sqrt(A), [{probs}]) AS q
        FROM read_parquet('{RUN_DIR}/*.parquet')
        WHERE A > 0
        GROUP BY det
    )
    SELECT det, unnest(range(len(q))) AS i, unnest(q) AS q FROM t ORDER BY det, i
    """).fetchnumpy()
    out["z_ref"] = z_ref[out["i"]]
    return out


lo, hi, width = -10.0, 10.0, 0.1
n_bins = round((hi - lo) / width)
edges = np.linspace(lo, hi, n_bins + 1)
centers = 0.5 * (edges[1:] + edges[:-1])

hist = pooled_hist(lo, hi, width)
quant = pooled_quantiles()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for d in np.unique(hist["det"]):
    mask = hist["det"] == d
    counts = np.zeros(n_bins + 2)  # [underflow, bins..., overflow]
    counts[hist["bin"][mask] + 1] = hist["n"][mask]
    total = counts.sum()
    frac_out = (counts[0] + counts[-1]) / total
    axs[0].stairs(
        counts[1:-1] / (total * width),
        edges,
        color=f"C{d - 1}",
        label=f"D{d} ({100 * frac_out:.1f}% outside)",
    )

    mask = quant["det"] == d
    axs[1].plot(
        quant["z_ref"][mask], quant["q"][mask], color=f"C{d - 1}", label=f"D{d}"
    )

axs[0].plot(
    centers,
    np.exp(-0.5 * centers**2) / np.sqrt(2 * np.pi),
    "k--",
    lw=1,
    label="N(0, 1)",
)
axs[0].set_yscale("log")
axs[0].set_ylim(1e-6, 1)
axs[0].set_xlabel(r"$z = B / \sqrt{A}$")
axs[0].set_ylabel("density")
axs[0].legend(fontsize="small")

z_ref = np.unique(quant["z_ref"])
axs[1].plot(z_ref, z_ref, "k--", lw=1)
axs[1].set_yscale("symlog", linthresh=3)
axs[1].set_xlabel("N(0, 1) quantile")
axs[1].set_ylabel(r"quantile of $z$")
axs[1].legend(fontsize="small")

fig.tight_layout()
fig.savefig(FIG_DIR / "pull-pooled.png")
plt.close()


# Pinetti App. C corrections II and III. her off positions / neighboring masses are spaced in
# units of the DM line width, which at SPHEREx resolution (R ~ 40) would span far more than a
# detector; we use the other channels of the same detector with 1 <= |l - j| <= K instead
MIN_NEIGHBORS = 5  # fewer than this -> S = 1 (her rule for III; applied to II too)


@cached
def correction_2(K):
    """per-bin MAD rescaling (Correction II), summed over bins per (lam_idx, det)

    for each bin i and channel j: S_ij = 1.4826 * MAD of z_il over the same bin and
    detector, 1 <= |l - j| <= K; A_ij, B_ij are scaled by min(1, 1 / S_ij^2).
    bins never span partitions, so this runs one partition file at a time (all at once
    gets OOM-killed on the login node), several files in parallel
    """

    def one_part(part):
        con = duckdb.connect(config={"threads": 2})  # login node caps threads per user
        return con.sql(f"""
        WITH t AS (
            SELECT hpbin, det, lam_idx, A, B, B / sqrt(A) AS z
            FROM read_parquet('{part}')
            WHERE A > 0
        ),
        w AS (
            SELECT
                *,
                count(z) OVER nb AS n_nb,
                1.4826 * mad(z) OVER nb AS S,
            FROM t
            WINDOW nb AS (
                PARTITION BY hpbin, det ORDER BY lam_idx
                RANGE BETWEEN {K} PRECEDING AND {K} FOLLOWING EXCLUDE CURRENT ROW
            )
        ),
        f AS (
            SELECT *, CASE WHEN n_nb >= {MIN_NEIGHBORS} AND S > 1 THEN 1 / S^2 ELSE 1 END AS f
            FROM w
        )
        SELECT
            lam_idx,
            det,
            sum(A) AS A_raw,
            sum(B) AS B_raw,
            sum(f * A) AS A,
            sum(f * B) AS B,
            countif(f < 1) AS n_rescaled,
            count() AS n,
        FROM f
        GROUP BY lam_idx, det
        """).to_arrow_table()

    files = sorted(RUN_DIR.glob("*.parquet"))
    partial = []
    with ThreadPoolExecutor(8) as pool:
        for k, table in enumerate(pool.map(one_part, files)):
            partial.append(table)
            if (k + 1) % 25 == 0 or k + 1 == len(files):
                print(f"  correction_2(K={K}): {k + 1}/{len(files)} files", flush=True)
    partial = pa.concat_tables(partial)
    return duckdb.sql("""
    SELECT
        lam_idx,
        det,
        sum(A_raw) AS A_raw,
        sum(B_raw) AS B_raw,
        sum(A) AS A,
        sum(B) AS B,
        sum(n_rescaled) / sum(n) AS frac_rescaled,
    FROM partial
    GROUP BY lam_idx, det
    ORDER BY det, lam_idx
    """).fetchnumpy()


def correction_3(lam_idx, det, A, B, K):
    """joint MAD rescaling (Correction III) per (lam_idx, det)

    S_j = max(1, 1.4826 * MAD of joint pulls z_l over the same detector, 1 <= |l - j| <= K),
    or 1 with fewer than MIN_NEIGHBORS finite neighbors; A_j, B_j are scaled by 1 / S_j^2
    """
    z = B / np.sqrt(A)
    S = np.ones_like(A)
    for k in range(len(A)):
        dist = np.abs(lam_idx - lam_idx[k])
        nb = z[(det == det[k]) & (dist >= 1) & (dist <= K) & np.isfinite(z)]
        if len(nb) >= MIN_NEIGHBORS:
            S[k] = max(1.0, 1.4826 * np.median(np.abs(nb - np.median(nb))))
    return A / S**2, B / S**2, S


def combine_detectors(lam_idx, A, B):
    """sum (lam_idx, det) entries over detectors; returns lam_idx, A, B sorted by lam_idx"""
    u, inv = np.unique(lam_idx, return_inverse=True)
    return u, np.bincount(inv, A), np.bincount(inv, B)


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for K, ls in [(5, "-"), (10, "--")]:
    c2 = correction_2(K)
    A3, B3, S3 = correction_3(c2["lam_idx"], c2["det"], c2["A"], c2["B"], K)

    curves = {
        "raw": (c2["A_raw"], c2["B_raw"]),
        "II": (c2["A"], c2["B"]),
        "II+III": (A3, B3),
    }
    for color, (label, (A, B)) in zip(["C7", "C0", "C3"], curves.items()):
        if label == "raw" and K != 5:
            continue  # independent of K
        li, A, B = combine_detectors(c2["lam_idx"], A, B)
        lam = WAVES[li]
        name = label if label == "raw" else f"{label}, K={K}"
        axs[0].plot(lam, np.abs(B / np.sqrt(A)), ls=ls, color=color, label=name)
        m_a = m_a_from_lam_a(lam)
        axs[1].plot(
            m_a, g_agg_from_gamma(gamma_95(A, B), m_a), ls=ls, color=color, label=name
        )

    for d in np.unique(c2["det"]):
        mask = c2["det"] == d
        lam = WAVES[c2["lam_idx"][mask]]
        axs[2].plot(
            lam, S3[mask], ls=ls, color=f"C{d - 1}", label=f"D{d}" if K == 5 else None
        )
        axs[2].plot(
            lam,
            c2["A"][mask] / c2["A_raw"][mask],
            ls=ls,
            color=f"C{d - 1}",
            alpha=0.4,
            lw=0.8,
        )

axs[0].set_ylabel(r"joint $|z|$")
axs[1].set_ylabel(r"$g_{a\gamma\gamma}$ 95% upper limit (PCL) [GeV$^{-1}$]")
axs[2].set_ylabel(r"$S_\mathrm{III}$ (dark), $A_\mathrm{II} / A_\mathrm{raw}$ (light)")
axs[0].legend(fontsize="small")
axs[2].legend(
    fontsize="small", ncols=2, title="solid K=5, dashed K=10", title_fontsize="small"
)
for ax in axs:
    ax.set_yscale("log")
for ax in axs[[0, 2]]:
    ax.set_xlabel(r"$\lambda$ [um]")
    ax.set_xscale("log")
    set_ticks(ax.xaxis, WAVE_TICKS)
mass_axis(axs[1])

fig.tight_layout()
fig.savefig(FIG_DIR / "corrections.png")
plt.close()

# headline limit: Corrections II + III with K = 5, power-constrained
K = 5
c2 = correction_2(K)
A3, B3, _ = correction_3(c2["lam_idx"], c2["det"], c2["A"], c2["B"], K)
li, A, B = combine_detectors(c2["lam_idx"], A3, B3)
m_a = m_a_from_lam_a(WAVES[li])

# Regis+2025-style MW forecasts (scripts/claude/regis_forecast.py) and our statistics-only sensitivity
forecast = np.load(Path(__file__).parent.parent / "data/regis-mw-forecast.npz")
_, A_raw, _ = combine_detectors(c2["lam_idx"], c2["A_raw"], c2["B_raw"])

fig, ax = plt.subplots(figsize=(8.5, 4))
ax.plot(
    m_a,
    g_agg_from_gamma(gamma_95(A, B, pcl=False), m_a),
    color="C7",
    lw=0.8,
    label="no PCL",
)
ax.plot(m_a, g_agg_from_gamma(gamma_95(A, B), m_a), color="C3", label="PCL (16%)")
ax.plot(
    m_a,
    g_agg_from_gamma(1.96 / np.sqrt(A_raw), m_a),
    color="C3",
    lw=0.8,
    ls=":",
    label="ours, statistics only",
)
ax.plot(
    forecast["m_a"],
    forecast["g_disc_1deg"],
    color="k",
    ls="--",
    lw=1,
    label="forecast, 1° disc",
)
ax.plot(
    forecast["m_a"],
    forecast["g_deep_field"],
    color="k",
    ls=":",
    lw=1,
    label="forecast, north DF (100 deg²)",
)
ax.set_yscale("log")
ax.set_ylabel(r"$g_{a\gamma\gamma}$ 95% upper limit [GeV$^{-1}$]")
ax.set_title(f"Corrections II + III (K = {K})", fontsize="medium")
ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.02, 0.5))
mass_axis(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "constraint.png")
plt.close()
