"""
analyze two_sided.py output: one or more runs (e.g. the two deep fields), each corrected
on its own (Corrections II, III), then summed per mass

usage: two_sided_analysis.py RUN [RUN ...] [--refresh]
    RUN: a run directory name under /mnt/sdceph/users/chainje/spax
"""

import os

# popeye's login node caps threads per user (pids.max = 256): keep BLAS and duckdb small
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import colorsys
import functools
import hashlib
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chic  # matplotlib style; turns on text.usetex (module load texlive)
import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.special import ndtr

# all text sans serif, math too: only sansmathfonts has sans Greek (sfmath, sansmath keep cmmi's)
plt.rcParams["text.latex.preamble"] += r"\usepackage{sansmathfonts}"

REFRESH = "--refresh" in sys.argv
duckdb.execute("SET threads = 8")
# a run is ~10^10 rows: exact median/MAD/quantiles (which keep every value) don't fit in memory, so
# the diagnostics use approx_quantile (t-digest, bounded memory). under Slurm, keep duckdb below
# the allocation so it spills instead of being OOM-killed
# spill outside ~/spax: it is mutagen-synced to the laptop
DUCKDB_TMP = "/mnt/sdceph/users/chainje/spax/duckdb-tmp"
duckdb.execute(f"SET temp_directory = '{DUCKDB_TMP}'")
if "SLURM_MEM_PER_NODE" in os.environ:
    duckdb.execute(f"SET memory_limit = '{int(os.environ['SLURM_MEM_PER_NODE']) * 85 // 100}MB'")

HBAR_EV_S = 6.582119569e-16  # https://en.wikipedia.org/wiki/Planck_constant
SPEED_OF_LIGHT_CGS = 2.99792458e10  # https://en.wikipedia.org/wiki/Speed_of_light

RUN_NAMES = [a for a in sys.argv[1:] if not a.startswith("--")]
assert RUN_NAMES, __doc__
RUNS = [Path("/mnt/sdceph/users/chainje/spax") / name for name in RUN_NAMES]
FIG_DIR = Path(__file__).parent.parent / "plots" / "+".join(RUN_NAMES)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# the mass grid and Gamma unit must agree between runs for them to be summed
CONFIGS = [json.loads((run / "config.json").read_text()) for run in RUNS]
for c in CONFIGS[1:]:
    assert c["windows"] == CONFIGS[0]["windows"]
    assert c["gamma_unit_per_s"] == CONFIGS[0]["gamma_unit_per_s"]
GAMMA_UNIT = CONFIGS[0]["gamma_unit_per_s"]  # A, B are stored in units of this Gamma
N_SUB = len(CONFIGS[0]["windows"][0]["masses_um"])
N_CHAN = 1 + max(w["lam_idx"] for w in CONFIGS[0]["windows"])
WAVES = np.full((N_CHAN, N_SUB), np.nan)  # mass wavelength [um] by (lam_idx, sub)
for w in CONFIGS[0]["windows"]:
    WAVES[w["lam_idx"]] = w["masses_um"]


def masses(run):
    return f"read_parquet('{run}/*.parquet')"


def windows(run):
    return f"read_parquet('{run}/windows/*.parquet')"


def cached(fn):
    """cache a reduction of one run (the first argument) to npz in the run's directory"""

    @functools.wraps(fn)
    def wrapper(run, *args, **kwargs):
        key = inspect.getsource(fn) + repr(args) + repr(sorted(kwargs.items()))
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        cache_dir = run / "cache"
        path = cache_dir / f"{fn.__name__}-{h}.npz"
        if path.exists() and not REFRESH:
            print(f"using cached result for {fn.__name__} on {run.name}")
            return dict(np.load(path))
        print(f"running {fn.__name__} on {run.name}")
        out = fn(run, *args, **kwargs)
        cache_dir.mkdir(exist_ok=True)
        np.savez(path, **out)
        return out

    return wrapper


@cached
def per_mass(run):
    return duckdb.sql(f"""
    SELECT lam_idx, sub, sum(A) AS A, sum(B) AS B
    FROM {masses(run)}
    GROUP BY lam_idx, sub
    """).fetchnumpy()


def sum_over(keys, A, B):
    """sum A, B over entries with equal keys (a list of int arrays); returns the unique
    keys (sorted by wavelength) and the sums"""
    k = np.stack(keys, axis=-1)
    u, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    A, B = np.bincount(inv, A), np.bincount(inv, B)
    order = np.argsort(WAVES[u[:, 0], u[:, 1]], kind="stable")
    return [col for col in u[order].T], A[order], B[order]


def m_a_from_lam_a(lam_a):
    return (4 * np.pi) / lam_a * 1e4 * SPEED_OF_LIGHT_CGS * HBAR_EV_S  # eV


def g_agg_from_gamma(gamma, m_a):
    return 1e9 * np.sqrt(64 * np.pi * HBAR_EV_S * gamma / m_a**3)


def gamma_95(A, B, pcl=True):
    """one-sided 95% upper limit on Gamma [GAMMA_UNIT], Delta chi^2 = 2.71 from max(0, B/A)

    with pcl, power-constrained (Cowan+ 1105.3166) at the 16% threshold of Janish & Pinetti
    2310.15395 Eq. 14: the limit is monotonic in z = B / sqrt(A) ~ N(0, 1) under background
    only, so its 15.87% quantile is the limit at z = -1, i.e. (sqrt(1 + 2.71) - 1) / sqrt(A)
    """
    limit = (B + np.sqrt(2.71 * A + np.minimum(B, 0.0) ** 2)) / A
    if pcl:
        limit = np.maximum(limit, (np.sqrt(1 + 2.71) - 1) / np.sqrt(A))
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
    ax.tick_params(axis="x", which="both", top=False)  # the top axis is wavelength's alone

    def convert(x):
        with np.errstate(divide="ignore"):  # matplotlib probes x = 0
            return m_a_from_lam_a(x)

    top = ax.secondary_xaxis("top", functions=(convert, convert))
    top.set_xlabel(r"$\lambda$ [$\mu$m]")
    set_ticks(ax.xaxis, MASS_TICKS)
    set_ticks(top.xaxis, WAVE_TICKS)


for run in RUNS:
    r = per_mass(run)
    (li, sub), A, B = sum_over([r["lam_idx"], r["sub"]], r["A"], r["B"])
    plt.plot(WAVES[li, sub], np.abs(B / np.sqrt(A)), label=run.name, lw=0.8)
plt.legend(fontsize="small")
plt.xlabel(r"$\lambda$ [$\mu$m]")
plt.ylabel(r"$|z|$")
plt.xscale("log")
plt.yscale("log")
set_ticks(plt.gca().xaxis, WAVE_TICKS)
plt.savefig(FIG_DIR / "pull.png")
plt.close()


@cached
def pull_stats(run):
    """per (lam_idx, sub, det): z statistics, and the window's continuum-fit chi^2

    two streaming passes over the run (one query referencing the rows twice makes duckdb
    materialize all ~10^10 of them): the median first, then the MAD about it
    """
    m = duckdb.sql(f"""
    SELECT
        lam_idx,
        sub,
        det,
        approx_quantile(B / sqrt(A), 0.5) AS median_z,
        countif(abs(B / sqrt(A)) > 3.0) / count() AS frac_g3,
    FROM {masses(run)}
    WHERE A > 0  -- z is 0/0 = NaN otherwise, which duckdb sorts above everything
    GROUP BY lam_idx, sub, det
    """).to_arrow_table()
    z = duckdb.sql(f"""
    SELECT
        lam_idx,
        sub,
        det,
        any_value(median_z) AS median_z,
        any_value(frac_g3) AS frac_g3,
        1.4826 * approx_quantile(abs(B / sqrt(A) - median_z), 0.5) AS robust_width,
    FROM {masses(run)} JOIN m USING (lam_idx, sub, det)
    WHERE A > 0
    GROUP BY lam_idx, sub, det
    """).to_arrow_table()
    return duckdb.sql(f"""
    WITH w AS (
        SELECT
            lam_idx,
            det,
            approx_quantile(sqrt(chi2_0 / (n_rows - rank)), 0.5) FILTER (WHERE n_rows > rank) AS median_s_chi2,
        FROM {windows(run)}
        GROUP BY lam_idx, det
    )
    SELECT * FROM z JOIN w USING (lam_idx, det)
    ORDER BY det, lam_idx, sub
    """).fetchnumpy()


fig, axs = plt.subplots(len(RUNS), 4, figsize=(13, 3 * len(RUNS)), sharex=True, squeeze=False)

for run, axrow in zip(RUNS, axs):
    result = pull_stats(run)
    lam = WAVES[result["lam_idx"], result["sub"]]
    for d in np.unique(result["det"]):
        mask = result["det"] == d
        kw = dict(color=f"C{d - 1}", label=f"D{d}")
        axrow[0].plot(lam[mask], result["median_z"][mask], **kw)
        axrow[1].plot(lam[mask], result["robust_width"][mask], **kw)
        axrow[2].plot(lam[mask], result["frac_g3"][mask], **kw)
        axrow[1].plot(lam[mask], result["median_s_chi2"][mask], ls="--", color=f"C{d - 1}")
        axrow[3].plot(
            lam[mask], result["robust_width"][mask] / result["median_s_chi2"][mask], **kw
        )
    axrow[0].set_title(run.name, fontsize="small", loc="left")

# expectation for N(0, 1) pulls and chi^2-distributed chi2_0
for axrow in axs:
    axrow[0].axhline(0, color="k", lw=0.5)
    axrow[1].axhline(1, color="k", lw=0.5)
    axrow[2].axhline(0.0027, color="k", lw=0.5)
    axrow[3].axhline(1, color="k", lw=0.5)
    axrow[0].set_yscale("symlog", linthresh=1)
    axrow[1].set_yscale("log")
    axrow[2].set_yscale("log")
    axrow[3].set_yscale("log")
    axrow[0].set_ylabel(r"median $z$")
    axrow[1].set_ylabel(r"width of $z$ (solid), $\sqrt{\chi^2_0 / \nu}$ (dashed)")
    axrow[2].set_ylabel(r"frac $|z| > 3$")
    axrow[3].set_ylabel(r"width of $z$ / $\sqrt{\chi^2_0 / \nu}$")
axs[0, 0].legend(fontsize="small", ncols=2)

for ax in axs.flat:
    ax.set_xscale("log")
    set_ticks(ax.xaxis, WAVE_TICKS)
for ax in axs[-1]:
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")

fig.tight_layout()
fig.savefig(FIG_DIR / "pull-stats.png")
plt.close()


@cached
def pooled_hist(run, lo=-10.0, hi=10.0, width=0.1):
    """per-detector histogram of z over all lam, bins. bin -1 / n_bins are under/overflow"""
    n_bins = round((hi - lo) / width)
    return duckdb.sql(f"""
    SELECT
        det,
        greatest(least(floor((B / sqrt(A) - {lo}) / {width}), {n_bins}), -1)::INTEGER AS bin,
        count() AS n,
    FROM {masses(run)}
    WHERE A > 0
    GROUP BY det, bin
    ORDER BY det, bin
    """).fetchnumpy()


@cached
def pooled_quantiles(run, z_max=4.5, n=181):
    """per-detector quantiles of z at the N(0, 1) quantiles z_ref"""
    z_ref = np.linspace(-z_max, z_max, n)
    probs = ", ".join(repr(float(p)) for p in ndtr(z_ref))
    out = duckdb.sql(f"""
    WITH t AS (
        SELECT det, approx_quantile(B / sqrt(A), [{probs}]::FLOAT[]) AS q
        FROM {masses(run)}
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

fig, axs_all = plt.subplots(len(RUNS), 2, figsize=(10, 4 * len(RUNS)), squeeze=False)
for run, axs in zip(RUNS, axs_all):
    hist = pooled_hist(run, lo, hi, width)
    quant = pooled_quantiles(run)

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
            label=f"D{d} ({100 * frac_out:.1f}\\% outside)",
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
    axs[0].set_title(run.name, fontsize="small", loc="left")

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
# detector; we use other channels of the same detector instead: the masses at the same
# sub-channel position (sub) of the channels 2 <= |l - j| <= K + 1 away. same sub: masses a
# fraction of a channel apart are strongly correlated (rho ~ 0.76 at 1/4 channel), which
# would bias the MAD low. skipping |l - j| = 1: through the continuum fit, a line in channel
# j pulls z down in j +- 1 (rho ~ -0.38), so it would inflate its own S
# (scripts/claude/proto_oversample.py, 2026-10-26)
MIN_NEIGHBORS = 5  # fewer than this -> S = 1 (her rule for III; applied to II too)
NB_SKIP = 1  # channels skipped on either side


@cached
def correction_2(run, K):
    """per-bin MAD rescaling (Correction II), summed over bins per (lam_idx, sub, det)

    for each bin i and mass (j, sub): S = 1.4826 * MAD of z over the 2K nearest masses of
    the same bin, detector, and sub with |l - j| > NB_SKIP (as in correction_3: extended
    one-sidedly near detector edges); A, B are scaled by min(1, 1 / S^2).
    bins never span partitions, so this runs one partition file at a time (all at once
    gets OOM-killed on the login node), several files in parallel
    """

    def one_part(part):
        con = duckdb.connect(config={"threads": 2, "temp_directory": DUCKDB_TMP})  # login node caps threads per user
        return con.sql(f"""
        WITH t AS (
            SELECT hpbin, det, sub, lam_idx, A, B, B / sqrt(A) AS z
            FROM read_parquet('{part}')
            WHERE A > 0
        ),
        w AS (
            SELECT
                *,
                -- sorting structs sorts by distance, then lam_idx (as correction_3's stable sort)
                list_transform(
                    list_slice(
                        list_sort(
                            list_filter(
                                list_transform(
                                    list(struct_pack(l := lam_idx, z := z)) OVER nb,
                                    x -> struct_pack(d := abs(x.l - lam_idx), l := x.l, z := x.z)
                                ),
                                x -> x.d > {NB_SKIP}
                            )
                        ),
                        1,
                        {2 * K}
                    ),
                    x -> x.z
                ) AS z_nb,
            FROM t
            WINDOW nb AS (PARTITION BY hpbin, det, sub)
        ),
        f AS (
            SELECT
                *,
                1.4826 * list_aggregate(z_nb, 'mad') AS S,
            FROM w
        ),
        g AS (
            SELECT
                *,
                CASE WHEN len(z_nb) >= {MIN_NEIGHBORS} AND S > 1 THEN 1 / S^2 ELSE 1 END AS f,
            FROM f
        )
        SELECT
            lam_idx,
            sub,
            det,
            sum(A) AS A_raw,
            sum(B) AS B_raw,
            sum(f * A) AS A,
            sum(f * B) AS B,
            countif(f < 1) AS n_rescaled,
            count() AS n,
        FROM g
        GROUP BY lam_idx, sub, det
        """).to_arrow_table()

    files = sorted(run.glob("*.parquet"))
    partial = []
    with ThreadPoolExecutor(8) as pool:
        for k, table in enumerate(pool.map(one_part, files)):
            partial.append(table)
            if (k + 1) % 100 == 0 or k + 1 == len(files):
                print(f"  correction_2(K={K}): {k + 1}/{len(files)} files", flush=True)
    partial = pa.concat_tables(partial)
    return duckdb.sql("""
    SELECT
        lam_idx,
        sub,
        det,
        sum(A_raw) AS A_raw,
        sum(B_raw) AS B_raw,
        sum(A) AS A,
        sum(B) AS B,
        sum(n_rescaled) / sum(n) AS frac_rescaled,
    FROM partial
    GROUP BY lam_idx, sub, det
    ORDER BY det, sub, lam_idx
    """).fetchnumpy()


def correction_3(lam_idx, sub, det, A, B, K):
    """joint MAD rescaling (Correction III) per (lam_idx, sub, det)

    S = max(1, 1.4826 * MAD of joint pulls z over the 2K nearest masses of the same detector
    and sub with |l - j| > NB_SKIP), or 1 with fewer than MIN_NEIGHBORS finite neighbors; A, B
    are scaled by 1 / S^2. away from detector edges these are NB_SKIP + 1 <= |l - j| <=
    NB_SKIP + K on both sides; near an edge the neighborhood extends further on the side that
    exists, so every mass gets 2K: with the K one-sided neighbors alone, a MAD of 5 collapsed
    when 3 happened to cluster (S = 35 vs ~400-1400 nearby at 2.327 um, 2026-10-29)
    """
    z = B / np.sqrt(A)
    S = np.ones_like(A)
    for k in range(len(A)):
        dist = np.abs(lam_idx - lam_idx[k])
        cand = np.flatnonzero(
            (det == det[k]) & (sub == sub[k]) & (dist > NB_SKIP) & np.isfinite(z)
        )
        nb = z[cand[np.argsort(dist[cand], kind="stable")[: 2 * K]]]
        if len(nb) >= MIN_NEIGHBORS:
            S[k] = max(1.0, 1.4826 * np.median(np.abs(nb - np.median(nb))))
    return A / S**2, B / S**2, S


def corrected(K):
    """Correction II on each run, then the runs summed per (lam_idx, sub, det), then
    Correction III on the summed pulls. returns {name: dict of per (lam_idx, sub, det)
    arrays: A_raw, B_raw (no corrections), A_II, B_II, A, B (II + III), S3}, with a single
    entry named after all runs

    III is joint, not per run: the residual it measures is common to the fields (S grows
    by ~sqrt(2) from one deep field to two, 2026-10-29), so per-field S underestimates
    the scatter of the summed pulls"""
    c2s = [correction_2(run, K) for run in RUNS]
    keys = [np.concatenate([c2[k] for c2 in c2s]) for k in ("lam_idx", "sub", "det")]
    cols = {}
    for a, b, a2, b2 in [("A_raw", "B_raw", "A_raw", "B_raw"), ("A_II", "B_II", "A", "B")]:
        A = np.concatenate([c2[a2] for c2 in c2s])
        B = np.concatenate([c2[b2] for c2 in c2s])
        summed_keys, cols[a], cols[b] = sum_over(keys, A, B)  # same keys -> same order
    A3, B3, S3 = correction_3(*summed_keys, cols["A_II"], cols["B_II"], K)
    return {"+".join(RUN_NAMES): dict(keys=summed_keys, S3=S3, A=A3, B=B3, **cols)}


def combine(res, a="A", b="B", over_det=True):
    """sum (A, B) columns over runs (and detectors); returns keys, A, B by wavelength"""
    li = np.concatenate([r["keys"][0] for r in res.values()])
    sub = np.concatenate([r["keys"][1] for r in res.values()])
    det = np.concatenate([r["keys"][2] for r in res.values()])
    A = np.concatenate([r[a] for r in res.values()])
    B = np.concatenate([r[b] for r in res.values()])
    keys = [li, sub] if over_det else [li, sub, det]
    return sum_over(keys, A, B)


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for K, ls in [(5, "-"), (10, "--")]:
    res = corrected(K)
    for color, label, (a, b) in [
        ("C7", "raw", ("A_raw", "B_raw")),
        ("C0", "II", ("A_II", "B_II")),
        ("C3", "II+III", ("A", "B")),
    ]:
        if label == "raw" and K != 5:
            continue  # independent of K
        (li, sub), A, B = combine(res, a, b)
        lam = WAVES[li, sub]
        name = label if label == "raw" else f"{label}, K={K}"
        axs[0].plot(lam, np.abs(B / np.sqrt(A)), ls=ls, color=color, label=name, lw=0.8)
        m_a = m_a_from_lam_a(lam)
        axs[1].plot(
            m_a,
            g_agg_from_gamma(GAMMA_UNIT * gamma_95(A, B), m_a),
            ls=ls,
            color=color,
            label=name,
            lw=0.8,
        )

    for run_name, r in res.items():
        li, sub, det = r["keys"]
        for d in np.unique(det):
            mask = det == d
            o = np.argsort(WAVES[li[mask], sub[mask]])
            lam = WAVES[li[mask], sub[mask]][o]
            kw = dict(ls=ls, color=f"C{d - 1}")
            axs[2].plot(lam, r["S3"][mask][o], label=f"D{d}" if K == 5 else None, lw=0.8, **kw)
            axs[2].plot(lam, (r["A_II"] / r["A_raw"])[mask][o], alpha=0.4, lw=0.6, **kw)

axs[0].set_ylabel(r"joint $|z|$ (all runs)")
axs[1].set_ylabel(r"$g_{a\gamma\gamma}$ 95\% upper limit (PCL) [GeV$^{-1}$]")
axs[2].set_ylabel(r"$S_\mathrm{III}$ (dark), $A_\mathrm{II} / A_\mathrm{raw}$ (light)")
axs[0].legend(fontsize="small")
axs[2].legend(
    fontsize="small", ncols=2, title="solid K=5, dashed K=10", title_fontsize="small"
)
for ax in axs:
    ax.set_yscale("log")
for ax in axs[[0, 2]]:
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_xscale("log")
    set_ticks(ax.xaxis, WAVE_TICKS)
mass_axis(axs[1])

fig.tight_layout()
fig.savefig(FIG_DIR / "corrections.png")
plt.close()

# headline limit: Corrections II + III with K = 5, power-constrained, all runs
K = 5
res = corrected(K)
(li, sub), A, B = combine(res)
m_a = m_a_from_lam_a(WAVES[li, sub])
_, A_raw, _ = combine(res, "A_raw", "B_raw")

# sensitivity forecast for the analyzed field(s), as Regis+2025 but from the measured noise
# (scripts/measured_forecast.py): A = sum a^2 / var over the rows the runs use, no continuum,
# no Correction 1. A grows with the data, so it is scaled from what we have to the 25 month mission
# (Explanatory Supplement v2.0, surveys 1-4). same threshold as the Cramer-Rao curve (2.71)
T_MISSION = 25  # months
forecast = np.load(Path(__file__).parent.parent / "data/measured-forecast.npz")
fields = [c["field"] for c in CONFIGS]
_A_forecast = {
    key: a for key, a in zip(zip(forecast["lam_idx"], forecast["sub"]), sum(forecast[f"A_{f}"] for f in fields))
}
A_forecast = np.array([_A_forecast[key] for key in zip(li, sub)]) * T_MISSION / forecast["t_now_months"]
g_forecast = g_agg_from_gamma(GAMMA_UNIT * np.sqrt(2.71 / A_forecast), m_a)

upper = g_agg_from_gamma(GAMMA_UNIT * gamma_95(A, B), m_a)

# known astrophysical / atmospheric lines, for reference [um]: He I 1.0830 (atmospheric, also
# leaks into other detectors' pixels), H Paschen-alpha 1.8751, 3.3 um PAH, H Brackett-alpha 4.0512
LINES = {"He I": 1.0830, r"Pa$\alpha$": 1.8751, "PAH": 3.29, r"Br$\alpha$": 4.0512}
# where He I 1.083 um leaks into other pixels (Explanatory Supplement sec. 2.3): D1 pixels with
# bandpass center near 0.84 um, D2 near 1.25 and 1.49 um
HE_LEAK = [0.84, 1.25, 1.49]
# detector boundaries [um]: geometric midpoint between one detector's last mass and the next's
# first (D1 and D2 overlap a little, at 1.09-1.12 um)
DET_RANGE = {}
for w in CONFIGS[0]["windows"]:
    lo, hi = DET_RANGE.get(w["det"], (np.inf, 0))
    DET_RANGE[w["det"]] = (min(lo, min(w["masses_um"])), max(hi, max(w["masses_um"])))
STRIP = 0.05  # height of the detector strip, as a fraction of the axes
DET_EDGES = {
    f"D{d}$|$D{d + 1}": np.sqrt(DET_RANGE[d][1] * DET_RANGE[d + 1][0]) for d in sorted(DET_RANGE)[:-1]
}

# existing limits, as drawn by Pinetti 2503.11753 Fig. 2: one region above their envelope. from
# cajohare/AxionLimits limit_data/AxionPhoton (commit in axionlimits/SOURCE.txt), except: Pinetti
# v2's curves, from Elena Pinetti directly (axionlimits/JWST_Pinetti.txt is her v1, ~50x tighter;
# not used), and Gemmell+ 2026 (commit in gemmell2026/SOURCE.txt)
EXTERNAL = Path(__file__).parent.parent / "data/external"
CAST = [
    "axionlimits/CAST.txt",  # helioscope
    "axionlimits/CAST_highm.txt",
]
# limits on the decay itself: for the lifetime panel, these alone (Elena: model-independent)
DECAY = [
    "axionlimits/JWST_Saha.txt",  # JWST blank sky, Saha+ 2025
    "axionlimits/JWST.txt",  # JWST blank sky, Janish & Pinetti 2023
    "axionlimits/WINERED.txt",  # Yin+ 2024
    "axionlimits/Telescopes_MUSE.txt",  # MUSE dwarfs, Todarello+ 2023
    "elena/bounds_miri.csv",  # JWST, Pinetti 2503.11753 v2
    "elena/bounds_nirspec.csv",
    "gemmell2026/MIRI_limits_25_spurious_signal.csv",  # JWST, Gemmell+ 2609.10674
    "gemmell2026/NIRSPEC_limits_25_spurious_signal.csv",
]
EXISTING = CAST + DECAY
GLOBULAR = "axionlimits/GlobularClusters.txt"  # stellar cooling: a line, not a region
# Elena: below this her NIRSpec curve is at masses outside the JWST band, which overlap it only
# through the tail of the LSF (g up to 1e63 GeV^-1)
MIN_MASS = {"elena/bounds_nirspec.csv": 0.468}  # eV
TOP = 1e-8  # GeV^-1, top of the plot (existing limits are filled up to here)


def load_limit(name):
    """(m [eV], g [GeV^-1]) of a limit curve: AxionLimits' .txt, or a .csv with mass, limit in its
    first two columns (Gemmell+'s have blank rows, at masses they mask)"""
    if name.endswith(".csv"):
        m, g = np.genfromtxt(EXTERNAL / name, delimiter=",", skip_header=1, usecols=(0, 1), unpack=True)
        keep = np.isfinite(g) & (m >= MIN_MASS.get(name, 0))
        return m[keep], g[keep]
    return np.loadtxt(EXTERNAL / name, comments="#", unpack=True, usecols=(0, 1))


def bin_edges(m):
    """edges between sorted masses: geometric midpoints, the ends mirrored"""
    mid = np.sqrt(m[1:] * m[:-1])
    return np.concatenate([[m[0] ** 2 / mid[0]], mid, [m[-1] ** 2 / mid[-1]]])


# our curves as stairs, one step per mass
order = np.argsort(m_a)
edges = bin_edges(m_a[order])
cr = g_agg_from_gamma(GAMMA_UNIT * np.sqrt(2.71 / A_raw), m_a)



def envelope(names, m):
    """the tightest of the limit curves `names` at masses m (inf where none applies), each
    interpolated in log-log"""
    out = np.full_like(m, np.inf)
    for name in names:
        mc, gc = load_limit(name)
        o = np.argsort(mc)
        mc, gc = mc[o], gc[o]
        inside = (m >= mc[0]) & (m <= mc[-1])
        out[inside] = np.minimum(
            out[inside], np.exp(np.interp(np.log(m[inside]), np.log(mc), np.log(gc)))
        )
    return out


def qcd_axion_g(m_a, E_over_N):
    """g_agg [GeV^-1] of the QCD axion of mass m_a [eV] for anomaly ratio E/N: (alpha / 2 pi)
    |E/N - 1.92| / f_a with m_a = 5.70 ueV (1e12 GeV / f_a), Grilli di Cortona+ 1511.02867"""
    f_a = 5.70e-6 * 1e12 / m_a  # GeV
    return 1 / 137.035999 / (2 * np.pi) * np.abs(E_over_N - 1.92) / f_a


# evaluated on every mass any curve samples, so their spikes survive
m_grid = np.unique(
    np.concatenate([edges] + [load_limit(name)[0] for name in EXISTING])
)
m_grid = m_grid[(m_grid >= edges[0]) & (m_grid <= edges[-1])]
THIS_WORK = "Hainje et al. (this work)"
FORECAST = "Sensitivity forecast (25 mo.)"
CR_BOUND = "Cram\\'er--Rao bound (16.5 mo.)"
STYLE = {  # label: (color, lw, ls)
    THIS_WORK: ("k", 1.1, "-"),
    FORECAST: ("C1", 1, "--"),
    CR_BOUND: ("C2", 0.9, "-"),
    "Existing limits": ("C0", None, None),
    "QCD axion": ("#DDB945", None, None),  # Purdue "Field", purdue.edu/brand-studio/brand/visual-identity
    "Globular clusters": ("C5", 1.2, ":"),
}
ALPHA = {"Existing limits": 0.2, "QCD axion": 1}  # the regions' fill alphas


def text_along(ax, x, y_of, s, dy=0, **kw):
    """`s` at (x, y_of(x)), shifted dy points up, rotated to follow the curve y_of in display
    space (its mean slope over +-25% in x, about a label's length); call after the axes' limits
    and layout are final"""
    p0, p1 = ax.transData.transform([(x / 1.25, y_of(x / 1.25)), (x * 1.25, y_of(x * 1.25))])
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(
        s, (x, y_of(x)), xytext=(0, dy), textcoords="offset points", rotation=angle, rotation_mode="anchor", **kw
    )


def curve(m, g):
    """g(m) as a function, interpolated in log-log (any order of the points)"""
    o = np.argsort(m)
    return lambda x: np.exp(np.interp(np.log(x), np.log(m[o]), np.log(g[o])))


def tau_from_g(g, m):
    """the lifetime tau = 1 / Gamma [s] of an axion of mass m [eV] with coupling g [GeV^-1]: the
    inverse of g_agg_from_gamma"""
    return 64 * np.pi * HBAR_EV_S / ((1e-9 * g) ** 2 * m**3)


TAU_LIM = (1e20, 1e33)  # s, the lifetime panel's y range
FIG_SIZE = (7.25, 4.6)  # in: for \\includegraphics at the page's full width, so text keeps its pt size


def draw_constraint(ax, he_leak=False, detectors=False, tau=False):
    """the limit plot's curves on ax, in g_agg or (tau) as lower limits on the lifetime tau,
    where an upper limit on g is a lower limit on tau and "excluded" lies below the curve, and
    with only the limits on the decay itself: no QCD axion band, globular clusters, or CAST"""
    Y = (lambda g, m: tau_from_g(g, m)) if tau else (lambda g, m: g)
    fill = lambda name, alpha: dict(color=STYLE[name][0], alpha=alpha, lw=0)
    line = lambda name: dict(zip(["color", "lw", "ls"], STYLE[name]))
    # existing limits exclude large g, so small tau: fill to the panel's edge
    far = TAU_LIM[0] if tau else TOP
    ax.fill_between(
        m_grid, Y(np.minimum(envelope(DECAY if tau else EXISTING, m_grid), TOP), m_grid), far, zorder=1, rasterized=True,
        **fill("Existing limits", ALPHA["Existing limits"]),
    )
    if not tau:
        # Pinetti's band: DFSZ (E/N = 8/3) to KSVZ (E/N = 0)
        m_qcd = np.geomspace(edges[0], edges[-1], 200)  # a smooth power law: few points will do
        ax.fill_between(
            m_qcd, qcd_axion_g(m_qcd, 8 / 3), qcd_axion_g(m_qcd, 0), zorder=1.5,
            **fill("QCD axion", ALPHA["QCD axion"]),
        )
        m_gc, g_gc = load_limit(GLOBULAR)
        ax.plot(m_gc, g_gc, zorder=2, **line("Globular clusters"))
    m_o = m_a[order]
    ax.stairs(Y(cr[order], m_o), edges, baseline=None, zorder=2, **line(CR_BOUND))
    ax.stairs(Y(g_forecast[order], m_o), edges, baseline=None, zorder=2, **line(FORECAST))
    # the reference lines and their labels over everything but our limit
    lines = list(LINES.items()) + ([("He I leak", lam) for lam in HE_LEAK] if he_leak else [])
    for name, lam_line in lines:
        ax.axvline(m_a_from_lam_a(lam_line), ymin=STRIP if detectors else 0, color="0.6", lw=0.6, ls=":", zorder=3)
        ax.text(
            m_a_from_lam_a(lam_line),
            0.98,
            name,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize="small",
            color="0.4",
            zorder=3,
        )
    if detectors:
        # a strip along the bottom with each detector's span, alternately shaded. zorder 1.2: under
        # the ticks (axes.axisbelow = "line" puts them at 1.5); in g nothing else reaches down
        # here, in tau it covers the existing limits' fill
        m_bounds = sorted(m_a_from_lam_a(lam) for lam in DET_EDGES.values())
        strip = blended_transform_factory(ax.transData, ax.transAxes)
        spans = zip([edges[0]] + m_bounds, m_bounds + [edges[-1]])
        for k, (m_lo, m_hi) in enumerate(spans):  # from low mass: D6 first
            ax.add_patch(
                Rectangle(
                    (m_lo, 0), m_hi - m_lo, STRIP, transform=strip,
                    facecolor="0.88" if k % 2 else "white", edgecolor="0.4", lw=0.6, zorder=1.2,
                )
            )
            ax.text(
                np.sqrt(m_lo * m_hi), STRIP / 2, f"D{6 - k}", transform=strip,
                ha="center", va="center", fontsize="small", color="0.3", zorder=1.2,
            )
    ax.stairs(Y(upper[order], m_o), edges, baseline=None, zorder=4, **line(THIS_WORK))

    ax.set_yscale("log")
    if tau:
        ax.set_ylabel(r"$\tau$ [s]")
        ax.set_ylim(*TAU_LIM)
    else:
        ax.set_ylabel(r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]")
        ax.set_ylim(4e-14 if detectors else 1e-13, TOP)  # room for the detector strip
    ax.set_xlim(edges[0], edges[-1])


def label_constraint(ax, tau=False):
    """draw_constraint's labels next to the curves themselves; call after the layout is final:
    text_along's angles depend on the axes' shape. in tau the curves' order flips, so each label
    goes to the other side of its curve"""
    Y = (lambda g, m: tau_from_g(g, m)) if tau else (lambda g, m: g)
    flip = {"top": "bottom", "bottom": "top"}
    side = lambda va, dy: (flip[va], -dy) if tau else (va, dy)  # (va, dy) on the g panel
    kw = dict(fontsize="small", ha="center")
    # the fill's color, darkened to read on the fill: `f` x its HLS lightness, hue and saturation
    # kept (scaling RGB instead also desaturates light colors like the QCD gold)
    def dark(name, f):
        h, l, s = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(STYLE[name][0]))
        return colorsys.hls_to_rgb(h, f * l, s)
    # under our limit at the low-mass end
    va, _ = side("top", 0)
    ax.text(0.68, Y(1.2e-11, 0.68), THIS_WORK, va=va, color=STYLE[THIS_WORK][0], **kw)
    ax.text(1.0, Y(2e-9, 1.0), "Existing limits", va="center", color=dark("Existing limits", 0.7), **kw)
    if not tau:
        # above its line, at the right edge
        m_gc, g_gc = load_limit(GLOBULAR)
        text_along(
            ax, edges[-1] * 0.97, curve(m_gc, g_gc), "Globular\nclusters", dy=2, va="bottom",
            color=STYLE["Globular clusters"][0], **kw | dict(ha="right", multialignment="right"),
        )
        # the band's middle; dy: the text's box sits a little high of its ink, so va="center" alone
        # reads as above the middle
        band = lambda m: np.sqrt(qcd_axion_g(m, 8 / 3) * qcd_axion_g(m, 0))
        text_along(ax, 1.5, band, "QCD axion", dy=-0.75, va="center", color=dark("QCD axion", 0.5), **kw)
    # clear of the bump at the D3-D4 boundary, ~1.03 eV
    va, dy = side("bottom", 5)
    text_along(ax, 1.5, curve(m_a, Y(cr, m_a)), CR_BOUND, dy=dy, va=va, color=STYLE[CR_BOUND][0], **kw)
    va, dy = side("top", -5)
    text_along(ax, 1.15, curve(m_a, Y(g_forecast, m_a)), FORECAST, dy=dy, va=va, color=STYLE[FORECAST][0], **kw)


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300)  # dpi of the rasterized existing limits
    plt.close(fig)


def plot_constraint(inline, he_leak=False, detectors=False):
    """the limit plot; labels in a legend, or (inline) next to the curves themselves. he_leak:
    also mark where He I leaks into other pixels; detectors: the detector boundaries"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    draw_constraint(ax, he_leak, detectors)
    mass_axis(ax)
    if not inline:
        fill = lambda name, alpha: dict(color=STYLE[name][0], alpha=alpha, lw=0)
        line = lambda name: dict(zip(["color", "lw", "ls"], STYLE[name]))
        handles = [
            Patch(**fill(name, ALPHA[name]), label=name)
            if STYLE[name][1] is None
            else Line2D([], [], **line(name), label=name)
            for name in STYLE
        ]
        ax.legend(handles=handles, fontsize="small", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    if inline:
        label_constraint(ax)
    name = (
        ("constraint-labeled" if inline else "constraint")
        + ("-he-leak" if he_leak else "")
        + ("-detectors" if detectors else "")
    )
    save(fig, name)


def plot_constraint_tau():
    """plot_constraint(inline=True, he_leak=True, detectors=True), as lower limits on the
    lifetime tau"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    draw_constraint(ax, he_leak=True, detectors=True, tau=True)
    mass_axis(ax)
    fig.tight_layout()
    label_constraint(ax, tau=True)
    save(fig, "constraint-tau-labeled-he-leak-detectors")

# where this work beats the existing limits, per mass (m_a ordered)
best = envelope(EXISTING, m_a)
print(f"this work / existing: median {np.median(upper / best):.2f}; tighter at {(upper < best).sum()}/{len(upper)} masses")
print(
    f"median g: this work {np.median(upper):.3e}, Cramer-Rao {np.median(cr):.3e}, forecast {np.median(g_forecast):.3e}; "
    f"this work / forecast: median {np.median(upper / g_forecast):.1f}"
)
plot_constraint(inline=False)
plot_constraint(inline=True)
plot_constraint(inline=True, he_leak=True)
plot_constraint(inline=True, he_leak=True, detectors=True)
plot_constraint_tau()
