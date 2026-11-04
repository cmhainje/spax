"""
small figures for the paper: the sky coverage of the two deep-field runs, and a sample
spectrum of one analysis bin (with the spectral images two of its measurements come from)

    sbatch scripts/claude/coverage.sbatch  # once: counts the measurements used
    module load texlive  # chic turns on text.usetex
    uv run python scripts/paper_plots.py coverage
    uv run python scripts/paper_plots.py coverage-rough  # footprint only, from the run outputs
    uv run python scripts/paper_plots.py spectrum
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TALLTABLE_DB_DIR", "/mnt/sdceph/users/spherex/talltable")

import chic
import healpy
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import offset_copy
from talltable.constants import HP_HIGH_LEVEL
from talltable.partition import find_partitions_ipix
from talltable.paths import IMAGE_DB_PATH, PIXEL_DB_PATH, WAVES_DB_PATH
from talltable.waveid import waveid_to_rowcoldet

RUN_DIR = Path("/mnt/sdceph/users/chainje/spax")
RUNS = {"NDF": "two-sided-2026-10-26-ndf", "SDF": "two-sided-2026-10-26-sdf"}
FIG_DIR = Path(__file__).parent.parent / "plots/paper"
BIN_LEVEL = 15  # as in two_sided.py

plt.rcParams["image.cmap"] = "tol_iridescent"
# all text sans serif, math too: only sansmathfonts has sans Greek (sfmath, sansmath keep cmmi's)
plt.rcParams["text.latex.preamble"] += r"\usepackage{sansmathfonts}"
FIG_WIDTH = 7.25  # in: for \\includegraphics at the page's full width, so text keeps its pt size
FIELD_NAMES = {"NDF": "North Deep Field", "SDF": "South Deep Field"}


def config(run):
    return json.loads((RUN_DIR / run / "config.json").read_text())


# --- sky coverage ---

# the runs' partitions are healpix pixels at levels 8 and 9, so a level-8 map is exact
MAP_LEVEL = 8


COVERAGE_DIR = RUN_DIR / f"coverage-level{MAP_LEVEL}"


def coverage_scan():
    """this task's share (partitions[i::n], as in two_sided.py) of the coverage counts per
    level-MAP_LEVEL pixel (nested, equatorial): all rows, the rows passing the cuts of
    two_sided.read_partition (the ones the analysis uses), and the rows lost only to
    flux <= 0. run under srun (scripts/claude/coverage.sbatch). checkpointed: one
    COVERAGE_DIR/part={p}.npz per partition (nonzero pixels only), skipped when it exists"""
    task = int(os.environ.get("SLURM_PROCID", "0"))
    n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    parts = sorted({p for run in RUNS.values() for p in config(run)["partitions"]})
    COVERAGE_DIR.mkdir(exist_ok=True)
    shift = 2 * (HP_HIGH_LEVEL - MAP_LEVEL)
    for part in parts[task::n_tasks]:
        out_path = COVERAGE_DIR / f"part={part}.npz"
        if out_path.exists():
            continue
        # ceph is slow at scattered reads: reading the whole file and parsing it in memory
        # is ~2x faster than read_table(columns=..., pre_buffer=True)
        with open(PIXEL_DB_PATH / f"part={part}/compacted.parquet", "rb") as f:
            buf = pa.BufferReader(f.read())
        t = pq.read_table(buf, columns=["hphigh", "flags", "flux", "variance", "zodi"])
        del buf
        hphigh, flags, flux, variance, zodi = (
            t.column(c).to_numpy() for c in t.column_names
        )
        del t
        # the row cuts of two_sided.read_partition (ALLOWED_BITS = 0), but flux > 0
        ok = (flags == 0) & (variance > 0)
        ok &= np.isfinite(flux) & np.isfinite(variance) & np.isfinite(zodi)
        # a partition (level >= MAP_LEVEL here) spans only a few pixels: count from the first
        pix = hphigh >> shift
        del hphigh
        lo = pix.min()
        pix -= lo
        tmp = out_path.with_name(
            f"tmp-{out_path.name}"
        )  # not matched by coverage_map's glob
        np.savez(
            tmp,
            pix=lo + np.arange(pix.max() + 1),
            n_all=np.bincount(pix),
            n_used=np.bincount(pix[ok & (flux > 0)], minlength=pix.max() + 1),
            n_nonpos=np.bincount(pix[ok & (flux <= 0)], minlength=pix.max() + 1),
        )
        os.replace(tmp, out_path)
        print(f"task {task}: part {part} done", flush=True)


def coverage_map():
    """coverage_scan's counts, summed over partitions. cached in COVERAGE_DIR/total.npz: delete
    it after re-running coverage-scan"""
    path = COVERAGE_DIR / "total.npz"
    if path.exists():
        return dict(np.load(path))
    npix = healpy.nside2npix(2**MAP_LEVEL)
    out = {k: np.zeros(npix, dtype=np.int64) for k in ["n_all", "n_used", "n_nonpos"]}
    for f in COVERAGE_DIR.glob("part=*.npz"):
        d = np.load(f)
        for k in out:
            np.add.at(out[k], d["pix"], d[k])
    np.savez(path, **out)
    return out


def coverage_map_rough():
    """quick stand-in for coverage_map from the runs' outputs: per level-MAP_LEVEL pixel, the
    number of analysis bins with any measurement that passed read_partition's cuts (a bin
    is in windows/ iff it has one). exact footprint, but no depth. cached in
    COVERAGE_DIR/rough.npz"""
    path = COVERAGE_DIR / "rough.npz"
    if path.exists():
        return dict(np.load(path))
    import duckdb

    files = [str(RUN_DIR / run / "windows/*.parquet") for run in RUNS.values()]
    res = duckdb.sql(f"""
    SELECT hpbin >> {2 * (BIN_LEVEL - MAP_LEVEL)} AS pix, count(DISTINCT hpbin) AS n_bins
    FROM read_parquet({files})
    GROUP BY pix
    """).fetchnumpy()
    out = {"n_bins": np.zeros(healpy.nside2npix(2**MAP_LEVEL), dtype=np.int64)}
    out["n_bins"][res["pix"]] = res["n_bins"]
    COVERAGE_DIR.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


def circle(ra, dec, radius, n=361):
    """(ra, dec) [deg] of the circle of `radius` [deg] around (ra, dec)"""
    c = healpy.ang2vec(ra, dec, lonlat=True)
    e1 = np.cross([0, 0, 1], c)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    phi, r = np.linspace(0, 2 * np.pi, n)[:, None], np.radians(radius)
    v = np.cos(r) * c + np.sin(r) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    return healpy.vec2ang(v, lonlat=True)


def zoom_graticule(ax, dpar, dmer, side="left", fontsize="small"):
    """a zoom's (gnomview's) graticule at multiples of dpar, dmer [deg], labeled where its lines
    leave the axes: latitudes on `side`, longitudes on the bottom (healpy axes have no ticks,
    and healpy's own gnomview graticule is not at round coordinates)"""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    t = np.linspace(-180, 180, 3601)
    for b in np.arange(-90 + dpar, 90, dpar):
        x, y = ax.proj.ang2xy(t, np.full_like(t, b), lonlat=True)
        inside = np.isfinite(x) & (x > x0) & (x < x1) & (y > y0) & (y < y1)
        if inside.any():
            ax.plot(np.where(inside, x, np.nan), y, color="0.5", lw=0.4, ls=":")
            k = np.flatnonzero(inside)[
                np.argmin(x[inside]) if side == "left" else np.argmax(x[inside])
            ]
            ax.text(
                x0 - 0.005 if side == "left" else x1 + 0.005,
                y[k],
                f"${b:+.0f}^\\circ$".replace("+0", "0"),
                ha="right" if side == "left" else "left",
                va="center",
                fontsize=fontsize,
                clip_on=False,
            )
    t = np.linspace(-89.9, 89.9, 1799)
    for l in np.arange(0, 360, dmer):
        x, y = ax.proj.ang2xy(np.full_like(t, l), t, lonlat=True)
        inside = np.isfinite(y) & (x > x0) & (x < x1) & (y > y0) & (y < y1)
        if inside.any():
            ax.plot(x, np.where(inside, y, np.nan), color="0.5", lw=0.4, ls=":")
            k = np.flatnonzero(inside)[np.argmin(y[inside])]
            ax.text(
                x[k],
                y0 - 0.005,
                f"${l:.0f}^\\circ$",
                ha="center",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )
    ax.text(0.5, -0.13, r"$\ell$", transform=ax.transAxes, ha="center", va="top")
    ax.text(
        -0.3 if side == "left" else 1.3,
        0.5,
        r"$b$",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


# full-sky test versions: everything outside the runs' pixels, at this opacity
GHOST_ALPHA = 0.35


def plot_coverage(rough=False, ghost=None):
    """ghost (full-sky test versions, scripts/claude/coverage_fullsky.py's map as the background):
    "sky" on the full sky only, "all" in the zooms too"""
    if rough:
        # fraction of each pixel's analysis bins that the analysis used
        counts = coverage_map_rough()["n_bins"]
        unit, out_name = "fraction of bins used", "coverage-rough"
    else:
        cov = coverage_map()
        counts = cov["n_used"]
        print(
            f"rows used: {counts.sum() / cov['n_all'].sum():.3f} of all; lost only to flux <= 0: "
            f"{cov['n_nonpos'].sum() / cov['n_all'].sum():.2e}"
        )
        # per analysis bin (level 15: sqrt(4 pi / (12 4^15)) = 6.44 arcsec), so the color bar reads
        # as "measurements in one spectrum"
        unit, out_name = r"measurements per $6.44^{\prime\prime}$ bin", "coverage"
    per_bin = counts / 4 ** (BIN_LEVEL - MAP_LEVEL)
    m = np.where(counts > 0, per_bin, healpy.UNSEEN)
    alpha = np.ones(len(m))
    if ghost:
        # all rows, so scaled by the fraction that pass the cuts in the runs' own pixels (which
        # is only that: it varies over the sky). approximate, like the footers' counts
        n_all = np.load(COVERAGE_DIR / "fullsky-footers.npz")["n_all"]
        f = counts.sum() / n_all[counts > 0].sum()
        print(f"used / footer rows in the runs' pixels: {f:.3f}")
        outside = (counts == 0) & (n_all > 0)
        m[outside] = f * n_all[outside] / 4 ** (BIN_LEVEL - MAP_LEVEL)
        alpha[outside] = GHOST_ALPHA
        print(f"median per bin outside: {np.median(m[outside]):.0f}; inside: {np.median(per_bin[counts > 0]):.0f}")
        out_name += f"-ghost-{ghost}"

    # full: the color floor is 100, not the minimum, which is ~5e-4 from map pixels that only
    # clip a partition's corner. the sparser pixels (<1% of them) show at the floor color
    vmin, vmax = (0, 1) if rough else (100, per_bin.max())

    # to galactic: each galactic pixel takes the value of the equatorial pixel at its center
    # (the map is equatorial, talltable's healpix). then nothing below rotates coordinates
    nside = 2**MAP_LEVEL
    l, b = healpy.pix2ang(nside, np.arange(len(m)), nest=True, lonlat=True)
    ra, dec = healpy.Rotator(coord=["G", "C"])(l, b, lonlat=True)
    from_eq = healpy.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    m, alpha = m[from_eq], alpha[from_eq]
    to_gal = healpy.Rotator(coord=["C", "G"])
    kw = dict(
        nest=True,
        norm=None if rough else "log",
        min=vmin,
        max=vmax,
        cbar=False,
        notext=True,
        badcolor="0.92",
    )

    # full sky in the middle, the two fields' zooms either side: NDF (l ~ 96 deg) is on the
    # left of a galactic Mollweide, SDF (l ~ 270 deg) on the right
    fig = plt.figure(figsize=(FIG_WIDTH, 2.55))
    healpy.mollview(m, title="", fig=fig.number, sub=(1, 3, 2), alpha=alpha if ghost else None, **kw)
    ax_sky = plt.gca()
    # graticules per axes: healpy.graticule draws on every healpy axes in the figure
    ax_sky.graticule(dpar=30, dmer=60, alpha=0.3)
    ax_sky.projtext(
        0, 0, "Galactic equator", lonlat=True, ha="center", va="top", fontsize="small", color="0.3",
        transform=offset_copy(ax_sky.transData, fig=fig, y=-1.5, units="points"),
    )
    ax_sky.set_position([0.295, 0.34, 0.41, 0.64])
    for (name, run), position, side in zip(
        RUNS.items(), [[0.075, 0.2, 0.2, 0.2 * FIG_WIDTH / 2.55], [0.725, 0.2, 0.2, 0.2 * FIG_WIDTH / 2.55]], [1, 0]
    ):
        c = config(run)
        ring_l, ring_b = to_gal(
            *circle(c["center_ra_deg"], c["center_dec_deg"], c["radius_deg"]),
            lonlat=True,
        )
        center = to_gal(c["center_ra_deg"], c["center_dec_deg"], lonlat=True)

        # the field on the sky
        ax_sky.projplot(ring_l, ring_b, lonlat=True, color="k", lw=0.6)

        # its zoom (20 deg across); the disc two_sided.py selected partitions with
        healpy.gnomview(
            np.where(alpha == 1, m, healpy.UNSEEN) if ghost == "sky" else m,
            alpha=alpha if ghost == "all" else None,
            rot=center,
            xsize=800,
            reso=1.5,
            title=FIELD_NAMES[name],
            fig=fig.number,
            sub=(1, 3, 1 if side else 3),
            **kw,
        )
        ax = plt.gca()
        ax.title.set_fontsize("medium")
        ax.projplot(ring_l, ring_b, lonlat=True, color="k", lw=0.8)
        ax.set_position(position)
        zoom_graticule(ax, 5, 10, side="left" if side else "right")
        # healpy turns the axes frame off; a border in the connecting lines' style
        ax.add_patch(
            Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, color="0.4", lw=0.6, clip_on=False)
        )

        # the zoom's footprint on the sky: its border, taken back through the gnomonic projection
        # (each edge is a great-circle arc, not straight on the Mollweide)
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        t = np.linspace(0, 1, 50)
        edge_x = np.concatenate([x0 + (x1 - x0) * t, np.full(50, x1), x1 - (x1 - x0) * t, np.full(50, x0)])
        edge_y = np.concatenate([np.full(50, y0), y0 + (y1 - y0) * t, np.full(50, y1), y1 - (y1 - y0) * t])
        box_l, box_b = ax.proj.xy2ang(edge_x, edge_y, lonlat=True)
        ax_sky.projplot(box_l, box_b, lonlat=True, color="0.4", lw=0.6)
        # x, y = ax_sky.proj.ang2xy(box_l, box_b, lonlat=True)
        # ax_sky.text(np.mean(x), np.max(y) + 0.04, name, ha="center", va="bottom", fontsize="small")

        # lines from the footprint's near corners to the same corners of the zoom
        for corner_y in [0, 1]:
            corner_l, corner_b = ax.proj.xy2ang(x1 if side else x0, y1 if corner_y else y0, lonlat=True)
            fig.add_artist(
                ConnectionPatch(
                    xyA=ax_sky.proj.ang2xy(corner_l, corner_b, lonlat=True),
                    coordsA=ax_sky.transData,
                    xyB=(side, corner_y),
                    coordsB=ax.transAxes,
                    color="0.4",
                    lw=0.6,
                )
            )

    cax = fig.add_axes([0.355, 0.235, 0.29, 0.04])
    # healpy's "log" norm logs the data itself, so a colorbar of its image has linear ticks;
    # build the bar from a matplotlib norm instead
    norm = Normalize(vmin, vmax) if rough else LogNorm(vmin, vmax)
    # the map's colormap (healpy uses image.cmap), fresh: healpy's own copy has the background as
    # its under color, while its map clips to the floor color
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.rcParams["image.cmap"])
    fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        label=unit,
        extend="neither" if rough else "min",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{out_name}.png", dpi=200)
    fig.savefig(FIG_DIR / f"{out_name}.pdf", dpi=200)


# --- sample spectrum ---

# the analysis bin to show. not the field center: deep-field visits return the center to
# nearly the same detector pixels, so its samples cluster at one phase within each channel;
# 8' out, field rotation spreads them evenly. this bin, 8' from the NDF center at position
# angle 60 deg, has no known-source flags and the floor median flux - zodi of a 36-bin scan
SPECTRUM_RUN = RUNS["NDF"]
HPBIN = 4205503889  # level-BIN_LEVEL nested index; None = the field center
# show the images of the measurements nearest these wavelengths [um]
HIGHLIGHT = [1.5, 3.5]
CUTOUT = 15  # half-width of the image cutouts [pixels]
LINTHRESH = 0.1  # [MJy/sr]: the flux axis is linear within +-LINTHRESH, log outside


def read_bin(hpbin):
    """every measurement in one analysis bin, sorted by wavelength"""
    (part,) = find_partitions_ipix(hpbin, level=BIN_LEVEL)
    shift = 2 * (HP_HIGH_LEVEL - BIN_LEVEL)
    table = pq.read_table(
        PIXEL_DB_PATH / f"part={part}/compacted.parquet",
        filters=[
            ("hphigh", ">=", hpbin << shift),
            ("hphigh", "<", (hpbin + 1) << shift),
        ],
    )
    rows = {c: table.column(c).to_numpy() for c in table.column_names}
    # the same row cuts as two_sided.read_partition (with ALLOWED_BITS = 0)
    keep = (rows["flags"] == 0) & (rows["flux"] > 0) & (rows["variance"] > 0)
    keep &= (
        np.isfinite(rows["flux"])
        & np.isfinite(rows["variance"])
        & np.isfinite(rows["zodi"])
    )
    rows = {k: v[keep] for k, v in rows.items()}

    waves = pq.read_table(WAVES_DB_PATH, columns=["waveid", "wavelength"])
    wid, wl = waves.column("waveid").to_numpy(), waves.column("wavelength").to_numpy()
    rows["wavelength"] = wl[np.searchsorted(wid, rows["waveid"])]
    rows["row"], rows["col"], rows["det"] = waveid_to_rowcoldet(rows["waveid"])
    order = np.argsort(rows["wavelength"])
    return {k: v[order] for k, v in rows.items()}


def image_path(imageid):
    t = pq.read_table(
        IMAGE_DB_PATH,
        columns=["imageid", "filepath"],
        filters=[("imageid", "==", imageid)],
    )
    return t.column("filepath")[0].as_py()


def plot_spectrum():
    c = config(SPECTRUM_RUN)
    hpbin = HPBIN
    if hpbin is None:
        hpbin = healpy.ang2pix(
            2**BIN_LEVEL,
            c["center_ra_deg"],
            c["center_dec_deg"],
            nest=True,
            lonlat=True,
        )
    rows = read_bin(hpbin)
    print(f"bin {hpbin}: {len(rows['flux'])} measurements")
    d = (
        rows["flux"] - rows["zodi"]
    )  # what the continuum fit sees (two_sided.read_partition)

    fig = plt.figure(figsize=(FIG_WIDTH, 5.4))
    # bottom row: per highlighted measurement, its whole image and a zoom on the pixel
    gs = fig.add_gridspec(2, 2 * len(HIGHLIGHT), height_ratios=[1.6, 1])
    ax = fig.add_subplot(gs[0, :])
    for det in range(1, 7):
        s = rows["det"] == det
        eb = ax.errorbar(
            rows["wavelength"][s],
            d[s],
            yerr=np.sqrt(rows["variance"][s]),
            fmt=".",
            ms=2,
            lw=0.5,
            color=f"C{det - 1}",
            label=f"D{det}",
        )
        # ~29k points and error bars: raster them in the PDF (it otherwise renders slowly);
        # axes, text, and the highlighted measurements below stay vector
        for artist in eb.get_children():
            artist.set_rasterized(True)
    # log in lambda; symlog in flux - zodi, which is <= 0 for 11% of the NEP bin's measurements
    # (~27% in D5, D6), so a log axis would drop them
    ax.set_xscale("log")
    ax.set_xticks([0.75, 1, 1.5, 2, 3, 4, 5], ["0.75", "1", "1.5", "2", "3", "4", "5"])
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    ax.set_ylim(top=20)  # room for the legend
    # the bin's center [deg], for the title
    ra, dec = healpy.pix2ang(2**BIN_LEVEL, hpbin, nest=True, lonlat=True)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel("flux $-$ zodi [MJy/sr]")
    ax.set_title(
        rf"RA = {ra:.3f}$^\circ$, Dec = {dec:+.3f}$^\circ$",
        fontsize="medium",
    )
    ax.legend(fontsize="small", ncols=6, loc="upper right", frameon=False, columnspacing=1, handletextpad=0.2)

    for j, lam in enumerate(HIGHLIGHT):
        i = np.argmin(np.abs(rows["wavelength"] - lam))
        r, col = rows["row"][i], rows["col"][i]
        path = image_path(rows["imageid"][i])
        # qr2 images: the talltable fluxes have been recalibrated (talltable qr2_to_qr3.parquet),
        # so the image's pixel value can differ slightly from the spectrum's
        image = fits.getdata(path, "IMAGE") - fits.getdata(path, "ZODI")
        sub, zoom = fig.add_subplot(gs[1, 2 * j]), fig.add_subplot(gs[1, 2 * j + 1])
        # each panel clipped to its own 1-99 percentiles: the zoom's are the cutout's
        cut = image[r - CUTOUT : r + CUTOUT + 1, col - CUTOUT : col + CUTOUT + 1]
        for a, data in [(sub, image), (zoom, cut)]:
            vmin, vmax = np.nanpercentile(data, [1, 99])
            a.imshow(image, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
            a.set_xlabel("column", fontsize="small", labelpad=1)
            a.set_ylabel("row", fontsize="small", labelpad=1)
            a.tick_params(labelsize="small")
            a.locator_params(nbins=3)
        zoom.set_xlim(col - CUTOUT - 0.5, col + CUTOUT + 0.5)
        zoom.set_ylim(r - CUTOUT - 0.5, r + CUTOUT + 0.5)
        # the cutout on the full image, and lines from its left corners to the zoom's
        # (indicate_inset_zoom picks its own corners)
        x0, y0 = col - CUTOUT - 0.5, r - CUTOUT - 0.5
        sub.add_patch(Rectangle((x0, y0), 2 * CUTOUT + 1, 2 * CUTOUT + 1, fill=False, color="k", lw=0.6))
        for corner_y in [0, 1]:
            fig.add_artist(
                ConnectionPatch(
                    xyA=(x0, y0 + corner_y * (2 * CUTOUT + 1)),
                    coordsA=sub.transData,
                    xyB=(0, corner_y),
                    coordsB=zoom.transAxes,
                    color="#666",
                    lw=0.6,
                )
            )
        # the pixel itself (pixel centers are at integer row, col)
        zoom.add_patch(Rectangle((col - 0.5, r - 0.5), 1, 1, fill=False, color="k", lw=0.8))
                # one title for both panels, from the left so the arrow out of the zoom misses it
        sub.set_title(
            f"D{rows['det'][i]}, {Path(path).name[7:24]}\n"
            f"pixel ({r}, {col}), {rows['wavelength'][i]:.3f} $\\mu$m",
            fontsize="small", loc="left", multialignment="left",
        )
        # the measurement redrawn on top: its marker in its detector's color, outlined in
        # black, and its error bar a capped black line as thick as the outline
        ax.errorbar(
            rows["wavelength"][i],
            d[i],
            yerr=np.sqrt(rows["variance"][i]),
            fmt="o",
            ms=4,
            mfc=f"C{rows['det'][i] - 1}",
            mec="k",
            mew=0.8,
            ecolor="k",
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            zorder=5,
        )
        # an arrow from the pixel's square up to the measurement in the spectrum; the
        # shrinkB (half the marker size, plus a bit) stops it at the outlined marker's edge
        fig.add_artist(
            ConnectionPatch(
                xyA=(col, r + 0.5),  # the pixel's top edge
                coordsA=zoom.transData,
                xyB=(rows["wavelength"][i], d[i]),
                coordsB=ax.transData,
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                shrinkA=0,
                shrinkB=3,
                color="#666",
                lw=0.8,
            )
        )

    fig.tight_layout(w_pad=0.6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "spectrum.png", dpi=200)
    fig.savefig(FIG_DIR / "spectrum.pdf", dpi=300)  # the rastered crowd's resolution


if __name__ == "__main__":
    {
        "coverage-scan": coverage_scan,
        "coverage": plot_coverage,
        "coverage-rough": lambda: plot_coverage(rough=True),
        "coverage-ghost-sky": lambda: plot_coverage(ghost="sky"),
        "coverage-ghost-all": lambda: plot_coverage(ghost="all"),
        "spectrum": plot_spectrum,
    }[sys.argv[1]]()
