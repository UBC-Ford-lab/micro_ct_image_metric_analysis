"""One volume, one command, one figure: ``ct-report``.

Runs every metric an ROI table defines on a volume, writes the curves
(``<kind>.npz`` + ``.json``), a combined ``metrics.json`` and a three-panel
figure (MTF with its 50 % and 10 % crossings, NPS with its peak and the
noise sigma, d' against disc diameter with the detectable size at the Rose
criterion). ``--metrics metrics.json`` redraws the figure from a saved run.

    ct-report VOLUME --rois rois.json --out results/
    ct-report --metrics results/metrics.json --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .api import load_metrics, run_metrics, summary_table
from .rois import PhantomROIs
from .volumes import load_volume

FIG_W_IN = 886.4 / 72.0
FIG_H_IN = 307.770625 / 72.0
ACCENT = '#0072B2'          # Okabe-Ito blue
ANNOT_GREY = '0.5'
RC = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'grid.linewidth': 0.5,
    'grid.color': '#b0b0b0',
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'figure.dpi': 150,
    'pdf.fonttype': 42,     # TrueType, so the PDF passes a publisher's font check
    'ps.fonttype': 42,
}


def iq_figure(metrics: dict, out_stem, *, title=None, fmax=None, dmax=2.0, dpi=300,
              panel_labels=False, formats=('png', 'pdf')) -> list:
    """The three-panel figure from a ``run_metrics`` result.

    Panels present depend on what was measured: MTF, NPS and d' when all
    three exist, fewer otherwise. Returns the written paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    res = metrics['results']
    sc = metrics.get('scalars', {})
    panels = [k for k in ('mtf', 'nps', 'd_prime') if k in res]
    if not panels:
        raise ValueError("nothing to draw: the run has no MTF, NPS or d' result")

    with plt.rc_context(RC):
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(FIG_W_IN * n / 3, FIG_H_IN), squeeze=False)
        axes = axes[0]
        hi = fmax
        if 'mtf' in res:
            f, v = res['mtf'].x, res['mtf'].y
            v = v / max(v[0], 1e-12)
            if hi is None:
                below = f[v < 0.05]
                hi = float(np.ceil((below[0] if below.size else f[-1]) * 1.25 * 2) / 2)
        elif 'nps' in res:
            hi = hi or float(res['nps'].x[-1])

        for ax, what in zip(axes, panels):
            if what == 'mtf':
                ax.plot(f, v, color=ACCENT)
                for lev, key, name in ((0.5, 'mtf50', 'MTF$_{50}$'), (0.1, 'mtf10', 'MTF$_{10}$')):
                    xc = float(sc.get(key, np.nan))
                    if not np.isfinite(xc):
                        continue
                    ax.axhline(lev, color='0.55', lw=0.5, ls=(0, (2, 2)), zorder=0)
                    ax.plot([xc, xc], [0, lev], color='0.55', lw=0.5, ls=(0, (2, 2)), zorder=0)
                    ax.annotate(f'{name} = {xc:.2f} lp/mm', xy=(xc, lev), xytext=(6, 6),
                                textcoords='offset points', fontsize=6.5, color='0.35')
                ax.set_xlim(0, hi); ax.set_ylim(0, 1.04)
                ax.set_xlabel('Spatial frequency (lp/mm)'); ax.set_ylabel('MTF')
                ax.set_title('MTF')
            elif what == 'nps':
                fn, vn = res['nps'].x, res['nps'].y
                pos = vn > 0
                ax.semilogy(fn[pos], vn[pos], color=ACCENT)
                fpk = float(sc.get('nps_f_peak', np.nan))
                if np.isfinite(fpk):
                    ax.axvline(fpk, color='0.55', lw=0.5, ls=(0, (2, 2)), zorder=0)
                    ax.annotate(f'$f_{{peak}}$ = {fpk:.2f} lp/mm', xy=(fpk, vn.max()),
                                xytext=(5, -2), textcoords='offset points', fontsize=6.5,
                                color='0.35')
                ax.set_xlim(0, hi)
                ax.set_xlabel('Spatial frequency (lp/mm)')
                ax.set_ylabel('NPS (HU$^2$·mm$^2$)')
                ax.set_title('NPS')
                sig = sc.get('noise_std_hu', sc.get('noise_std'))
                if sig is not None:
                    ax.text(0.975, 0.045, f"σ = {sig:.0f} HU", transform=ax.transAxes,
                            fontsize=6.3, ha='right', va='bottom',
                            bbox=dict(boxstyle='square,pad=0.35', fc='white', ec='0.6', lw=0.6))
            else:
                dp = res['d_prime']
                thr = float(dp.summary.get('rose_threshold', 3.0))
                xd = np.asarray(dp.extra.get('curve_diameters_mm', dp.x), float)
                yd = np.asarray(dp.extra.get('curve_d_prime', dp.y), float)
                x_hi = float(dmax) if dmax else float(xd[-1])
                ax.plot(xd, yd, color=ACCENT)
                ax.axhline(thr, color=ANNOT_GREY, lw=1.1, ls=':', zorder=0)
                dc = float(dp.summary.get('detectable_size_mm', np.nan))
                if np.isfinite(dc):
                    ax.plot([dc, dc], [0, thr], color=ANNOT_GREY, lw=1.1, ls=':', zorder=0)
                    ax.annotate(f'{dc:.2f} mm', xy=(dc, thr), xytext=(-6, 6),
                                textcoords='offset points', fontsize=6.5, color='0.35',
                                ha='right')
                ax.set_xlim(0, x_hi)
                top = float(np.nanmax(yd[xd <= x_hi])) if np.any(xd <= x_hi) else float(np.nanmax(yd))
                ax.set_ylim(0, max(top, thr) * 1.08)
                ax.text(x_hi * 0.985, thr, f"Rose criterion (d'={thr:g})", color=ANNOT_GREY,
                        fontsize=6.3, ha='right', va='bottom')
                ax.set_xlabel('Disc diameter (mm)'); ax.set_ylabel("d' (NPW observer)")
                ax.set_title("Detectability (d')")
                ax.text(0.975, 0.045, f"ΔC = {dp.summary.get('contrast_hu', 0):g} HU",
                        transform=ax.transAxes, fontsize=6.3, ha='right', va='bottom',
                        bbox=dict(boxstyle='square,pad=0.35', fc='white', ec='0.6', lw=0.6))

        for tag, ax in zip('abc', axes):
            ax.grid(True, which='major'); ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if panel_labels:
                ax.text(-0.26, 1.06, f'({tag})', transform=ax.transAxes, fontsize=9,
                        fontweight='bold', va='bottom', ha='left')
        if title:
            fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        out = Path(out_stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for ext in formats:
            p = out.with_suffix('.' + ext)
            fig.savefig(p, dpi=dpi if ext == 'png' else None)
            written.append(p)
        plt.close(fig)
    return written


def run_report(volume_path, rois_path, out_dir, *, pixel_size=None, slice_thickness=None,
               flip=(), plots=False, verbose=True, title=None, fmax=None, dmax=2.0,
               dpi=300) -> dict:
    """Load a volume, run every metric in the ROI table, save everything, draw the figure."""
    image, frame, info = load_volume(volume_path, pixel_size=pixel_size,
                                     slice_thickness=slice_thickness, flip=flip)
    rois = PhantomROIs.load(rois_path)
    if verbose:
        print(f"{volume_path}: {info['shape_zyx']} @ {frame.dx:g} mm "
              f"(voxel size from {info['source']})")
    metrics = run_metrics(image, frame, rois, out_dir=out_dir, plots=plots, verbose=verbose)
    metrics['volume'] = str(volume_path)
    metrics['rois'] = str(rois_path)
    metrics['info'] = info
    from .api import save_metrics
    save_metrics(metrics, Path(out_dir) / 'metrics.json')
    written = iq_figure(metrics, Path(out_dir) / 'iq_report',
                        title=title or Path(volume_path).name, fmax=fmax, dmax=dmax, dpi=dpi)
    metrics['figure'] = [str(p) for p in written]
    if verbose:
        print(summary_table(metrics))
        print(f"  written: {Path(out_dir) / 'metrics.json'}, " + ", ".join(map(str, written)))
    return metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog='ct-report',
        description="Run every metric an ROI table defines on one volume and draw the "
                    "MTF / NPS / d' figure.")
    ap.add_argument('volume', nargs='?', help='volume (.npy, .npz or .vff)')
    ap.add_argument('--rois', help='ROI table in mm (from ct-find-rois or by hand)')
    ap.add_argument('--metrics', help='redraw the figure from a saved metrics.json instead')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--pixel-size', type=float, default=None, help='voxel size in mm')
    ap.add_argument('--slice-thickness', type=float, default=None)
    ap.add_argument('--flip', nargs='+', default=(), choices=('x', 'y', 'z'))
    ap.add_argument('--plots', action='store_true',
                    help="also save each calculator's diagnostic figure")
    ap.add_argument('--title', default=None)
    ap.add_argument('--fmax', type=float, default=None, help='frequency axis limit (lp/mm)')
    ap.add_argument('--dmax', type=float, default=2.0, help="d' panel diameter limit (mm)")
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.metrics:
        metrics = load_metrics(args.metrics)
        written = iq_figure(metrics, Path(args.out) / 'iq_report', title=args.title,
                            fmax=args.fmax, dmax=args.dmax, dpi=args.dpi)
        if not args.quiet:
            print(summary_table(metrics))
            print("  written: " + ", ".join(map(str, written)))
        return
    if not (args.volume and args.rois):
        raise SystemExit("ct-report needs VOLUME and --rois (or --metrics metrics.json)")
    run_report(args.volume, args.rois, args.out, pixel_size=args.pixel_size,
               slice_thickness=args.slice_thickness, flip=args.flip, plots=args.plots,
               verbose=not args.quiet, title=args.title, fmax=args.fmax, dmax=args.dmax,
               dpi=args.dpi)


if __name__ == '__main__':
    main()
