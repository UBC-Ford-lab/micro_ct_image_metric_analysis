"""The high-level API: one function per metric, one result type.

Each function wraps the corresponding calculator, keeps only the
non-negative half of its frequency axis, and attaches the summary numbers a
table wants. Nothing is plotted or printed unless asked. The calculators
themselves (``get_MTF`` and friends) stay available for anyone who wants
the raw two-sided curves.

Regions are pixel indices on the image passed in; use :mod:`rois` to derive
them from millimetres, or :func:`run_metrics` to do the whole thing from an
ROI table.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import numpy as np

from . import d_prime_calculator as _dp
from . import mtf_calculator as _mtf
from . import neq_calculator as _neq
from . import nps_calculator as _nps
from . import ttf_calculator as _ttf
from .results import MetricResult, crossing, nps_summary, positive_half, trapezoid
from .rois import Frame, PhantomROIs


def _target(plot_dir):
    if plot_dir is None:
        return os.getcwd(), False
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    return str(plot_dir), True


def _select(image, slices):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[np.newaxis]
    if slices is not None:
        image = image[slices]
    return image


# --------------------------------------------------------------------------

def mtf(image, crop_indices, pixel_size, *, edge_angle, high_to_low=True,
        slices=None, plot_dir=None, process_lsf=True) -> MetricResult:
    """Slanted-edge MTF of a ``(z, y, x)`` image.

    :param crop_indices: ``[y1, y2, x1, x2]`` around the edge; the height
        should be twice the width.
    :param pixel_size: mm.
    :param edge_angle: degrees; sign says which way the edge leans.
    :param high_to_low: True when intensity drops with increasing x.
    :param slices: z selection (slice, index array or None).
    :param plot_dir: save the calculator's diagnostic figure there.
    """
    data = _select(image, slices)
    target, plot = _target(plot_dir)
    f, m = _mtf.get_MTF(data, list(crop_indices), find_absolute_MTF=True,
                        pixel_size=float(pixel_size), target_directory=target,
                        plot_results=plot, edge_angle=float(edge_angle),
                        high_to_low=bool(high_to_low), process_LSF=process_lsf)
    f, m = positive_half(f, m)
    m = m / m[0] if m.size and m[0] > 0 else m
    return MetricResult(
        kind='mtf', x=f, y=m,
        summary={'mtf50': crossing(f, m, 0.5), 'mtf10': crossing(f, m, 0.1),
                 'nyquist': float(0.5 / pixel_size)},
        meta={'pixel_size_mm': float(pixel_size), 'crop_indices': list(crop_indices),
              'edge_angle_deg': float(edge_angle), 'high_to_low': bool(high_to_low),
              'n_slices': int(data.shape[0])})


def nps(image, roi_bounds, pixel_size, *, slices=None, plot_dir=None) -> MetricResult:
    """Noise power spectrum from square ROIs of a ``(z, y, x)`` image.

    :param roi_bounds: ``[[y1, y2, x1, x2], ...]``, all the same square size.
    """
    data = _select(image, slices)
    rois = np.asarray(roi_bounds, dtype=int)
    target, plot = _target(plot_dir)
    f, p = _nps.get_NPS(data, rois, pixel_size=float(pixel_size),
                        target_directory=target, plot_results=plot)
    f, p = positive_half(f, p)
    stds = [float(data[:, b[0]:b[1], b[2]:b[3]].std()) for b in rois]
    means = [float(data[:, b[0]:b[1], b[2]:b[3]].mean()) for b in rois]
    summary = nps_summary(f, p)
    summary['noise_std'] = float(np.mean(stds))
    summary['roi_mean'] = float(np.mean(means))
    summary['nyquist'] = float(0.5 / pixel_size)
    return MetricResult(
        kind='nps', x=f, y=p, summary=summary,
        meta={'pixel_size_mm': float(pixel_size), 'roi_bounds': rois.tolist(),
              'n_rois': int(len(rois)), 'roi_size_px': int(rois[0][1] - rois[0][0]),
              'n_slices': int(data.shape[0])})


def neq(mtf_result: MetricResult, nps_result: MetricResult) -> MetricResult:
    """NEQ = MTF^2 / NPS on the overlap of the two frequency axes."""
    f, v = _neq.neq_from_curves(mtf_result.x, mtf_result.y, nps_result.x, nps_result.y)
    good = np.isfinite(v)
    total = float(trapezoid(v[good], f[good])) if good.sum() > 1 else float('nan')
    return MetricResult(kind='neq', x=f, y=v, summary={'neq_total': total},
                        meta={'from': ['mtf', 'nps']})


def ttf(image, centre_pixels, radius, pixel_size, *, materials=None, slices=None,
        plot_dir=None, cnr_threshold=None, process_lsf=True) -> MetricResult:
    """Task transfer function of circular inserts, one curve per insert.

    :param centre_pixels: ``[[y, x], ...]``.
    :param radius: analysis radius in pixels (crop is ``2*radius`` square).
    :param cnr_threshold: inserts below it get a NaN TTF50 (default: the
        calculator's ``DEFAULT_CNR_THRESHOLD``).
    """
    data = _select(image, slices)
    centres = [list(c) for c in centre_pixels]
    names = list(materials) if materials else [f"insert {i+1}" for i in range(len(centres))]
    floor = float(_ttf.DEFAULT_CNR_THRESHOLD if cnr_threshold is None else cnr_threshold)
    target, plot = _target(plot_dir)
    with contextlib.redirect_stdout(io.StringIO()):   # progress prints
        f, arr, cnr = _ttf.get_TTF(data, centres, int(radius), materials=names,
                                   find_absolute_TTF=True, pixel_size=float(pixel_size),
                                   target_directory=target, plot_results=plot,
                                   process_LSF=process_lsf, cnr_threshold=floor)
    f, arr = positive_half(f, np.asarray(arr))
    cnr = np.asarray(cnr, dtype=float).ravel()
    summary = {'cnr_floor': floor}
    usable = []
    for name, curve, c in zip(names, arr, cnr):
        key = name.lower().replace(' ', '_')
        summary[f'cnr_{key}'] = float(c)
        if c > floor:
            v = crossing(f, curve, 0.5)
            usable.append(v)
        else:
            v = float('nan')
        summary[f'ttf50_{key}'] = v
    summary['ttf50_mean'] = float(np.nanmean(usable)) if usable else float('nan')
    summary['ttf50_n_used'] = len(usable)
    return MetricResult(
        kind='ttf', x=f, y=arr, labels=names, summary=summary,
        meta={'pixel_size_mm': float(pixel_size), 'centre_pixels': centres,
              'radius_px': int(radius), 'n_slices': int(data.shape[0]),
              'cnr': cnr.tolist()})


def detectability(mtf_result: MetricResult, nps_result: MetricResult, *,
                  disc_diameters_mm=None, contrast_hu=None, rose_threshold=None,
                  plot_dir=None) -> MetricResult:
    """NPW-observer d' for disc tasks, from the MTF and NPS curves.

    ``x`` holds the requested disc diameters and ``y`` d' at each. The dense
    curve used to read off the detectable size is in ``extra``.
    """
    target, plot = _target(plot_dir)
    thr = float(_dp.ROSE_THRESHOLD if rose_threshold is None else rose_threshold)
    with contextlib.redirect_stdout(io.StringIO()):
        r = _dp.get_d_prime_npw(mtf_result.x, mtf_result.y, nps_result.x, nps_result.y,
                                disc_diameters_mm=disc_diameters_mm, contrast_hu=contrast_hu,
                                rose_threshold=thr, plot_results=plot, target_directory=target)
    d = np.asarray(r['disc_diameters_mm'], float)
    v = np.asarray(r['d_prime'], float)
    summary = {f"d_prime_{float(dd):g}mm": float(vv) for dd, vv in zip(d, v)}
    summary['detectable_size_mm'] = float(r['diameter_at_threshold_mm'])
    summary['rose_threshold'] = thr
    summary['contrast_hu'] = float(r['contrast_hu'])
    return MetricResult(
        kind='d_prime', x=d, y=v, summary=summary,
        meta={'contrast_hu': float(r['contrast_hu']), 'rose_threshold': thr},
        extra={'curve_diameters_mm': np.asarray(r['curve_diameters_mm']),
               'curve_d_prime': np.asarray(r['curve_d_prime'])})


# --------------------------------------------------------------------------
# All of it from an ROI table
# --------------------------------------------------------------------------

def run_metrics(image, frame: Frame, rois: PhantomROIs, *, out_dir=None,
                plots: bool = False, verbose: bool = False) -> dict:
    """Every metric the ROI table defines, on one ``(z, y, x)`` image.

    Returns ``{'results': {kind: MetricResult}, 'scalars': {...},
    'skipped': [(what, why), ...], 'pixel_size_mm', 'slice_thickness_mm'}``.
    Anything the table asks for that this volume cannot supply is listed in
    ``skipped`` rather than silently missing. With ``out_dir`` every result
    is saved as ``<kind>.npz`` + ``<kind>.json`` and, if ``plots``, the
    calculators' figures go there too.
    """
    image = frame.check_image(image)
    resolved = rois.resolve(frame)
    plot_dir = str(out_dir) if (plots and out_dir is not None) else None
    say = print if verbose else (lambda *a, **k: None)
    out = {'pixel_size_mm': frame.dx, 'slice_thickness_mm': frame.dz,
           'skipped': list(resolved['skipped']), 'results': {}, 'scalars': {}}
    res = out['results']

    if 'mtf' in resolved:
        s = resolved['mtf']
        say(f"  MTF : {s['slices'].stop - s['slices'].start} slices, crop "
            f"{s['crop_indices']}, edge {s['edge_angle']:.2f} deg")
        res['mtf'] = mtf(image, s['crop_indices'], frame.dx, edge_angle=s['edge_angle'],
                         high_to_low=s['high_to_low'], slices=s['slices'], plot_dir=plot_dir)
        res['mtf'].meta['z_slices'] = [s['slices'].start, s['slices'].stop]

    if 'nps' in resolved:
        s = resolved['nps']
        say(f"  NPS : {s['slices'].stop - s['slices'].start} slices, "
            f"{len(s['roi_bounds'])} ROIs of {s['roi_bounds'][0][1] - s['roi_bounds'][0][0]} px")
        res['nps'] = nps(image, s['roi_bounds'], frame.dx, slices=s['slices'], plot_dir=plot_dir)
        res['nps'].meta['z_slices'] = [s['slices'].start, s['slices'].stop]

    if 'mtf' in res and 'nps' in res:
        res['neq'] = neq(res['mtf'], res['nps'])

    if 'ttf' in resolved:
        s = resolved['ttf']
        say(f"  TTF : {s['slices'].stop - s['slices'].start} slices, "
            f"{len(s['centre_pixels'])} inserts, radius {s['radius']} px")
        res['ttf'] = ttf(image, s['centre_pixels'], s['radius'], frame.dx,
                         materials=s['materials'], slices=s['slices'], plot_dir=plot_dir)
        res['ttf'].meta['z_slices'] = [s['slices'].start, s['slices'].stop]
        n_bad = len(s['materials']) - res['ttf'].summary['ttf50_n_used']
        if n_bad:
            out['skipped'].append(
                ('ttf', f"{n_bad} of {len(s['materials'])} inserts are below CNR "
                        f"{res['ttf'].summary['cnr_floor']:g}; their TTF would be a "
                        f"noise spectrum, so they are reported as unmeasured"))

    if 'mtf' in res and 'nps' in res:
        cfg = rois.d_prime or {}
        res['d_prime'] = detectability(
            res['mtf'], res['nps'], disc_diameters_mm=cfg.get('disc_diameters_mm'),
            contrast_hu=cfg.get('contrast_hu'), rose_threshold=cfg.get('rose_threshold'),
            plot_dir=plot_dir)
        if not np.isfinite(res['d_prime'].summary['detectable_size_mm']):
            cd = res['d_prime'].extra['curve_diameters_mm']
            out['skipped'].append(
                ('d_prime', f"d' never crosses {res['d_prime'].summary['rose_threshold']:g} "
                            f"on {cd[0]:.3g}-{cd[-1]:.3g} mm discs"))

    for kind, r in res.items():
        for k, v in r.summary.items():
            out['scalars'][k if kind != 'nps' or k != 'noise_std' else 'noise_std_hu'] = v
        if out_dir is not None:
            r.save(Path(out_dir) / kind)
    if out_dir is not None:
        save_metrics(out, Path(out_dir) / 'metrics.json')
    return out


def save_metrics(metrics: dict, path) -> Path:
    """Write a ``run_metrics`` result (curves included) as one JSON file."""
    import json
    from .results import _jsonable
    doc = {k: _jsonable(v) for k, v in metrics.items() if k != 'results'}
    doc['results'] = {kind: r.as_dict() for kind, r in metrics['results'].items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_metrics(path) -> dict:
    """Read back what :func:`save_metrics` wrote."""
    import json
    doc = json.loads(Path(path).read_text())
    doc['results'] = {k: MetricResult.from_dict(v) for k, v in doc.get('results', {}).items()}
    doc['skipped'] = [tuple(s) for s in doc.get('skipped', [])]
    return doc


def summary_table(metrics: dict) -> str:
    """The headline numbers of a ``run_metrics`` result as text."""
    sc = metrics['scalars']
    rows = [
        ('MTF50', sc.get('mtf50'), 'lp/mm'),
        ('MTF10', sc.get('mtf10'), 'lp/mm'),
        ('noise sigma', sc.get('noise_std_hu'), 'HU'),
        ('NPS peak frequency', sc.get('nps_f_peak'), 'lp/mm'),
        ('NPS mean frequency', sc.get('nps_f_mean'), 'lp/mm'),
        ('NEQ integral', sc.get('neq_total'), ''),
        ('TTF50 (mean of usable inserts)', sc.get('ttf50_mean'), 'lp/mm'),
        ('detectable disc at Rose d\'', sc.get('detectable_size_mm'), 'mm'),
    ]
    lines = []
    for name, v, unit in rows:
        if v is None:
            continue
        val = f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else ("n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        lines.append(f"  {name:<32} {val:>10} {unit}")
    for what, why in metrics.get('skipped', []):
        lines.append(f"  [{what}] {why}")
    return "\n".join(lines)
