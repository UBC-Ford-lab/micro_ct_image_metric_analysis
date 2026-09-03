"""Find a phantom's metric regions automatically, in millimetres.

The calculators need an edge angle, insert centres and noise ROI positions,
and each of them is a measurement rather than something to type in from a
screenshot. This module measures them once on a reference volume:

* the MTF slab's z extent (largest dark block per slice), then the angle
  and position of its straightest usable edge from a robust iterated line
  fit. The angle is held fixed for every volume afterwards: refitting it per
  volume would let a noisier edge produce a different MTF for reasons that
  have nothing to do with resolution.
* each TTF insert's centre and edge radius from a circle fit to the radial
  gradient about a seed position, plus its fill and background HU. Inserts
  with too little contrast or whose analysis crop would run into the
  phantom shell are dropped WITH A REASON.
* the NPS ROIs, on a ring in a section verified to be featureless, checked
  for clearance against everything the fitter can see there.

Driven by a small SEED file of approximate positions (see
``seed_template``). Output is a :class:`PhantomROIs` JSON plus a
verification figure. Look at the figure: a region that has drifted onto the
wrong feature still produces numbers.

    ct-find-rois VOLUME --seed seed.json --out rois.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .rois import Frame, PhantomROIs
from .volumes import load_volume


class ROIFitError(RuntimeError):
    """A region could not be measured on the reference volume."""


def seed_template(body_radius_mm=28.5) -> dict:
    """A seed dict with every key the fitter reads, to edit for a phantom."""
    return {
        "scan": "my_phantom",
        "notes": "Approximate positions in mm; every number is refined on the reference volume.",
        "body_radius_mm": body_radius_mm,
        "mtf": {"z_search_mm": [2.0, 16.0], "width_mm": 5.0, "centre_y_mm": 6.0,
                "z_margin_mm": 1.0, "min_angle_deg": 1.5, "max_angle_deg": 12.0},
        "nps": {"z_mm": [1.5, 5.5], "ring_radius_mm": 14.0, "n_roi": 8, "size_mm": 6.0},
        "ttf": {"z_mm": [-24.5, -20.5], "crop_radius_mm": 7.0, "min_contrast_hu": 100.0,
                "seed_centres_mm": [[-20.0, 6.0], [17.3, -9.8], [-13.7, -9.5], [1.7, -16.1]]},
        "d_prime": {"disc_diameters_mm": [0.15, 0.5, 1.0, 3.0], "contrast_hu": 300.0},
    }


# --------------------------------------------------------------------------
# Helpers on a loaded reference
# --------------------------------------------------------------------------

class Reference:
    """A reference volume with mm axes. Works internally in ``(x, y, z)``."""

    def __init__(self, image_zyx, frame: Frame, body_radius_mm: float):
        self.frame = frame
        self.vol = np.asarray(frame.check_image(image_zyx), dtype=np.float32).transpose(2, 1, 0)
        self.nx, self.ny, self.nz = frame.shape
        self.dx, self.dz = frame.dx, frame.dz
        self.x = np.asarray(frame.x_mm(np.arange(self.nx)))
        self.y = np.asarray(frame.y_mm(np.arange(self.ny)))
        self.z = np.asarray(frame.z_mm(np.arange(self.nz)))
        self.R = np.hypot(self.x[:, None], self.y[None, :])
        self.body = self.R < float(body_radius_mm)
        if not self.body.any():
            raise ROIFitError(f"body_radius_mm {body_radius_mm} contains no voxel of this volume")

    def slab(self, z_lo, z_hi) -> np.ndarray:
        """The mean image over a mm interval, ``(x, y)``."""
        s = self.frame.z_slice((z_lo, z_hi))
        return self.vol[:, :, s].mean(axis=2)

    def at(self, image, x_mm, y_mm):
        """Bilinear sample of an ``(x, y)`` image at mm coordinates."""
        from scipy.ndimage import map_coordinates
        ix = (np.asarray(x_mm) - self.x[0]) / self.dx
        iy = (np.asarray(y_mm) - self.y[0]) / self.dx
        return map_coordinates(image, [ix.ravel(), iy.ravel()], order=1,
                               mode='nearest').reshape(np.shape(x_mm))


# --------------------------------------------------------------------------
# MTF: find the slab, then fit its straightest usable edge
# --------------------------------------------------------------------------

def find_slab(ref: Reference, z_search, *, contrast_hu=300.0, min_area_mm2=300.0, log=None):
    """The z interval of the largest low-HU block inside the search window."""
    from scipy import ndimage as ndi

    lo, hi = (float(v) for v in z_search)
    s = ref.frame.z_slice((lo, hi))
    bg = float(np.median(ref.vol[:, :, s][ref.body]))
    level = bg - float(contrast_hu)
    areas = np.zeros(ref.nz)
    for k in range(s.start, s.stop):
        m = (ndi.uniform_filter(ref.vol[:, :, k], 5) < level) & ref.body
        lab, n = ndi.label(m)
        if n:
            areas[k] = float(ndi.sum(m, lab, range(1, n + 1)).max()) * ref.dx * ref.dx
    hit = np.where(areas > float(min_area_mm2))[0]
    if hit.size == 0:
        raise ROIFitError(
            f"no dark slab found between z = {lo} and {hi} mm "
            f"(largest dark component peaked at {areas.max():.0f} mm^2)")
    runs = np.split(hit, np.where(np.diff(hit) != 1)[0] + 1)
    run = max(runs, key=len)
    if log:
        log(f"  slab: z {ref.z[run.min()]:+.2f} .. {ref.z[run.max()]:+.2f} mm "
            f"({run.size} slices), peak area {areas[run].max():.0f} mm^2")
    return float(ref.z[run.min()]), float(ref.z[run.max()])


def fit_edges(ref: Reference, z_lo, z_hi, *, contrast_hu=300.0, log=None):
    """Both x-constant edges of the slab: angle, straightness, position.

    Iterated: a first pass off the mask boundary, then refinements with a
    shrinking search window, because an edge slanted by 4 degrees over 34 mm
    moves 2.4 mm, wider than any sensible fixed window.
    """
    from scipy import ndimage as ndi

    img = ref.slab(z_lo, z_hi)
    bg = float(np.median(img[ref.body]))
    sm = ndi.uniform_filter(img, 3)
    mask = (sm < bg - float(contrast_hu)) & ref.body
    lab, n = ndi.label(mask)
    if n == 0:
        raise ROIFitError("no dark region in the MTF slab")
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    sel = ndi.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
    dark = float(np.median(img[sel]))
    mid = 0.5 * (dark + bg)
    i0, i1 = np.where(sel.any(axis=1))[0][[0, -1]]
    j0, j1 = np.where(sel.any(axis=0))[0][[0, -1]]
    if log:
        log(f"  slab: {dark:.0f} HU against {bg:.0f} HU (contrast {bg - dark:.0f} HU), "
            f"x {ref.x[i0]:+.2f}..{ref.x[i1]:+.2f}, y {ref.y[j0]:+.2f}..{ref.y[j1]:+.2f} mm")

    min_rows = max(20, int(0.4 * (j1 - j0)))
    out = {}
    for which in ('min', 'max'):
        js = np.arange(j0 + 8, j1 - 8)
        if js.size < min_rows:
            continue
        seed = np.array([
            (np.where(sel[:, j])[0].min() if which == 'min' else np.where(sel[:, j])[0].max())
            if sel[:, j].any() else np.nan for j in js], dtype=float)
        g = np.isfinite(seed)
        if g.sum() < min_rows:
            continue
        c = np.polyfit(js[g], seed[g], 1)
        pos = None
        for half in (25, 12, 8):
            pos = np.full(js.size, np.nan)
            for t, j in enumerate(js):
                base = np.polyval(c, j)
                a, b = int(round(base - half)), int(round(base + half))
                if a < 0 or b >= ref.nx:
                    continue
                seg = img[a:b, j]
                cross = np.where(np.diff(np.sign(seg - mid)) != 0)[0]
                if cross.size != 1:
                    continue
                k = cross[0]
                pos[t] = a + k + (mid - seg[k]) / (seg[k + 1] - seg[k])
            g = np.isfinite(pos)
            if g.sum() < min_rows:
                break
            c = np.polyfit(js[g], pos[g], 1)
            res = pos[g] - np.polyval(c, js[g])
            keep = np.abs(res) < 4 * max(res.std(), 1e-6)
            c = np.polyfit(js[g][keep], pos[g][keep], 1)
        g = np.isfinite(pos)
        if g.sum() < min_rows:
            continue
        res = pos[g] - np.polyval(c, js[g])
        out[which] = {
            'angle_deg': float(np.degrees(np.arctan(c[0]))),
            'rms_px': float(res.std()),
            'n_rows': int(g.sum()),
            'y_span_mm': float((js[g].max() - js[g].min()) * ref.dx),
            'slope': float(c[0]),
            'intercept_px': float(c[1]),
            'y_range_mm': (float(ref.y[js[g].min()]), float(ref.y[js[g].max()])),
        }
        if log:
            e = out[which]
            log(f"    edge x-{which}: {e['angle_deg']:+.3f} deg, rms {e['rms_px']:.2f} px, "
                f"{e['n_rows']} rows, span {e['y_span_mm']:.1f} mm")
    if not out:
        raise ROIFitError("neither slab edge could be traced")
    return out, dark, bg


def choose_edge(edges: dict, *, min_angle=1.5, max_angle=12.0):
    """The usable edge: slanted enough to supersample, straight, and long.

    Below about 1.5 degrees the edge does not cross a whole pixel over the
    ROI's height and the MTF is an artefact of how the rows happen to land.
    """
    usable = {k: e for k, e in edges.items() if min_angle <= abs(e['angle_deg']) <= max_angle}
    if not usable:
        angles = {k: round(e['angle_deg'], 2) for k, e in edges.items()}
        raise ROIFitError(
            f"no slab edge is slanted enough for the slanted-edge method "
            f"(need {min_angle}-{max_angle} deg, measured {angles})")
    return max(usable.items(), key=lambda kv: kv[1]['y_span_mm'] / max(kv[1]['rms_px'], 0.2))


# --------------------------------------------------------------------------
# TTF: refine each seeded insert
# --------------------------------------------------------------------------

def refine_insert(ref: Reference, image, x0, y0, *, search_mm=2.5, r_lo=None, r_hi=8.0,
                  n_az=72, rounds=3):
    """Centre, edge radius, fill and background HU of one insert.

    The centre comes from a least-squares circle fit to the edge points
    found along ``n_az`` rays, iterated. Rays whose edge is far from the
    consensus (a neighbour, the shell) are rejected before the fit.
    """
    r_lo = max(3 * ref.dx, 0.15 * r_hi) if r_lo is None else float(r_lo)
    th = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    cos, sin = np.cos(th), np.sin(th)
    rs = np.arange(0.2, r_hi + 0.6, min(0.05, ref.dx / 2))
    cx, cy = float(x0), float(y0)

    for _ in range(rounds):
        xs = cx + rs[:, None] * cos[None, :]
        ys = cy + rs[:, None] * sin[None, :]
        rays = ref.at(image, xs, ys)
        grad = np.abs(np.gradient(rays, rs[1] - rs[0], axis=0))
        band = (rs > r_lo) & (rs < r_hi)
        hit = rs[band][np.argmax(grad[band], axis=0)]
        med = float(np.median(hit))
        good = np.abs(hit - med) < max(0.6, 0.25 * med)
        if good.sum() < n_az // 3:
            break
        px = cx + hit[good] * cos[good]
        py = cy + hit[good] * sin[good]
        A = np.column_stack([2 * px, 2 * py, np.ones(px.size)])
        sol, *_ = np.linalg.lstsq(A, px ** 2 + py ** 2, rcond=None)
        nx_, ny_ = float(sol[0]), float(sol[1])
        if np.hypot(nx_ - x0, ny_ - y0) > search_mm:
            break
        cx, cy = nx_, ny_

    xs = cx + rs[:, None] * cos[None, :]
    ys = cy + rs[:, None] * sin[None, :]
    p = np.median(ref.at(image, xs, ys), axis=1)
    g = np.abs(np.gradient(p, rs[1] - rs[0]))
    band = (rs > r_lo) & (rs < r_hi)
    r_edge = float(rs[band][int(np.argmax(g[band]))])
    inner = rs < 0.5 * r_edge
    outer = (rs > 1.25 * r_edge) & (rs < 1.6 * r_edge)
    fill = float(np.median(p[inner])) if inner.any() else float(p[0])
    bg = float(np.median(p[outer])) if outer.any() else float(np.median(p[-5:]))
    return {'centre_mm': [float(cx), float(cy)], 'r_edge_mm': r_edge,
            'fill_hu': fill, 'background_hu': bg, 'contrast_hu': fill - bg}


# --------------------------------------------------------------------------
# NPS: place the ROIs and verify the section really is featureless
# --------------------------------------------------------------------------

def find_features(ref: Reference, img, *, min_area_mm2=1.0, k_sigma=4.0):
    """Blobs in a section that an NPS ROI must not contain: (x, y, radius) mm."""
    from scipy import ndimage as ndi
    bg = float(np.median(img[ref.body]))
    noise = float(np.std(img[ref.body]))
    feat = (np.abs(ndi.uniform_filter(img, 9) - bg) > k_sigma * noise) & ref.body
    lab, n = ndi.label(feat)
    avoid = []
    for i in range(1, n + 1):
        m = lab == i
        area = m.sum() * ref.dx * ref.dx
        if area < min_area_mm2:
            continue
        ci, cj = ndi.center_of_mass(m)
        avoid.append((float(ref.x[int(ci)]), float(ref.y[int(cj)]), float(np.sqrt(area / np.pi))))
    return avoid


def place_nps(ref: Reference, spec, *, avoid, log=None):
    """ROI centres on a ring, checked for clearance and for flatness."""
    z_mm = [float(v) for v in spec['z_mm']]
    side = float(spec['size_mm'])
    ring = float(spec['ring_radius_mm'])
    n = int(spec['n_roi'])
    img = ref.slab(*z_mm)
    half_diag = side * np.sqrt(2) / 2.0

    centres, dropped = [], []
    for t in range(n):
        a = 2 * np.pi * t / n
        cx, cy = ring * np.cos(a), ring * np.sin(a)
        clash = [f"({fx:+.1f},{fy:+.1f})" for fx, fy, fr in avoid
                 if np.hypot(cx - fx, cy - fy) < half_diag + fr + 0.5]
        if clash:
            dropped.append((cx, cy, f"overlaps {', '.join(clash)}"))
            continue
        i0 = int(round(ref.frame.ix(cx)))
        j0 = int(round(ref.frame.iy(cy)))
        h = max(2, int(round(side / ref.dx / 2)))
        patch = img[max(0, i0 - h):i0 + h, max(0, j0 - h):j0 + h]
        if patch.size == 0:
            dropped.append((cx, cy, "outside the volume"))
            continue
        centres.append({'centre_mm': [float(cx), float(cy)],
                        'mean_hu': float(patch.mean()), 'std_hu': float(patch.std()),
                        'p2p_hu': float(np.percentile(patch, 99) - np.percentile(patch, 1))})
    if not centres:
        raise ROIFitError("every NPS ROI position was rejected")
    stds = np.array([c['std_hu'] for c in centres])
    med = float(np.median(stds))
    keep, rejected = [], []
    for c in centres:
        (rejected if c['std_hu'] > 2.0 * med else keep).append(c)
    if log:
        log(f"  NPS: {len(keep)} ROIs of {side:.1f} mm on a {ring:.1f} mm ring, "
            f"z {z_mm[0]:+.1f}..{z_mm[1]:+.1f} mm; median ROI std {med:.1f} HU")
        for cx, cy, why in dropped:
            log(f"    dropped ({cx:+.1f},{cy:+.1f}): {why}")
        for c in rejected:
            log(f"    dropped ({c['centre_mm'][0]:+.1f},{c['centre_mm'][1]:+.1f}): "
                f"std {c['std_hu']:.1f} HU is more than twice the median, there is structure in it")
    return keep, med


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------

def fit_rois(image_zyx, frame: Frame, seed: dict, *, reference_name="", log=None) -> tuple:
    """Measure every region the seed asks for. Returns ``(rois, details)``.

    ``details`` holds the per-insert fits and the chosen edge, for the
    verification figure. Sections missing from the seed are skipped.
    """
    body_r = float(seed.get('body_radius_mm', 28.5))
    ref = Reference(image_zyx, frame, body_r)
    rois = PhantomROIs(
        scan=seed.get('scan', ''), reference_volume=str(reference_name),
        created=datetime.date.today().isoformat(),
        notes=("Regions in isocentre-centred mm, MEASURED on the reference volume "
               "named above by ct-find-rois. Refit rather than hand-edit, so the "
               "verification figure stays in step with the numbers."),
        d_prime=dict(seed.get('d_prime', {})))
    details = {'ttf_fits': [], 'edge': None}

    if 'mtf' in seed:
        spec = seed['mtf']
        if log:
            log("MTF")
        contrast = float(spec.get('slab_contrast_hu', 300.0))
        z_lo, z_hi = find_slab(ref, spec['z_search_mm'], contrast_hu=contrast, log=log)
        margin = float(spec.get('z_margin_mm', 1.0))
        z_use = (z_lo + margin, z_hi - margin) if z_hi - z_lo > 2 * margin else (z_lo, z_hi)
        edges, dark, bright = fit_edges(ref, *z_use, contrast_hu=contrast, log=log)
        which, edge = choose_edge(edges, min_angle=float(spec.get('min_angle_deg', 1.5)),
                                  max_angle=float(spec.get('max_angle_deg', 12.0)))
        width = float(spec['width_mm'])
        cy = float(spec.get('centre_y_mm', 0.5 * sum(edge['y_range_mm'])))
        x_at_y0 = float(ref.x[0] + np.polyval([edge['slope'], edge['intercept_px']],
                                              (0.0 - ref.y[0]) / ref.dx) * ref.dx)
        cx = x_at_y0 + np.tan(np.radians(edge['angle_deg'])) * cy
        rois.mtf = {
            'z_mm': [round(z_use[0], 3), round(z_use[1], 3)],
            'centre_mm': [round(float(cx), 3), round(cy, 3)],
            'size_mm': [round(width, 3), round(2.0 * width, 3)],
            'edge_angle_deg': round(edge['angle_deg'], 4),
            'edge_x_at_y0_mm': round(x_at_y0, 4),
            'edge_rms_px': round(edge['rms_px'], 3),
            'edge_y_span_mm': round(edge['y_span_mm'], 2),
            'edge_side': f"x-{which}",
            # Scanning along +x we cross from dark to bright on the x-max
            # edge and the other way on x-min; getting it backwards inverts
            # the ERF.
            'high_to_low': bool(which == 'min'),
            'slab_hu': round(dark, 1),
            'background_hu': round(bright, 1),
        }
        details['edge'] = edge
        if log:
            log(f"  chosen: x-{which} at {edge['angle_deg']:+.3f} deg, crop "
                f"{width:.1f} x {2*width:.1f} mm centred at ({cx:+.2f}, {cy:+.2f}) mm")

    if 'ttf' in seed:
        spec = seed['ttf']
        if log:
            log("TTF")
        img = ref.slab(*spec['z_mm'])
        crop_r = float(spec['crop_radius_mm'])
        floor = float(spec.get('min_contrast_hu', 100.0))
        fits = []
        for x0, y0 in spec['seed_centres_mm']:
            f = refine_insert(ref, img, float(x0), float(y0), r_hi=crop_r + 1.0)
            cx, cy = f['centre_mm']
            r_pol = float(np.hypot(cx, cy))
            reasons = []
            if abs(f['contrast_hu']) < floor:
                reasons.append(f"contrast {f['contrast_hu']:+.0f} HU below the {floor:.0f} HU floor")
            if r_pol + crop_r > body_r:
                reasons.append(f"crop reaches r = {r_pol + crop_r:.1f} mm, past the "
                               f"{body_r:.1f} mm body edge")
            if f['r_edge_mm'] >= crop_r:
                reasons.append(f"edge at {f['r_edge_mm']:.1f} mm is outside the "
                               f"{crop_r:.1f} mm analysis radius")
            f['used'] = not reasons
            f['rejected_because'] = reasons
            fits.append(f)
            if log:
                log(f"  {'use ' if f['used'] else 'DROP'} ({cx:+7.2f},{cy:+7.2f})  "
                    f"r_edge {f['r_edge_mm']:.2f} mm  contrast {f['contrast_hu']:+7.0f} HU"
                    + ("" if f['used'] else "   [" + "; ".join(reasons) + "]"))
        used = [f for f in fits if f['used']]
        details['ttf_fits'] = fits
        if not used:
            raise ROIFitError("no TTF insert survived; nothing to measure")
        used = [used[i] for i in np.argsort([-abs(f['contrast_hu']) for f in used])]
        names = spec.get('materials')
        rois.ttf = {
            'z_mm': [float(v) for v in spec['z_mm']],
            'centres_mm': [[round(v, 3) for v in f['centre_mm']] for f in used],
            'radius_mm': crop_r,
            # Named by MEASURED contrast unless the seed names the materials.
            'materials': ([str(names[fits.index(f)]) for f in used] if names
                          else [f"{f['contrast_hu']:+.0f}HU" for f in used]),
            'measured': [{'contrast_hu': round(f['contrast_hu'], 1),
                          'fill_hu': round(f['fill_hu'], 1),
                          'background_hu': round(f['background_hu'], 1),
                          'r_edge_mm': round(f['r_edge_mm'], 3)} for f in used],
        }

    if 'nps' in seed:
        spec = seed['nps']
        if log:
            log("NPS")
        # Only what is visible in the NPS section itself is an obstacle.
        avoid = find_features(ref, ref.slab(*spec['z_mm']))
        if log:
            log(f"  {len(avoid)} features to clear in this section")
        keep, med = place_nps(ref, spec, avoid=avoid, log=log)
        rois.nps = {
            'z_mm': [float(v) for v in spec['z_mm']],
            'roi_centres_mm': [[round(v, 3) for v in c['centre_mm']] for c in keep],
            'size_mm': float(spec['size_mm']),
            'measured': {'median_roi_std_hu': round(med, 2),
                         'roi_mean_hu': [round(c['mean_hu'], 1) for c in keep]},
        }

    details['reference'] = ref
    return rois, details


# --------------------------------------------------------------------------
# Verification figure
# --------------------------------------------------------------------------

def verification_figure(ref: Reference, rois: PhantomROIs, path, *, details):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [k for k in ('ttf', 'nps', 'mtf') if getattr(rois, k)]
    fig, axes = plt.subplots(1, max(1, len(panels)), figsize=(8 * max(1, len(panels)), 8.6),
                             squeeze=False)
    axes = axes[0]
    ext = [ref.x[0], ref.x[-1], ref.y[0], ref.y[-1]]
    lim = float(np.max(np.abs(ref.x))) * 0.95

    def show(ax, z_mm, title, vmin=None, vmax=None):
        img = ref.slab(*z_mm)
        if vmin is None:
            vmin, vmax = np.percentile(img[ref.body], [1, 99.5])
        ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=ext)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')

    for ax, what in zip(axes, panels):
        if what == 'ttf':
            show(ax, rois.ttf['z_mm'],
                 f"TTF inserts   z {rois.ttf['z_mm'][0]:+.1f}..{rois.ttf['z_mm'][1]:+.1f} mm")
            rad = rois.ttf['radius_mm']
            for f in details.get('ttf_fits', []):
                cx, cy = f['centre_mm']
                col = 'lime' if f.get('used') else 'red'
                ax.add_patch(plt.Circle((cx, cy), f['r_edge_mm'], fill=False, color=col, lw=1.6))
                ax.add_patch(plt.Circle((cx, cy), rad, fill=False, color=col, lw=0.8, ls='--'))
                ax.text(cx, cy, f"{f['contrast_hu']:+.0f}", color=col, fontsize=8,
                        ha='center', va='center')
        elif what == 'nps':
            show(ax, rois.nps['z_mm'],
                 f"NPS ROIs   z {rois.nps['z_mm'][0]:+.1f}..{rois.nps['z_mm'][1]:+.1f} mm")
            s = rois.nps['size_mm']
            for cx, cy in rois.nps['roi_centres_mm']:
                ax.add_patch(plt.Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False,
                                           color='lime', lw=1.4))
        else:
            show(ax, rois.mtf['z_mm'],
                 f"MTF edge   z {rois.mtf['z_mm'][0]:+.1f}..{rois.mtf['z_mm'][1]:+.1f} mm")
            cx, cy = rois.mtf['centre_mm']
            w, h = rois.mtf['size_mm']
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                                       color='lime', lw=1.8))
            edge = details.get('edge')
            if edge:
                ys = np.array(edge['y_range_mm'])
                ax.plot(rois.mtf['edge_x_at_y0_mm']
                        + np.tan(np.radians(rois.mtf['edge_angle_deg'])) * ys, ys,
                        color='orange', lw=1.0, ls='--')
            ax.text(cx, cy - h / 2 - 2.5, f"{rois.mtf['edge_angle_deg']:+.2f} deg",
                    color='lime', ha='center', fontsize=10)

    fig.suptitle(f"{rois.scan} metric regions, fitted on {Path(rois.reference_volume).name}",
                 fontsize=13)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return Path(path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog='ct-find-rois',
        description="Measure a phantom's metric regions (edge angle, insert centres, "
                    "noise ROIs) on a reference volume and write them in mm.")
    ap.add_argument('volume', nargs='?', help='reference volume (.npy, .npz or .vff)')
    ap.add_argument('--seed', help='seed JSON with approximate positions (see --template)')
    ap.add_argument('--out', help='ROI table to write (JSON)')
    ap.add_argument('--figure', default=None, help='verification figure (default: <out>.png)')
    ap.add_argument('--pixel-size', type=float, default=None, help='voxel size in mm')
    ap.add_argument('--slice-thickness', type=float, default=None)
    ap.add_argument('--flip', nargs='+', default=(), choices=('x', 'y', 'z'),
                    help='reverse these axes after loading')
    ap.add_argument('--template', metavar='PATH',
                    help='write a seed template there and exit')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.template:
        Path(args.template).write_text(json.dumps(seed_template(), indent=2))
        print(f"seed template written to {args.template}; edit the positions, then run "
              f"ct-find-rois VOLUME --seed {args.template} --out rois.json")
        return
    if not (args.volume and args.seed and args.out):
        raise SystemExit("ct-find-rois needs VOLUME, --seed and --out (or --template PATH)")
    log = (lambda *a, **k: None) if args.quiet else print
    seed = json.loads(Path(args.seed).read_text())
    image, frame, info = load_volume(args.volume, pixel_size=args.pixel_size,
                                     slice_thickness=args.slice_thickness, flip=args.flip)
    log(f"reference: {args.volume}  {info['shape_zyx']} @ {frame.dx:g} mm "
        f"(voxel size from {info['source']})")
    try:
        rois, details = fit_rois(image, frame, seed, reference_name=args.volume, log=log)
    except ROIFitError as exc:
        raise SystemExit(f"error: {exc}") from exc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rois.save(out)
    fig = verification_figure(details['reference'], rois, args.figure or out.with_suffix('.png'),
                              details=details)
    log(f"\nROI table written to {out}\nverification figure written to {fig}")
    log("LOOK AT THE FIGURE before using this table: a region that has drifted onto "
        "the wrong feature still produces numbers.")


if __name__ == '__main__':
    main()
