"""``ct-demo``: the whole pipeline on a synthetic phantom with a known answer.

Builds a small phantom volume (a water cylinder holding a slanted dark slab,
a section of circular inserts and a featureless section), blurs it with a
Gaussian of known width, adds white noise, and then runs exactly what a
user would run on real data:

    ct-find-rois phantom.vff --seed seed.json --out rois.json
    ct-report    phantom.vff --rois rois.json --out results/

The printed table puts the measured MTF50 and noise next to the values the
phantom was built with, so the demo doubles as a self-check.

    ct-demo --out demo/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .helpers.vff_io import write_vff
from .roi_fit import fit_rois, verification_figure
from .report import run_report
from .rois import Frame

#: What the phantom is built with.
DEFAULTS = dict(pixel_size=0.25, blur_fwhm_mm=0.6, noise_hu=40.0, body_radius_mm=28.5,
                edge_angle_deg=4.0)


def synthetic_phantom(*, pixel_size=0.25, blur_fwhm_mm=0.6, noise_hu=40.0,
                      body_radius_mm=28.5, edge_angle_deg=4.0, seed=0):
    """A ``(z, y, x)`` HU volume, its :class:`Frame`, the seed dict and the truth."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    fov = 2 * body_radius_mm + 6.0
    n = int(round(fov / pixel_size))
    z_lo, z_hi = -27.0, 18.0
    nz = int(round((z_hi - z_lo) / pixel_size))
    frame = Frame({'vol_shape': (n, n, nz), 'vol_origin': (0.0, 0.0, 0.5 * (z_lo + z_hi)),
                   'dx': pixel_size, 'dz': pixel_size})
    x = frame.x_mm(np.arange(n))
    y = frame.y_mm(np.arange(n))
    z = frame.z_mm(np.arange(nz))
    X, Y = np.meshgrid(x, y, indexing='ij')                  # (x, y)
    body = np.hypot(X, Y) < body_radius_mm
    vol = np.full((n, n, nz), -1000.0, np.float32)           # (x, y, z)
    vol[body] = 0.0                                          # water

    # MTF section: a dark slab (air) whose x-edges lean by edge_angle
    t = np.tan(np.radians(edge_angle_deg))
    slab = (np.abs(Y) < 17.0) & (X > 2.0 + t * Y) & (X < 20.0 + t * Y)
    zs = (z > 2.0) & (z < 16.0)
    vol[:, :, zs] = np.where(slab[..., None], -900.0, vol[:, :, zs])

    # TTF section: inserts of several contrasts on a ring
    contrasts = [500.0, 300.0, -200.0, 150.0]
    centres = [(-16.0, 6.0), (13.0, -9.0), (-10.0, -10.0), (6.0, 15.0)]
    zt = (z > -25.5) & (z < -19.5)
    sec = vol[:, :, zt]
    for (cx, cy), c in zip(centres, contrasts):
        disc = np.hypot(X - cx, Y - cy) < 3.0
        sec[disc] = c
    vol[:, :, zt] = sec

    sigma_px = blur_fwhm_mm / 2.3548 / pixel_size
    vol = gaussian_filter(vol, (sigma_px, sigma_px, sigma_px))
    vol += rng.normal(0.0, noise_hu, vol.shape).astype(np.float32)

    seed_dict = {
        "scan": "synthetic_demo",
        "body_radius_mm": body_radius_mm,
        "mtf": {"z_search_mm": [0.0, 17.0], "width_mm": 5.0, "centre_y_mm": 0.0,
                "z_margin_mm": 1.0},
        "nps": {"z_mm": [-14.0, -8.0], "ring_radius_mm": 14.0, "n_roi": 8, "size_mm": 6.0},
        "ttf": {"z_mm": [-24.5, -20.5], "crop_radius_mm": 7.0, "min_contrast_hu": 100.0,
                "seed_centres_mm": [[cx + 0.4, cy - 0.3] for cx, cy in centres]},
        "d_prime": {"disc_diameters_mm": [0.15, 0.5, 1.0, 3.0], "contrast_hu": 300.0},
    }
    # The phantom's true MTF is the Gaussian blur times the voxel's box
    # aperture (a sinc), so the expected MTF50 is read off their product.
    from .results import crossing
    f = np.linspace(0.0, 1.0 / pixel_size, 8000)
    sig = blur_fwhm_mm / 2.3548
    true_mtf = np.exp(-2 * np.pi ** 2 * sig ** 2 * f ** 2) * np.abs(np.sinc(f * pixel_size))
    truth = {'mtf50_lp_per_mm': crossing(f, true_mtf, 0.5),
             'mtf50_gaussian_only': 0.4413 / blur_fwhm_mm,
             'noise_hu': noise_hu, 'edge_angle_deg': edge_angle_deg,
             'insert_contrasts_hu': contrasts}
    image = np.ascontiguousarray(vol.transpose(2, 1, 0))     # (z, y, x)
    return image, frame, seed_dict, truth


def run_demo(out_dir, *, quiet=False, **kw) -> dict:
    params = {**DEFAULTS, **kw}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = (lambda *a, **k: None) if quiet else print

    say(f"building a synthetic phantom @ {params['pixel_size']} mm, blur FWHM "
        f"{params['blur_fwhm_mm']} mm, noise {params['noise_hu']:.0f} HU ...")
    image, frame, seed, truth = synthetic_phantom(**params)
    vff = out / 'phantom.vff'
    write_vff(vff, {'bits': 16, 'elementsize': frame.dx}, image)
    (out / 'phantom.json').write_text(json.dumps({
        'voxel_size_mm': {'xy': frame.dx, 'z': frame.dz}, 'volume_shape': list(frame.shape),
        'vol_origin_mm': list(frame.origin), 'units': 'HU', 'truth': truth}, indent=2))
    (out / 'seed.json').write_text(json.dumps(seed, indent=2))
    say(f"  wrote {vff} {image.shape} and seed.json\n")

    say("ct-find-rois: measuring the regions on the phantom")
    rois, details = fit_rois(image, frame, seed, reference_name=str(vff), log=say)
    rois.save(out / 'rois.json')
    fig = verification_figure(details['reference'], rois, out / 'rois.png', details=details)
    say(f"  wrote {out / 'rois.json'} and {fig}\n")

    say("ct-report: running every metric")
    metrics = run_report(vff, out / 'rois.json', out / 'results', verbose=not quiet,
                         title='synthetic phantom')
    sc = metrics['scalars']
    say("\nself-check (measured vs built in):")
    say(f"  MTF50        {sc.get('mtf50', float('nan')):.3f} vs {truth['mtf50_lp_per_mm']:.3f} lp/mm")
    say(f"  noise sigma  {sc.get('noise_std_hu', float('nan')):.1f} vs {truth['noise_hu']:.1f} HU")
    say(f"  edge angle   {rois.mtf.get('edge_angle_deg', float('nan')):+.2f} vs "
        f"{truth['edge_angle_deg']:+.2f} deg")
    metrics['truth'] = truth
    return metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog='ct-demo', description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default='ct-demo', help='output directory (default: ./ct-demo)')
    ap.add_argument('--pixel-size', type=float, default=DEFAULTS['pixel_size'])
    ap.add_argument('--blur-fwhm', type=float, default=DEFAULTS['blur_fwhm_mm'],
                    help='Gaussian blur FWHM in mm')
    ap.add_argument('--noise', type=float, default=DEFAULTS['noise_hu'], help='noise sigma in HU')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_demo(args.out, quiet=args.quiet, pixel_size=args.pixel_size,
             blur_fwhm_mm=args.blur_fwhm, noise_hu=args.noise)


if __name__ == '__main__':
    main()
