"""Run every metric a JSON config asks for: ``ct-metrics``.

Each section of the config is optional. ``mtf`` and ``nps`` together add the
NEQ and the frequency-domain d'; ``ttf`` adds the TTF and, with ``nps`` and a
``d_prime`` section, the image-domain d'. See ``example_config.json``.
"""

import json
import os

import numpy as np

from . import d_prime_calculator
from .api import detectability, mtf as _mtf, neq as _neq, nps as _nps, ttf as _ttf
from .helpers.io_utils import ensure_output_dir, load_image_data, parse_slices


def _load(cfg):
    data = load_image_data(cfg['input'])
    if cfg.get('slices'):
        data = data[parse_slices(cfg['slices'])]
    return data


def run_all_metrics(config, *, verbose=True, plots=True):
    """Run the metrics the config defines; returns ``{kind: MetricResult}``.

    Every result is saved to ``output_dir`` as ``<kind>.npz`` + ``<kind>.json``.
    """
    say = print if verbose else (lambda *a, **k: None)
    out = ensure_output_dir(config.get('output_dir', './results'))
    plot_dir = out if plots else None
    px = float(config['pixel_size'])
    results = {}

    if 'mtf' in config:
        c = config['mtf']
        say("MTF")
        results['mtf'] = _mtf(_load(c), c['crop_indices'], px,
                              edge_angle=c.get('edge_angle', 5.0),
                              high_to_low=c.get('high_to_low', True), plot_dir=plot_dir)
        say("  " + results['mtf'].describe().replace("\n", "\n  "))

    nps_data = None
    if 'nps' in config:
        c = config['nps']
        say("NPS")
        nps_data = _load(c)
        results['nps'] = _nps(nps_data, np.array(c['roi_bounds']), px, plot_dir=plot_dir)
        say("  " + results['nps'].describe().replace("\n", "\n  "))

    if 'mtf' in results and 'nps' in results:
        say("NEQ")
        results['neq'] = _neq(results['mtf'], results['nps'])
        say("  " + results['neq'].describe().replace("\n", "\n  "))

    ttf_data = None
    if 'ttf' in config:
        c = config['ttf']
        say("TTF")
        ttf_data = _load(c)
        results['ttf'] = _ttf(ttf_data, c['centre_pixels'], c['radius'], px,
                              materials=c.get('materials'), plot_dir=plot_dir)
        say("  " + results['ttf'].describe().replace("\n", "\n  "))

    if 'mtf' in results and 'nps' in results:
        c = config.get('d_prime_npw', {})
        say("d' (NPW observer, from the MTF and NPS curves)")
        results['d_prime'] = detectability(
            results['mtf'], results['nps'], disc_diameters_mm=c.get('disc_diameters_mm'),
            contrast_hu=c.get('contrast_hu'), rose_threshold=c.get('rose_threshold'),
            plot_dir=plot_dir)
        say("  " + results['d_prime'].describe().replace("\n", "\n  "))

    if 'd_prime' in config and ttf_data is not None and nps_data is not None:
        c = config['d_prime']
        say("d' (image domain, from the TTF inserts)")
        size = c['task_object_size']
        task = d_prime_calculator.create_circular_task_function(
            c['task_contrast'], size, pixel_size=px, image_dimension=np.min(ttf_data.shape[1:]))
        d_prime_calculator.get_d_prime(
            ttf_data, config['ttf']['centre_pixels'], config['ttf']['radius'],
            config['ttf']['materials'], nps_data, np.array(config['nps']['roi_bounds']),
            task_function_data=task, task_function_material=c['task_material'],
            task_function_object_size=size, pixel_size=px, verbose=verbose,
            plot_results=plots, target_directory=out)

    for kind, r in results.items():
        r.save(os.path.join(out, kind))
    say(f"results saved to {out}")
    return results


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog='ct-metrics',
        description="Run every metric a JSON config asks for (any subset of mtf, nps, ttf, d_prime).")
    parser.add_argument('--config', required=True, help='JSON configuration (see example_config.json)')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--no_plot', action='store_true', help='Skip the diagnostic figures')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as f:
        config = json.load(f)
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    run_all_metrics(config, verbose=not args.quiet, plots=not args.no_plot)
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
