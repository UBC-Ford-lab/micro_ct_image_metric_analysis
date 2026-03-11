# Description: This script calculates all the metrics of an image data set: NPS, NEQ, MTF, TTF, and d'
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
from . import nps_calculator as NPS_calculator
from . import neq_calculator as NEQ_calculator
from . import mtf_calculator as MTF_calculator
from . import ttf_calculator as TTF_calculator
from . import d_prime_calculator


def run_all_metrics(config):
    """Run all metrics based on a configuration dictionary.

    :param config: Dictionary with keys for each metric's parameters.
                   See example_config.json for the expected format.
    """
    from .helper_scripts.io_utils import load_image_data, ensure_output_dir, parse_slices

    target_directory = config.get('output_dir', './results')
    ensure_output_dir(target_directory)
    pixel_size = config['pixel_size']

    # Calculate the NPS
    print("Calculating the NPS")
    nps_cfg = config['nps']
    image_data_NPS = load_image_data(nps_cfg['input'])
    if 'slices' in nps_cfg and nps_cfg['slices']:
        image_data_NPS = image_data_NPS[parse_slices(nps_cfg['slices'])]
    ROI_bounds_NPS = np.array(nps_cfg['roi_bounds'])

    _ = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=pixel_size,
                               target_directory=target_directory, plot_results=True)

    # Calculate the MTF
    print("Calculating the MTF")
    mtf_cfg = config['mtf']
    image_data_MTF = load_image_data(mtf_cfg['input'])
    if 'slices' in mtf_cfg and mtf_cfg['slices']:
        image_data_MTF = image_data_MTF[parse_slices(mtf_cfg['slices'])]
    crop_indices_MTF = mtf_cfg['crop_indices']

    _ = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=pixel_size,
                               target_directory=target_directory, plot_results=True,
                               edge_angle=mtf_cfg.get('edge_angle', 5.5),
                               high_to_low=mtf_cfg.get('high_to_low', True))

    # Calculate the NEQ
    print("Calculating the NEQ")
    _ = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=pixel_size,
                               target_directory=target_directory, plot_results=True)

    # Calculate the TTF
    print("Calculating the TTF")
    ttf_cfg = config['ttf']
    image_data_TTF = load_image_data(ttf_cfg['input'])
    if 'slices' in ttf_cfg and ttf_cfg['slices']:
        image_data_TTF = image_data_TTF[parse_slices(ttf_cfg['slices'])]
    centre_pixels_TTF = ttf_cfg['centre_pixels']
    radius_TTF = ttf_cfg['radius']
    materials_TTF = ttf_cfg['materials']

    _ = TTF_calculator.get_TTF(image_data_TTF, centre_pixels_TTF, radius_TTF, materials=materials_TTF,
                               find_absolute_TTF=True, pixel_size=pixel_size,
                               target_directory=target_directory, plot_results=True)

    # Calculate the detectability index d'
    print("Calculating the detectability index d'")
    dp_cfg = config['d_prime']
    task_function_object_size = dp_cfg['task_object_size']
    task_function_data = d_prime_calculator.create_circular_task_function(
        dp_cfg['task_contrast'], task_function_object_size,
        pixel_size=pixel_size,
        image_dimension=np.min(image_data_TTF.shape[1:]))

    _ = d_prime_calculator.get_d_prime(
        image_data_TTF, centre_pixels_TTF, radius_TTF, materials_TTF,
        image_data_NPS, ROI_bounds_NPS,
        task_function_data=task_function_data,
        task_function_material=dp_cfg['task_material'],
        task_function_object_size=task_function_object_size,
        pixel_size=pixel_size, verbose=True, plot_results=True,
        target_directory=target_directory)

    # Calculate the NPW detectability index (frequency-domain, from MTF/NPS)
    if 'd_prime_npw' in config:
        print("Calculating the NPW detectability index d' (frequency-domain)")
        npw_cfg = config['d_prime_npw']

        # Compute MTF and NPS if not already done
        mtf_freq, mtf_val = MTF_calculator.get_MTF(
            image_data_MTF, crop_indices_MTF, find_absolute_MTF=True,
            pixel_size=pixel_size, target_directory=None, plot_results=False,
            edge_angle=mtf_cfg.get('edge_angle', 5.5),
            high_to_low=mtf_cfg.get('high_to_low', True))
        nps_freq, nps_val = NPS_calculator.get_NPS(
            image_data_NPS, ROI_bounds_NPS, pixel_size=pixel_size,
            target_directory=None, plot_results=False)

        result = d_prime_calculator.get_d_prime_npw(
            mtf_freq, mtf_val, nps_freq, nps_val,
            disc_diameters_mm=npw_cfg.get('disc_diameters_mm'),
            contrast_hu=npw_cfg.get('contrast_hu'),
            plot_results=True, target_directory=target_directory)

        print(f"  NPW d' (ΔC={result['contrast_hu']:.0f} HU): "
              + ", ".join(f"{d:.2f}mm→{dp:.1f}"
                          for d, dp in zip(result['disc_diameters_mm'],
                                           result['d_prime'])))

    print("All metrics calculated successfully. Results saved to:", target_directory)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Calculate all CT image quality metrics (MTF, NPS, NEQ, TTF, d\') from a JSON config file.')
    parser.add_argument('--config', required=True, help='Path to JSON configuration file (see example_config.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    return parser.parse_args()


def main():
    import json
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.output_dir is not None:
        config['output_dir'] = args.output_dir

    run_all_metrics(config)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
