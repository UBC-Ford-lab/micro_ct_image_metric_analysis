# Description: This script calculates the Detectability index (d') of an image data set
# using the TTF, NPS, and W (task function)
# Written by Falk Wiegmann at the University of British Columbia in May 2024.
# Updated March 2026: added frequency-domain NPW observer (get_d_prime_npw) for
# pre-computed MTF/NPS curves following AAPM TG-233 (Samei et al., 2019).

import numpy as np
import os
import matplotlib.pyplot as plt
from photutils.profiles import RadialProfile
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import j1
import scipy.integrate
from . import ttf_calculator as TTF_calculator
from . import nps_calculator as NPS_calculator


# =============================================================================
# Frequency-domain NPW observer (AAPM TG-233)
# =============================================================================

# Default disc diameters for micro-CT task-based assessment (mm)
DEFAULT_DISC_DIAMETERS = [0.15, 0.5, 1.0, 3.0]

# Default contrast: soft-tissue lesion (100 HU) — clinically challenging
# scenario for micro-CT mouse imaging.  Other common choices:
#   ~300-700 HU  lung nodule vs parenchyma
#   ~1000 HU     air-water (phantom calibration)
DEFAULT_CONTRAST_HU = 100.0


def disc_task_function(f, radius_mm, contrast_hu):
    """Radial Fourier transform of a uniform disc (jinc function).

    For a disc of radius R and contrast ΔC, the 2D FT evaluated along
    the radial frequency axis is:

        W(f) = ΔC · π R² · 2 J₁(2πRf) / (2πRf)       f > 0
        W(0) = ΔC · π R²                               f = 0

    Args:
        f: 1D array of radial spatial frequencies (lp/mm or mm⁻¹).
        radius_mm: Disc radius in mm.
        contrast_hu: Contrast between disc and background in HU.

    Returns:
        W: 1D array, same shape as *f*, task function amplitude.
    """
    f = np.asarray(f, dtype=np.float64)
    W = np.empty_like(f)
    area = np.pi * radius_mm ** 2

    nonzero = f > 0
    arg = 2.0 * np.pi * radius_mm * f[nonzero]
    W[nonzero] = contrast_hu * area * 2.0 * j1(arg) / arg
    W[~nonzero] = contrast_hu * area
    return W


def get_d_prime_npw(mtf_freq, mtf, nps_freq, nps,
                    disc_diameters_mm=None, contrast_hu=None,
                    normalize_mtf=True, n_freq=500,
                    plot_results=True, target_directory=None):
    """Compute NPW-observer detectability index for multiple disc sizes.

    Uses pre-computed 1D MTF and NPS curves and the AAPM TG-233
    non-prewhitening (NPW) matched-filter formulation:

        d'² = [∫ |W(f)|² MTF²(f) f df]²
              / [∫ |W(f)|² MTF²(f) NPS(f) f df]

    where W(f) is the analytical Bessel-function task function for a
    uniform disc and the integrals run from 0 to the Nyquist frequency.

    Args:
        mtf_freq: 1D array of MTF frequencies (mm⁻¹).
        mtf: 1D array of MTF values (absolute or normalized).
        nps_freq: 1D array of NPS frequencies (mm⁻¹).
        nps: 1D array of NPS values (HU² mm²).
        disc_diameters_mm: List of disc diameters in mm.
            Defaults to DEFAULT_DISC_DIAMETERS [0.15, 0.5, 1.0, 3.0].
        contrast_hu: Contrast between disc and background in HU.
            Defaults to DEFAULT_CONTRAST_HU (100 HU, soft tissue).
        normalize_mtf: If True, normalize MTF to unity at f=0 before
            computing d'.  Recommended when absolute MTF scale varies
            across methods.
        n_freq: Number of frequency samples for integration grid.
        plot_results: Whether to save a diagnostic plot.
        target_directory: Output directory for the plot.
            If None, uses current working directory.

    Returns:
        dict with keys:
            'disc_diameters_mm': array of disc diameters
            'contrast_hu': contrast used
            'd_prime': array of d' values, one per disc diameter
            'freq': integration frequency grid
            'task_functions': dict mapping diameter → W(f) array
    """
    if disc_diameters_mm is None:
        disc_diameters_mm = list(DEFAULT_DISC_DIAMETERS)
    if contrast_hu is None:
        contrast_hu = DEFAULT_CONTRAST_HU
    if target_directory is None:
        target_directory = os.getcwd()

    disc_diameters_mm = np.asarray(disc_diameters_mm, dtype=np.float64)

    # Build a common frequency grid up to the shared Nyquist
    pos_mtf = mtf_freq >= 0
    f_max = min(mtf_freq[pos_mtf].max(), nps_freq.max())
    freq = np.linspace(0, f_max, n_freq)

    # Interpolate MTF and NPS onto the common grid
    mtf_interp = interp1d(mtf_freq[pos_mtf], mtf[pos_mtf],
                          bounds_error=False, fill_value=0.0)(freq)
    nps_interp = interp1d(nps_freq, nps,
                          bounds_error=False, fill_value='extrapolate')(freq)

    if normalize_mtf:
        mtf_max = mtf_interp.max()
        if mtf_max > 0:
            mtf_interp = mtf_interp / mtf_max

    # Protect against zero/negative NPS
    nps_interp = np.maximum(nps_interp, np.max(nps_interp) * 1e-12)

    # Compute d' for each disc diameter
    d_prime_values = np.empty(len(disc_diameters_mm))
    task_functions = {}

    for i, diam in enumerate(disc_diameters_mm):
        R = diam / 2.0
        W = disc_task_function(freq, R, contrast_hu)
        task_functions[diam] = W

        integrand_num = np.abs(W) ** 2 * mtf_interp ** 2 * freq
        integrand_den = np.abs(W) ** 2 * mtf_interp ** 2 * nps_interp * freq

        numerator = scipy.integrate.simpson(integrand_num, x=freq)
        denominator = scipy.integrate.simpson(integrand_den, x=freq)

        if denominator > 0 and numerator > 0:
            d_prime_values[i] = numerator / np.sqrt(denominator)
        else:
            d_prime_values[i] = 0.0

    if plot_results:
        _plot_d_prime_npw(freq, mtf_interp, nps_interp, task_functions,
                          disc_diameters_mm, d_prime_values, contrast_hu,
                          target_directory)

    return {
        'disc_diameters_mm': disc_diameters_mm,
        'contrast_hu': contrast_hu,
        'd_prime': d_prime_values,
        'freq': freq,
        'task_functions': task_functions,
    }


def _plot_d_prime_npw(freq, mtf, nps, task_functions,
                      diameters, d_prime, contrast_hu, target_directory):
    """Save diagnostic plot for NPW d' calculation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Task functions
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(diameters)))
    for diam, col in zip(diameters, colors):
        W = task_functions[diam]
        W_norm = W / W.max() if W.max() > 0 else W
        ax.plot(freq, W_norm, color=col, linewidth=1.5,
                label=f'{diam:.2f} mm')
    ax.set_xlabel('Spatial Frequency (mm$^{-1}$)')
    ax.set_ylabel('|W(f)| (normalized)')
    ax.set_title('(a) Disc task functions', fontweight='bold')
    ax.set_xlim([0, freq.max()])
    ax.legend(fontsize=8, title='Diameter')
    ax.grid(True, alpha=0.3)

    # (b) MTF and NPS
    ax = axes[1]
    ax.plot(freq, mtf, 'b-', linewidth=1.5, label='MTF (normalized)')
    ax2 = ax.twinx()
    ax2.semilogy(freq, nps, 'r--', linewidth=1.5, label='NPS', alpha=0.8)
    ax.set_xlabel('Spatial Frequency (mm$^{-1}$)')
    ax.set_ylabel('MTF', color='b')
    ax2.set_ylabel('NPS (HU$^2$ mm$^2$)', color='r')
    ax.set_title('(b) System MTF & NPS', fontweight='bold')
    ax.set_xlim([0, freq.max()])
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (c) d' bar chart
    ax = axes[2]
    x_pos = np.arange(len(diameters))
    bars = ax.bar(x_pos, d_prime, color=colors, edgecolor='black',
                  linewidth=0.5, width=0.6)
    ax.axhline(3.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(5.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(len(diameters) - 0.5, 3.2, "Rose (d'=3)", fontsize=7,
            color='gray', ha='right')
    ax.text(len(diameters) - 0.5, 5.2, "d'=5", fontsize=7,
            color='gray', ha='right')
    for bar, val in zip(bars, d_prime):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{d:.2f}' for d in diameters], fontsize=8)
    ax.set_xlabel('Disc diameter (mm)')
    ax.set_ylabel("Detectability index d'")
    ax.set_title(f"(c) NPW d' (ΔC = {contrast_hu:.0f} HU)",
                 fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(target_directory, 'Detectability_NPW_plot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  NPW d' plot saved to: {path}")

def get_d_prime(image_data_TTF, centre_pixels_TTF, radius_TTF, materials_TTF, image_data_NPS, ROI_bounds_NPS,
                task_function_data, task_function_material, task_function_object_size, pixel_size=0.05,
                verbose=True, plot_results=True, target_directory=os.getcwd()):
    """
    This function calculates the Detectability index (d') on image data. It uses the TTF and NPS functions from
    the TTF_calculator and NPS_calculator scripts respectively.
    :param image_data_TTF: The image data for the TTF calculation as a 3D numpy array (z, y, x)
    :param centre_pixels_TTF: The centre of the circular edge in pixel coordinates (y, x)
    :param radius_TTF: The radius of the circular edges in pixels
    :param materials_TTF: The materials of the circular edges (e.g. ['Air', 'Water', 'Bone', 'Iodine'])
    :param image_data_NPS: The image data for the NPS calculation as a 3D numpy array (z, y, x)
    :param ROI_bounds_NPS: Specify the region of interest to crop the images for the NPS calculation (y1, y2, x1, x2)
    :param task_function_data: The task function data as a 3D numpy array (z, y, x)
    :param task_function_material: The material of the task function (e.g. 'Iodine')
    :param task_function_object_size: The size of the task function object in mm
    :param pixel_size: The pixel size of the images in mm
    :param verbose: Boolean to print the results to the console
    :param plot_results: Boolean to plot the results or not
    :return: d_prime: The Detectability index (d')
    """

    # Calculate the TTF
    TTF_freq, TTF, CNR_array = TTF_calculator.get_TTF(image_data_TTF, centre_pixels_TTF, radius_TTF, materials=materials_TTF,
                                                      find_absolute_TTF=True, pixel_size=pixel_size,
                                                      target_directory=None, plot_results=False)

    # Calculate the NPS
    NPS_freq, NPS = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=pixel_size,
                                           target_directory=None, plot_results=False)

    try:
        CNR = CNR_array[materials_TTF.index(task_function_material)]
        TTF = TTF[materials_TTF.index(task_function_material)]
    except:
        raise ValueError('The task function material does not match the materials of the circular edges!')


    # Calculate the 2D FFT of the task function data to get the task function
    task_function = np.abs(np.fft.fftshift(np.fft.fft2(task_function_data)))

    # Calculate the radial profile of the 2d task function
    radial_profile = RadialProfile(task_function, (int(task_function.shape[1]/2), int(task_function.shape[0]/2)),
                                    np.arange(0, int(np.min([task_function.shape[1], task_function.shape[0]])/2)))

    task_function_freqs = radial_profile.radius*np.max(TTF_freq)/np.max(radial_profile.radius) # scale it to Nyquist frequency limit

    freqs = np.linspace(0, np.max(TTF_freq), 100) # freqs up to Nyquist frequency
    # Interpolate the TTF, NPS, and W to the desired frequency range
    TTF_spline = CubicSpline(TTF_freq, TTF)(freqs)
    NPS_spline = CubicSpline(NPS_freq, NPS)(freqs)
    W_spline = CubicSpline(task_function_freqs, radial_profile.profile)(freqs)

    # TTF_spline is unitless, TTF_freq is in mm^-1
    # NPS_spline is in HU^2 mm^2, NPS_freq is in mm^-1
    # W_spline is HU mm, task_function_freqs is in mm^-1

    # Calculate the Detectability index (d')
    d_prime = (scipy.integrate.simpson((np.abs(W_spline)**2 * TTF_spline**2), x=freqs)/np.sqrt(
                scipy.integrate.simpson((np.abs(W_spline)**2 * TTF_spline**2 * NPS_spline), x=freqs)))

    if verbose:
        print('The Detectability index, d\', for {}mm radius object of material ({}) with CNR={:.2f} is: {:.2f}'.format(
                task_function_object_size, task_function_material, CNR, d_prime))

    if plot_results:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        ax1 = axs[0]
        ax1.imshow(task_function_data, cmap='gray')
        ax1.set_xlabel('x (pixels)')
        ax1.set_ylabel('y (pixels)')
        ax1.set_title('Task Function Object')

        ax2 = axs[1]
        ax2.plot(freqs, TTF_spline, label='TTF')
        ax2.plot(freqs, NPS_spline, label='NPS (HU$^2$ mm$^2$)')
        ax2.plot(freqs, W_spline, label='Task function, W (HU mm)')
        ax2.set_xlabel('Spatial Frequency (mm$^{-1}$)')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim([0, np.max(freqs)])
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('TTF, NPS, and Task Function Plot')

        fig.suptitle('The Detectability index, d\', for {}mm radius object of material ({}) with CNR={:.2f} is: {:.2f}'.format(
                task_function_object_size, task_function_material, CNR, d_prime))

        plt.tight_layout()
        plt.savefig(target_directory+'/Detectability_plot.png', dpi=300)
        plt.close('all')

    return d_prime


def create_circular_task_function(contrast, radius, pixel_size=0.05, image_dimension=1000):
    """
    Create a circular task function for the detectability index calculation with a given contrast and radius.
    :param contrast: The contrast/signal of the task function circular point (in HU), e.g fat=-160
    :param radius: The radius of the circular task function in mm, e.g. 0.3 mm
    :param pixel_size: The pixel size of the task function in mm, default is 0.05 mm
    :param image_dimension: The dimension of the image in pixels, default is 1000
    This function creates a task function for the detectability index calculation
    :return: task_function_data: The task function data as a 2D numpy array (y, x)
    """

    # Create the task function data
    task_function_data = np.zeros((image_dimension, image_dimension))
    task_function_data[(np.sqrt((np.ogrid[:task_function_data.shape[0], :task_function_data.shape[1]][1] - int(task_function_data.shape[0] / 2))**2
                       + (np.ogrid[:task_function_data.shape[0], :task_function_data.shape[1]][0] - int(task_function_data.shape[1] / 2))**2) <= radius/pixel_size)] = contrast

    return task_function_data

def parse_args():
    import argparse
    from .helper_scripts.io_utils import parse_centre_pixels, parse_roi_bounds, parse_slices

    parser = argparse.ArgumentParser(
        description='Calculate the Detectability Index (d\') from CT image data.')
    sub = parser.add_subparsers(dest='mode', help='Calculation mode')

    # --- TTF-based mode (original) ---
    p_ttf = sub.add_parser('ttf', help='TTF-based d\' from circular phantom inserts')
    p_ttf.add_argument('--input_ttf', required=True, help='Path to TTF image file (.npy, .npz, or .vff)')
    p_ttf.add_argument('--input_nps', required=True, help='Path to NPS image file (.npy, .npz, or .vff)')
    p_ttf.add_argument('--centre_pixels', required=True, type=parse_centre_pixels,
                        help='Centre pixels as semicolon-separated y,x pairs (e.g. "686,398;418,132")')
    p_ttf.add_argument('--radius', required=True, type=int, help='Radius of circular edges in pixels')
    p_ttf.add_argument('--materials', required=True, type=str,
                        help='Comma-separated material names (e.g. "SB3,Teflon,Fat,Tissue")')
    p_ttf.add_argument('--roi_bounds', required=True, type=parse_roi_bounds,
                        help='NPS ROI bounds as semicolon-separated y1,y2,x1,x2 groups')
    p_ttf.add_argument('--task_material', required=True, type=str, help='Task function material (e.g. "Fat")')
    p_ttf.add_argument('--task_contrast', type=float, default=-160, help='Task function contrast in HU (default: -160)')
    p_ttf.add_argument('--task_object_size', type=float, default=0.2, help='Task function object radius in mm (default: 0.2)')
    p_ttf.add_argument('--slices_ttf', type=str, default=None, help='Slice selection for TTF data')
    p_ttf.add_argument('--slices_nps', type=str, default=None, help='Slice selection for NPS data')
    p_ttf.add_argument('--pixel_size', type=float, default=0.05, help='Pixel size in mm (default: 0.05)')
    p_ttf.add_argument('--output_dir', type=str, default='./results', help='Output directory (default: ./results)')
    p_ttf.add_argument('--no_plot', action='store_true', help='Disable plot generation')
    p_ttf.add_argument('--show', action='store_true', help='Display plots interactively')
    p_ttf.add_argument('--quiet', action='store_true', help='Suppress console output')

    # --- NPW mode (frequency-domain, from pre-computed MTF/NPS) ---
    p_npw = sub.add_parser('npw', help='NPW-observer d\' from pre-computed MTF/NPS curves (AAPM TG-233)')
    p_npw.add_argument('--mtf_npz', required=True,
                        help='Path to .npz with "mtf_freq" and "mtf" arrays')
    p_npw.add_argument('--nps_npz', required=True,
                        help='Path to .npz with "nps_freq" and "nps" arrays')
    p_npw.add_argument('--disc_diameters', type=str, default=None,
                        help='Comma-separated disc diameters in mm (default: 0.15,0.5,1.0,3.0)')
    p_npw.add_argument('--contrast', type=float, default=DEFAULT_CONTRAST_HU,
                        help=f'Contrast in HU (default: {DEFAULT_CONTRAST_HU})')
    p_npw.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    p_npw.add_argument('--no_plot', action='store_true', help='Disable plot generation')
    p_npw.add_argument('--show', action='store_true', help='Display plots interactively')

    # Default to TTF mode for backwards compatibility
    return parser.parse_args()


def main():
    args = parse_args()
    from .helper_scripts.io_utils import load_image_data, ensure_output_dir, parse_slices

    if args.mode == 'npw':
        output_dir = ensure_output_dir(args.output_dir)
        mtf_data = np.load(args.mtf_npz)
        nps_data = np.load(args.nps_npz)

        diameters = None
        if args.disc_diameters:
            diameters = [float(x) for x in args.disc_diameters.split(',')]

        result = get_d_prime_npw(
            mtf_data['mtf_freq'], mtf_data['mtf'],
            nps_data['nps_freq'], nps_data['nps'],
            disc_diameters_mm=diameters,
            contrast_hu=args.contrast,
            plot_results=not args.no_plot,
            target_directory=output_dir,
        )
        dp_header = "d'"
        print(f"\nNPW Detectability Index (ΔC = {result['contrast_hu']:.0f} HU)")
        print(f"{'Diameter (mm)':>14s}  {dp_header:>8s}  {'Detectable':>10s}")
        print('-' * 36)
        for d, dp in zip(result['disc_diameters_mm'], result['d_prime']):
            flag = 'Yes' if dp >= 3.0 else 'No'
            print(f'{d:>14.2f}  {dp:>8.1f}  {flag:>10s}')

    else:
        # TTF mode (original behaviour / default)
        image_data_TTF = load_image_data(args.input_ttf)
        if args.slices_ttf is not None:
            image_data_TTF = image_data_TTF[parse_slices(args.slices_ttf)]

        image_data_NPS = load_image_data(args.input_nps)
        if args.slices_nps is not None:
            image_data_NPS = image_data_NPS[parse_slices(args.slices_nps)]

        materials = [m.strip() for m in args.materials.split(',')]
        output_dir = ensure_output_dir(args.output_dir)

        task_function_data = create_circular_task_function(
            args.task_contrast, args.task_object_size,
            pixel_size=args.pixel_size,
            image_dimension=np.min(image_data_TTF.shape[1:])
        )

        _ = get_d_prime(image_data_TTF, args.centre_pixels, args.radius, materials,
                        image_data_NPS, args.roi_bounds,
                        task_function_data=task_function_data,
                        task_function_material=args.task_material,
                        task_function_object_size=args.task_object_size,
                        pixel_size=args.pixel_size, verbose=not args.quiet,
                        plot_results=not args.no_plot, target_directory=output_dir)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
