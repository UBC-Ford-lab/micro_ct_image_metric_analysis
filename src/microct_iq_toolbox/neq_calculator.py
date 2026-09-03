# Description: This script calculates the Noise-Equivalent Quanta (NEQ) of an image data set.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from . import nps_calculator as NPS_calculator
from . import mtf_calculator as MTF_calculator

def neq_from_curves(mtf_freq, mtf, nps_freq, nps, n_freq=100, signal=None):
    """NEQ from an MTF and an NPS that have ALREADY been measured.

    ``get_NEQ`` measures both itself, which means a caller who has already run
    ``get_MTF`` and ``get_NPS`` — to report them in the same table — pays for
    them twice AND can end up with an NEQ built from a different MTF than the
    one it published, because ``get_NEQ`` used to fix ``edge_angle`` at 5.5
    regardless of what the caller passed elsewhere. Working from the curves
    removes both problems.

    Only the OVERLAP of the two frequency axes is used. They are not the same
    axis in general — ``get_TTF``/``get_MTF`` and ``get_NPS`` reach different
    Nyquists — and extrapolating either one past its own limit invents the
    part of the curve that decides the answer.

    ``signal`` is the large-area signal level the classical definition
    multiplies by. Leave it None for the ratio MTF^2/NPS, which is what ranks
    two reconstructions of the SAME object: the factor is common to them and
    on an HU volume it is an arbitrary constant that depends on how much air
    the measurement region happened to contain.

    Returns ``(freqs, NEQ)``.
    """
    mf = np.asarray(mtf_freq, dtype=np.float64)
    mv = np.asarray(mtf, dtype=np.float64)
    nf = np.asarray(nps_freq, dtype=np.float64)
    nv = np.asarray(nps, dtype=np.float64)
    lo = max(float(np.min(np.abs(mf))), float(np.min(np.abs(nf))))
    hi = min(float(np.max(mf)), float(np.max(nf)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.array([])
    freqs = np.linspace(lo, hi, int(n_freq))
    mtf_i = interp1d(mf, mv, bounds_error=False, fill_value='extrapolate')(freqs)
    nps_i = interp1d(nf, nv, bounds_error=False, fill_value='extrapolate')(freqs)
    with np.errstate(divide='ignore', invalid='ignore'):
        neq = np.where(nps_i > 0, mtf_i ** 2 / nps_i, np.nan)
    if signal is not None:
        neq = neq * float(signal) ** 2
    return freqs, neq


def get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size, target_directory=os.getcwd(),
            plot_results=False, high_to_low_MTF=True, edge_angle=5.5):
    """
    This function calculates the Noise-Equivalent Quanta (NEQ) on image data. It uses the MTF and NPS functions from
    the MTF_calculator and NPS_calculator scripts respectively.
    :param image_data_MTF: The image data for the MTF calculation as a 3D numpy array (z, y, x)
    :param image_data_NPS: The image data for the NPS calculation as a 3D numpy array (z, y, x)
    :param crop_indices_MTF: Specify the region of interest to crop the images for the MTF calculation (y1, y2, x1, x2)
    :param ROI_bounds_NPS: Specify the region of interest to crop the images for the NPS calculation (y1, y2, x1, x2)
    :param pixel_size: The pixel size of the images in mm
    :param target_directory: Where to save the resulting plot
    :param plot_results: Whether to plot the results
    :param high_to_low_MTF: Whether the MTF edge goes from high to low pixel values (True) or low to high pixel values (False)
    :param edge_angle: The angle of the MTF edge in degrees. Must match what
        you pass to get_MTF, or the NEQ is built on a different MTF than the
        one you reported.
    :return: freqs: The frequency axis of the NEQ, NEQ: The Noise-Equivalent Quanta from interp1d interpolation of the MTF and NPS
    """

    # Calculate the MTF
    #
    # edge_angle is a PARAMETER now. It used to be pinned at 5.5 degrees here
    # while get_MTF took it from the caller, so an NEQ computed on a phantom
    # whose edge is at some other angle silently disagreed with the MTF the
    # same caller had just measured.
    MTF_freq, MTF = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=pixel_size,
      target_directory=target_directory, plot_results=False, edge_angle=edge_angle, high_to_low=high_to_low_MTF)

    freqs = np.linspace(np.sort(np.abs(MTF_freq))[0], MTF_freq[-1], 100) # freqs up to Nyquist frequency

    # Interpolate the MTF to the desired frequency range
    MTF_spline = interp1d(MTF_freq, MTF, bounds_error=False, fill_value='extrapolate')(freqs)

    # Calculate the NPS
    NPS_freq, NPS = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=pixel_size, target_directory=target_directory,
     plot_results=False)

    # Interpolate the NPS to the desired frequency range
    NPS_spline = interp1d(NPS_freq, NPS, bounds_error=False, fill_value='extrapolate')(freqs)

    # create an array of the cropped ROI regions for NPS
    ROI_NPS = np.empty((len(ROI_bounds_NPS), image_data_NPS.shape[0], ROI_bounds_NPS[0, 1] - ROI_bounds_NPS[0, 0],
                         ROI_bounds_NPS[0, 3] - ROI_bounds_NPS[0, 2]))
    for i in range(len(ROI_bounds_NPS)):
        ROI_NPS[i] = image_data_NPS[:,ROI_bounds_NPS[i][0]:ROI_bounds_NPS[i][1], ROI_bounds_NPS[i][2]:ROI_bounds_NPS[i][3]].astype(np.float64)

    # Calculate the NEQ
    NEQ = ((np.mean(image_data_MTF[:, crop_indices_MTF[0]:crop_indices_MTF[1], crop_indices_MTF[2]:crop_indices_MTF[3]])+
            np.mean(ROI_NPS))/2)**2 * MTF_spline**2 / NPS_spline

    if plot_results:
        # Plot the NEQ
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        ax1 = axs[0, 0]
        ax1.plot(MTF_freq, MTF)
        ax1.plot(freqs, MTF_spline, 'r--', label='Cubic Spline fit')
        ax1.set_xlabel('Spatial Frequency (mm$^{-1}$)')
        ax1.set_ylabel('MTF')
        ax1.set_title('Modulation Transfer Function (MTF) Plot')
        ax1.set_xlim([0, np.max(freqs)])
        ax1.legend()
        ax1.grid(True)

        ax2 = axs[0, 1]
        ax2.plot(NPS_freq, NPS)
        ax2.plot(freqs, NPS_spline, 'r--', label='Cubic Spline fit')
        ax2.set_xlabel('Spatial Frequency (mm$^{-1}$)')
        ax2.set_ylabel('NPS (HU$^2$ mm$^2$)')
        ax2.set_title('Noise Power Spectrum (NPS) Plot')
        ax2.set_xlim([0, np.max(freqs)])
        ax2.legend()
        ax2.grid(True)

        ax3 = axs[1, 0]
        ax3.plot(freqs, NEQ)
        ax3.set_xlabel('Spatial Frequency (mm$^{-1}$)')
        ax3.set_ylabel('NEQ (mm$^{-2}$)')
        ax3.set_title('Noise-Equivalent Quanta (NEQ) Plot')
        ax3.grid(True)

        axs[1,1].set_axis_off()

        plt.tight_layout()
        plt.savefig(os.path.join(target_directory, 'NEQ_plot.png'))
        plt.savefig(os.path.join(target_directory, 'NEQ_plot.pdf'))
        plt.close('all')

    return freqs, NEQ

def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog='ct-neq', description='NEQ = MTF^2 / NPS from saved ct-mtf and ct-nps results. '
                                   'Writes neq.npz + neq.json.')
    parser.add_argument('--mtf', required=True, help='mtf.npz (or its stem) written by ct-mtf')
    parser.add_argument('--nps', required=True, help='nps.npz (or its stem) written by ct-nps')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--no_plot', action='store_true', help='Skip the figure')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from .api import neq as _api_neq
    from .results import MetricResult
    from .helpers.io_utils import ensure_output_dir

    output_dir = ensure_output_dir(args.output_dir)
    result = _api_neq(MetricResult.load(args.mtf), MetricResult.load(args.nps))
    npz, js = result.save(os.path.join(output_dir, 'neq'))
    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result.x, result.y)
        ax.set_xlabel('Spatial frequency (mm$^{-1}$)'); ax.set_ylabel('NEQ')
        ax.set_title('Noise-equivalent quanta'); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'NEQ_plot.png'), dpi=300)
        if not args.show:
            plt.close(fig)
    if not args.quiet:
        print(result.describe())
        print(f"written: {npz}, {js}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
