# Description: This script calculates the Detectability index (d') of an image data set
# using the TTF, NPS, and W (task function)
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib.pyplot as plt
from photutils.profiles import RadialProfile
from scipy.interpolate import CubicSpline
import scipy.integrate
from . import ttf_calculator as TTF_calculator
from . import nps_calculator as NPS_calculator

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
    parser = argparse.ArgumentParser(description='Calculate the Detectability Index (d\') from CT image data.')
    parser.add_argument('--input_ttf', required=True, help='Path to TTF image file (.npy, .npz, or .vff)')
    parser.add_argument('--input_nps', required=True, help='Path to NPS image file (.npy, .npz, or .vff)')
    parser.add_argument('--centre_pixels', required=True, type=parse_centre_pixels,
                        help='Centre pixels as semicolon-separated y,x pairs (e.g. "686,398;418,132")')
    parser.add_argument('--radius', required=True, type=int, help='Radius of circular edges in pixels')
    parser.add_argument('--materials', required=True, type=str,
                        help='Comma-separated material names (e.g. "SB3,Teflon,Fat,Tissue")')
    parser.add_argument('--roi_bounds', required=True, type=parse_roi_bounds,
                        help='NPS ROI bounds as semicolon-separated y1,y2,x1,x2 groups')
    parser.add_argument('--task_material', required=True, type=str, help='Task function material (e.g. "Fat")')
    parser.add_argument('--task_contrast', type=float, default=-160, help='Task function contrast in HU (default: -160)')
    parser.add_argument('--task_object_size', type=float, default=0.2, help='Task function object radius in mm (default: 0.2)')
    parser.add_argument('--slices_ttf', type=str, default=None, help='Slice selection for TTF data')
    parser.add_argument('--slices_nps', type=str, default=None, help='Slice selection for NPS data')
    parser.add_argument('--pixel_size', type=float, default=0.05, help='Pixel size in mm (default: 0.05)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory (default: ./results)')
    parser.add_argument('--no_plot', action='store_true', help='Disable plot generation')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')
    return parser.parse_args()


def main():
    args = parse_args()
    from .helper_scripts.io_utils import load_image_data, ensure_output_dir, parse_slices

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
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
