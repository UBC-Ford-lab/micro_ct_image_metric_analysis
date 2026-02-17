# Description: This script calculates the Noise-Equivalent Quanta (NEQ) of an image data set.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from . import nps_calculator as NPS_calculator
from . import mtf_calculator as MTF_calculator

def get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size, target_directory=os.getcwd(),
            plot_results=True, high_to_low_MTF=True):
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
    :return: freqs: The frequency axis of the NEQ, NEQ: The Noise-Equivalent Quanta from interp1d interpolation of the MTF and NPS
    """

    # Calculate the MTF
    MTF_freq, MTF = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=pixel_size,
      target_directory=target_directory, plot_results=False, edge_angle=5.5, high_to_low=high_to_low_MTF)

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

    return freqs, NEQ

if __name__ == '__main__':
    # Load the image data
    from reconstruction.ct_core import vff_io as vff
    image_data = vff.read_vff("data/results/repaint_reconstruction.vff", verbose=False)[1]

    # select MTF slices
    # Define the MTF crop indices (y1, y2, x1, x2)
    image_data_MTF = image_data[228:229, :, :]
    crop_indices_MTF = [270, 664, 522, 640]

    # select NPS slices
    # Define the ROIs over which to compute the NPS (y1, y2, x1, x2) in pixel coordinates
    image_data_NPS = image_data[np.concatenate((np.arange(204, 208), np.arange(216, 226))), :, :]
    ROI_bounds_NPS = np.array([[178, 294, 510, 626], [258, 374, 750, 866], [310, 426, 328, 444], [432, 548, 830, 946], [488, 604, 248, 364], [580, 696, 730, 846], [624, 740, 414, 530], [724, 840, 598, 714]])


    _ = get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.085, target_directory=os.getcwd(),
                plot_results=True)


'''
For data/results/ground_truth_reconstruction.vff
    # select MTF slices
    # Define the MTF crop indices (y1, y2, x1, x2)
    image_data_MTF = image_data[228:229, :, :]
    crop_indices_MTF = [270, 664, 522, 640]

    # select NPS slices
    # Define the ROIs over which to compute the NPS (y1, y2, x1, x2) in pixel coordinates
    image_data_NPS = image_data[np.concatenate((np.arange(204, 208), np.arange(216, 226))), :, :]
    ROI_bounds_NPS = np.array([[178, 294, 510, 626], [258, 374, 750, 866], [310, 426, 328, 444], [432, 548, 830, 946], [488, 604, 248, 364], [580, 696, 730, 846], [624, 740, 414, 530], [724, 840, 598, 714]])

'''
