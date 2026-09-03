# Description: This script calculates the Noise Power Spectrum (NPS) of an image data set.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.integrate
from .helpers.radial import radial_profile as RadialProfile

# https://en.wikipedia.org/wiki/Spectral_density is useful to understand the concept of NPS
# https://www.sciencedirect.com/science/article/pii/S1120179715003294 is a good academic paper to cross-check understanding

def get_NPS(image_data, ROI_bounds, pixel_size, target_directory=os.getcwd(), plot_results=False, filter_low_freq=False, roi_shape="square"):
    """
    This function calculates the Noise Power Spectrum (NPS) on image data.
    :param image_data: 3D numpy array containing the image data
    :param ROI_bounds: 2D numpy array containing the edge coordinates for the ROIs (y1, y2, x1, x2)
    :param pixel_size: pixel size in mm
    :param target_directory: directory where the final NPS plot will be saved
    :param plot_results: Boolean to plot the results or not
    :param filter_low_freq: Boolean to filter the low frequency components of the FFT
    :param roi_shape: "square" (default) or "circular". Controls masking and
        normalization area. "square" normalizes by Nx*Ny (full ROI).
        "circular" applies a circular mask (inscribed circle) before the FFT
        and normalizes by pi*r^2.
    :return: NPS_freqs_transverse: 1D numpy array containing the transverse frequencies
    :return: NPS: 3D numpy array containing NPS values for each slice
    """

    # if only one image 2d data is provided, convert it to 3d
    if len(image_data.shape) == 2:
        image_data = image_data[np.newaxis, :, :]

    # if only one crop area is provided, convert it to a list of crop areas
    if len(ROI_bounds.shape) == 1:
        ROI_bounds = ROI_bounds[np.newaxis, :]

    # check that all ROIs will be the same size
    if not (np.all(ROI_bounds[:, 1] - ROI_bounds[:, 0] == ROI_bounds[0, 1] - ROI_bounds[0, 0]) and
            np.all(ROI_bounds[:, 3] - ROI_bounds[:, 2] == ROI_bounds[0, 3] - ROI_bounds[0, 2])):
        raise ValueError('All ROIs must be the same size')

    # check that ROI x and y dimensions are equal (square)
    if not (np.all((ROI_bounds[:, 1] - ROI_bounds[:, 0]) == (ROI_bounds[:, 3] - ROI_bounds[:, 2]))):
        raise ValueError('All ROIs must have equal x and y dimensions')

    Ny = int(ROI_bounds[0, 1] - ROI_bounds[0, 0])
    Nx = int(ROI_bounds[0, 3] - ROI_bounds[0, 2])

    if roi_shape == "circular":
        # Circular mask: inscribed circle within the square ROI
        cy, cx = Ny / 2.0, Nx / 2.0
        yy, xx = np.ogrid[:Ny, :Nx]
        r = min(Ny, Nx) / 2.0
        circular_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        norm_area = np.pi * r ** 2
    else:
        circular_mask = None
        norm_area = Ny * Nx

    # create an array of the cropped ROI regions
    ROI_array = np.empty((len(ROI_bounds), image_data.shape[0], ROI_bounds[0, 1] - ROI_bounds[0, 0], ROI_bounds[0, 3] - ROI_bounds[0, 2]))
    for i in range(len(ROI_bounds)):
        ROI_array[i] = image_data[:,ROI_bounds[i][0]:ROI_bounds[i][1], ROI_bounds[i][2]:ROI_bounds[i][3]].astype(np.float64)


    # Calculate the power spectrum of each ROI of each slice
    ROI_ffts = np.empty(ROI_array.shape)
    for i in range(len(ROI_array)): # loop over the ROIs
        for j in range(ROI_array.shape[1]): # loop over the slices
            # define the background ROI indices
            ROI_X, ROI_Y = np.arange(ROI_array.shape[3]), np.arange(ROI_array.shape[2])

            # fit a low-order polynomial to the ROI to get the background
            ROI_X = np.polynomial.polynomial.polyval(ROI_X, np.polynomial.polynomial.polyfit(ROI_X, np.mean((ROI_array[i][j]-np.mean(ROI_array[i][j])), axis=0), 3))
            ROI_Y = np.polynomial.polynomial.polyval(ROI_Y, np.polynomial.polynomial.polyfit(ROI_Y, np.mean((ROI_array[i][j]-np.mean(ROI_array[i][j])), axis=1), 3))

            # average the polynomial fit over the two dimensions
            ROI_background = np.mean(ROI_array[i][j])+(ROI_Y[:, np.newaxis] + ROI_X[np.newaxis, :])/2

            # calculate the 2d fft of the ROI and subtract the background
            detrended = ROI_array[i][j] - ROI_background
            if circular_mask is not None:
                detrended = detrended * circular_mask
            fft = np.fft.fftshift(np.fft.fft2(detrended))
            # filter the fft to remove the low frequency components (filters 1% of the lowest freq component radially)
            if filter_low_freq:
                fft[(np.sqrt((np.ogrid[:fft.shape[0], :fft.shape[1]][1] - int(fft.shape[0] / 2))**2
                           + (np.ogrid[:fft.shape[0], :fft.shape[1]][0] - int(fft.shape[1] / 2))**2) <= int(min(fft.shape) / 100))] = 0

            # calculate the power spectrum
            ROI_ffts[i][j] = np.abs(fft)**2


    # Calculate the NPS from the ROI_ffts by averaging over ROIs and correcting for spatial units
    NPS = (pixel_size ** 2 * np.sum(ROI_ffts, axis=0) / (norm_area * ROI_ffts.shape[0]))


    # Calculate the radial average of the NPS in the transverse plane
    NPS_radial_avg = np.empty((ROI_array.shape[1], int(ROI_array.shape[2]/2-1)))
    for j in range(ROI_array.shape[1]): # loop over the slices
        # get the radial profile of the NPS and store it in the NPS_radial_avg array
        radialprofile = RadialProfile(NPS[j], (int(ROI_array.shape[3]/2), int(ROI_array.shape[2]/2)), np.arange(0, int(ROI_array.shape[2]/2)))
        NPS_radial_avg[j] = radialprofile.profile


    # Calculate the transverse (in plane) frequencies of the FFT
    NPS_freqs_transverse = np.fft.fftshift(np.fft.fftfreq(NPS_radial_avg.shape[1], d=pixel_size))

    # get the noise variance of each 2d slice by integration of the NPS and then average over all slices
    noise_variance = np.array([])
    freqs = np.fft.fftshift(np.fft.fftfreq(NPS.shape[1], d=pixel_size))
    for i in range(NPS.shape[0]):
        # find the 2d integration by simpson rule and append to the noise_variance array
        noise_variance = np.append(noise_variance, scipy.integrate.simpson(scipy.integrate.simpson(NPS[i], x=freqs, axis=0), x=freqs))

    noise_variance = np.mean(noise_variance)

    # --------------------------------- Plotting the combined images ---------------------------------
    if plot_results:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=80)

        # plot the original image averaged over all slices
        image = axs[0, 0].pcolormesh(np.arange(0, pixel_size * image_data.shape[2], pixel_size),
                        np.arange(0, pixel_size * image_data.shape[1], pixel_size),
                        np.mean(image_data, axis=0), shading='auto', cmap='gray')
        axs[0, 0].set_title('Transverse view of phantom averaged over all slices')
        axs[0, 0].set_xlabel('X (mm)')
        axs[0, 0].set_ylabel('Y (mm)')
        fig.colorbar(image, ax=axs[0, 0], label='HU')
        # create ROI overlays matching the roi_shape used for computation
        for i in range(len(ROI_bounds)):
            if roi_shape == "circular":
                axs[0, 0].add_patch(plt.Circle((int((0.5*ROI_bounds[i][2]+0.5*ROI_bounds[i][3])*pixel_size),
                                                int((0.5*ROI_bounds[i][1]+0.5*ROI_bounds[i][0])*pixel_size)),
                                                0.5*(ROI_bounds[i][3]-ROI_bounds[i][2])*pixel_size, edgecolor='r',
                                                facecolor='none'))
            else:
                x0 = ROI_bounds[i][2] * pixel_size
                y0 = ROI_bounds[i][0] * pixel_size
                w = (ROI_bounds[i][3] - ROI_bounds[i][2]) * pixel_size
                h = (ROI_bounds[i][1] - ROI_bounds[i][0]) * pixel_size
                axs[0, 0].add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='r',
                                                   facecolor='none'))


        # plot the transverse plane of the NPS averaged over all slices
        NPS_mesh_transverse = axs[0, 1].pcolormesh(np.fft.fftshift(np.fft.fftfreq(NPS.shape[1], d=pixel_size)),
                                                    np.fft.fftshift(np.fft.fftfreq(NPS.shape[1], d=pixel_size)),
                                                    np.mean(NPS, axis=0), shading='auto')
        axs[0, 1].set_title('Mean NPS in Transverse Plane')
        axs[0, 1].set_xlabel('Radial Average Frequency (mm$^{-1}$)')
        axs[0, 1].set_ylabel('Radial Average Frequency (mm$^{-1}$)')
        fig.colorbar(NPS_mesh_transverse, ax=axs[0, 1], label='NPS (HU$^2$ mm$^2$)')


        # plot the radially averaged NPS for each z-axis slice of the data
        NPS_mesh = axs[1, 0].pcolormesh((radialprofile.radius*np.max(NPS_freqs_transverse)/np.max(radialprofile.radius)),
                                        np.arange(NPS_radial_avg.shape[0]), NPS_radial_avg, shading='auto')
        axs[1, 0].set_title('Radially averaged NPS in each Transverse Plane')
        axs[1, 0].set_xlabel('Radial Average Frequency (mm$^{-1}$)')
        axs[1, 0].set_ylabel('Slice Number (z-axis)')
        axs[1, 0].set_xlim(0, np.max(NPS_freqs_transverse))
        fig.colorbar(NPS_mesh, ax=axs[1, 0], label='NPS (HU$^2$ mm$^2$)')


        # plot the radially averaged NPS from all transverse planes
        axs[1, 1].plot((radialprofile.radius*np.max(NPS_freqs_transverse)/np.max(radialprofile.radius)), np.mean(NPS_radial_avg, axis=0), label='NPS')
        axs[1, 1].set_xlim(0, np.max(NPS_freqs_transverse))
        axs[1, 1].set_title('Radially averaged NPS from all Transverse Planes')
        axs[1, 1].set_xlabel('Radial Average Frequency (mm$^{-1}$)')
        axs[1, 1].set_ylabel('NPS (HU$^2$ mm$^2$)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].text(0.9, 0.88, f'$\sigma ^2$ = {noise_variance:.0f} HU$^2$', horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)
        # save the plot
        plt.tight_layout()
        plt.savefig(target_directory + '/NPS_results.png', dpi=300)
        plt.savefig(target_directory + '/NPS_results.pdf', dpi=300)
        plt.close('all')

    return (radialprofile.radius*np.max(NPS_freqs_transverse)/np.max(radialprofile.radius)), np.mean(NPS_radial_avg, axis=0)

def parse_args(argv=None):
    import argparse
    from .helpers.io_utils import parse_roi_bounds
    parser = argparse.ArgumentParser(
        prog='ct-nps', description='Noise power spectrum of a CT volume from square ROIs. Writes '
                                   'nps.npz + nps.json (curve, noise sigma, peak/mean frequency).')
    parser.add_argument('--input', required=True, help='Path to image file (.npy, .npz, or .vff)')
    parser.add_argument('--roi_bounds', required=True, type=parse_roi_bounds,
                        help='ROI bounds as "y1,y2,x1,x2;y1,y2,x1,x2;..." (all square, same size)')
    parser.add_argument('--slices', type=str, default=None,
                        help='Slice selection (e.g. "10:160" or "0:30,140:182")')
    parser.add_argument('--pixel_size', type=float, required=True, help='Pixel size in mm')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--no_plot', action='store_true', help='Skip the diagnostic figure')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from .api import nps as _api_nps
    from .helpers.io_utils import load_image_data, ensure_output_dir, parse_slices

    image_data = load_image_data(args.input)
    slices = parse_slices(args.slices) if args.slices else None
    output_dir = ensure_output_dir(args.output_dir)
    result = _api_nps(image_data, args.roi_bounds, args.pixel_size, slices=slices,
                      plot_dir=None if args.no_plot else output_dir)
    npz, js = result.save(os.path.join(output_dir, 'nps'))
    if not args.quiet:
        print(result.describe())
        print(f"written: {npz}, {js}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
