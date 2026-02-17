# Description: This script calculates the Task Transfer Function (TTF) of an image data set
# using a circular edge test pattern.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from photutils.profiles import RadialProfile

def get_TTF(image_data, centre_pixels, radius, materials=None, find_absolute_TTF=True, pixel_size=0.05,
            target_directory=os.getcwd(), plot_results=True, process_LSF=True):
    """
    This function calculates the Task Transfer Function (TTF) on image data.
    :param image_data: The image data as a 3D numpy array (z, y, x)
    :param centre_pixels: The centre of the circular edge in pixel coordinates (y, x)
    :param radius: The radius of the circular edges in pixels
    :param materials: The type of materials in the circular edge test pattern
    :param find_absolute_TTF: Boolean to calculate the absolute TTF (True) or the relative TTF (False)
    :param pixel_size: The pixel size of the images in mm (set to 1 for relative TTF)
                        (Note: you need to set the pixel size manually!)
    :param target_directory: Where to save the cropped image (if desired)
    :param plot_results: Boolean to plot the results or not
    :return: TTF_freq: The frequency axis of the TTF
    :return: TTF_array: The Task Transfer Functions for each material
    :return: CNR_array: The Contrast-to-Noise Ratio for each material
    """
    if process_LSF:
        from .helper_scripts import lsf_processing as LSF_processing

    # Define the sampling pixel increment. Determines how much it's supersampled
    sampling_pixel_increment = 1/4

    # For relative TTF calculation it is done per pixel (so =1)
    if find_absolute_TTF==False:
        pixel_size = 1

    # if only one image 2d data is provided, convert it to 3d
    if len(image_data.shape) == 2:
        image_data = image_data[np.newaxis, :, :]

    # define the array length along the radius
    # this is important and should be smaller than actual radius due to instabilities in the central region
    radial_length = int(0.85*radius/sampling_pixel_increment-1)


    # create empty arrays for the ERF, LSF
    ERF_array = np.empty((len(centre_pixels), image_data.shape[0], radial_length))
    LSF_array = np.empty((len(centre_pixels), radial_length))
    LSF_x_axis = np.linspace((radius-radial_length*sampling_pixel_increment)*pixel_size, radius*pixel_size, LSF_array.shape[1])

    # now the ROIs are iterated through
    for i in range(len(centre_pixels)):
        # the circular ROIs are created
        rectangular_ROI = image_data[:, centre_pixels[i][0]-radius:centre_pixels[i][0]+radius,
                                        centre_pixels[i][1]-radius:centre_pixels[i][1]+radius].astype(np.float64)

        # iterate through the slices
        for j in range(image_data.shape[0]):
            # print progress in percentage
            if int(100*(i+j/image_data.shape[0])/len(centre_pixels)) % 10 == 0 and int(100*(i+(j-1)/image_data.shape[0])/len(centre_pixels)) % 10 != 0:
                print(f'TTF calculation Progress: {100*(i+j/image_data.shape[0])/len(centre_pixels):.1f}%')

            # calculate the ERF using the RadialProfile class
            centre = (int(rectangular_ROI[j].shape[1]/2), int(rectangular_ROI[j].shape[0]/2))
            radial_profile = RadialProfile(rectangular_ROI[j], centre, np.arange(0, radius, sampling_pixel_increment))
            ERF_array[i, j] = radial_profile.profile[-radial_length:]

        # calculate the LSF of each ROI
        LSF = np.abs(np.gradient(np.mean(ERF_array[i], axis=0), pixel_size*sampling_pixel_increment)) # calculate the Line Spread Function

        # process the LSF
        if process_LSF:

            # change the LSF length if processing is required (since a lot is cut from the signal)
            if i == 0:
                LSF_x_axis_original = LSF_x_axis.copy()
                LSF_x_axis, LSF = LSF_processing.process_LSF(LSF_x_axis, LSF, pixel_size)

                LSF_array = np.empty((len(centre_pixels), len(LSF)))
                TTF_array = np.empty((len(centre_pixels), len(LSF)))

            else:
                # after the first iteration the array is constructed yet the new LSFs still may be a different length
                # this is only a small difference because the arrays are often very similar in size
                LSF_x_axis_new, LSF = LSF_processing.process_LSF(LSF_x_axis_original, LSF, pixel_size)

                if LSF_array.shape[1] < len(LSF):
                    # pad LSF_array with zeros on both sides
                    LSF_array = np.pad(LSF_array, ((0, 0), (int(np.floor(0.5*(len(LSF)-LSF_array.shape[1]))), int(np.ceil(0.5*(len(LSF)-LSF_array.shape[1]))))))
                    LSF_x_axis = LSF_x_axis_new
                elif LSF_array.shape[1] > len(LSF):
                    # pad LSF with zeros on both sides
                    LSF = np.pad(LSF, (int(np.floor(0.5*(LSF_array.shape[1]-len(LSF)))), int(np.ceil(0.5*(LSF_array.shape[1]-len(LSF))))))

            # apply a Hamming window to it (this was an old processing step instead of the one above (is done by ISO 12233 standards))
            #LSF = LSF*np.hamming(len(LSF))

        LSF_array[i] = LSF

    # calculate the TTF for each material
    TTF_array = np.empty((len(centre_pixels), len(LSF_array[1])))
    for i in range(len(centre_pixels)):
        # calculate the Task Transfer Function
        TTF_array[i] = np.abs(np.fft.fftshift(np.fft.fft(LSF_array[i])))
        # normalise the TTF
        TTF_array[i] = TTF_array[i]/TTF_array[i].max()

    # find the CNRs for each material
    CNR_array = np.array([])
    for i in range(len(centre_pixels)): # iterate through the ROIs
        # find CNR (by taking the difference of first 10% to last 10% of the ERF and dividing by std of background)
        # do this for every slice, then average. See AAPM 233 report
        slice_CNRs = np.array([])
        for j in range(ERF_array[i].shape[0]):
            CNR = (np.abs(np.mean(ERF_array[i,j,:int(0.15*ERF_array[i].shape[1])])-
                        np.mean(ERF_array[i,j,-int(0.15*ERF_array[i].shape[1]):]))/
                        np.std(ERF_array[i,j,-int(0.15*ERF_array[i].shape[1]):]))
            slice_CNRs = np.append(slice_CNRs, CNR)
        slice_CNRs = np.mean(slice_CNRs)
        CNR_array = np.append(CNR_array, slice_CNRs)

    # assign weights to the TTFs based on the CNR (0 if CNR < 5, 1 if CNR > 5)
    weights = np.where(CNR_array > 5, 1, 0)

    # calculate the TTF (average over all ROIs and all slices with weights based on CNR)
    TTF = np.average(TTF_array, axis=0, weights=weights)
    # calculate the TTF frequency axis
    TTF_freq = np.fft.fftshift(np.fft.fftfreq(len(TTF), d=pixel_size))

    # interpolate the TTF to find TTF50 and TTF10 (makes it more accurate)
    TTF_freq_interpolated = np.linspace(0, np.max(TTF_freq), 1000)
    TTF_interpolated = np.interp(TTF_freq_interpolated, TTF_freq[TTF_freq>0], TTF[TTF_freq>0])

    # find TTF50
    try:
        TTF50_index = np.array(np.where((TTF_interpolated < 0.5) & (TTF_freq_interpolated > 0))[0])[0]
        TTF50 = TTF_freq_interpolated[TTF50_index]
    except:
        TTF50_index = len(TTF_interpolated)-1
        TTF50 = 1/(2*pixel_size)

    # find TTF10
    try:
        TTF10_index = np.array(np.where((TTF_interpolated < 0.1) & (TTF_freq_interpolated > 0))[0])[0]
        TTF10 = TTF_freq_interpolated[TTF10_index]
    except:
        TTF10_index = len(TTF_interpolated)-1
        TTF10 = 1/(2*pixel_size)


    if plot_results:
        # Plot the results
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]

        # Plotting the combined images
        ax1.imshow(np.mean(image_data, axis=0), cmap='gray')

        # Plot the circular ROIs
        for i in range(len(centre_pixels)):
            ellipse = matplotlib.patches.Ellipse((centre_pixels[i][1], centre_pixels[i][0]),
                     radius*2, radius*2, edgecolor='red', facecolor='none')
            ax1.add_patch(ellipse)

        # Plot the ERF, LSF, TTF, and TTF average
        for i in range(len(centre_pixels)): # iterate through the ROIs
            ax2.plot(radial_profile.radius[-radial_length:]*pixel_size, np.mean(ERF_array[i], axis=0), label='ERF '+materials[i], alpha=0.5)
            ax3.plot(LSF_x_axis, LSF_array[i], label='LSF '+materials[i], alpha=0.5)
            if CNR_array[i] > 5:
                ax4.plot(TTF_freq, TTF_array[i], label=materials[i]+f' TTF for CNR={CNR_array[i]:.2f}', alpha=0.3)
            else:
                ax4.plot(np.nan, np.nan, label=materials[i]+f' TTF for CNR={CNR_array[i]:.2f}', alpha=0.3)

        ax4.plot(TTF_freq, TTF, color='black', label=f'Average TTF for CNR={np.mean(CNR_array):.2f}')

        # add the TTF50 and TTF10 to the plot
        ax4.axvline(TTF50, 0, TTF_interpolated[TTF50_index], color='black', linestyle='-.', label='Average $TTF_{50}$ ='+f' {TTF50:.2f}')
        ax4.axvline(TTF10, 0, TTF_interpolated[TTF10_index], color='black', linestyle='--', label='Average $TTF_{10}$ ='+f' {TTF10:.2f}')
        ax4.axhline(0.5, 0, TTF50/(1/(2*pixel_size)), color='black', linestyle='-.')
        ax4.axhline(0.1, 0, TTF10/(1/(2*pixel_size)), color='black', linestyle='--')

        # setting plot parameters
        ax1.axis('off')
        ax1.set_title('Cropped phantom scan (averaged)')
        if find_absolute_TTF==False:
            ax2.set_xlabel('Radial distance (pixels)')
        else:
            ax2.set_xlabel('Radial distance (mm)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Edge Response Function')
        ax2.legend()
        ax2.grid(True)
        if find_absolute_TTF==False:
            ax3.set_xlabel('Radial distance (pixels)')
        else:
            ax3.set_xlabel('Radial distance (mm)')
        ax3.set_ylabel('Intensity (gradient of the ERF)')
        ax3.set_title('Line Spread Function')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xlim([np.min(radial_profile.radius*pixel_size), np.max(radial_profile.radius*pixel_size)])
        if find_absolute_TTF==False:
            ax4.set_xlabel('Spatial Frequency (Cycles per pixel)')
        else:
            ax4.set_xlabel('Spatial Frequency (lp per mm)')
        ax4.set_xlim([0, 1/(2*pixel_size)]) # twice the Nyquist frequency (allowed since supersampled)
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Normalised Modulation (i.e. Contrast)')
        ax4.set_title('Task Transfer Function')
        ax4.legend()
        ax4.grid(True)
        plt.tight_layout()

        # Save the plot
        if find_absolute_TTF == False:
            plt.savefig(target_directory + '/TTF_results_relative.png', dpi=300)
        else:
            plt.savefig(target_directory + '/TTF_results_absolute.png', dpi=300)
        plt.close('all')

    return TTF_freq, TTF_array, CNR_array

def parse_args():
    import argparse
    from .helper_scripts.io_utils import parse_centre_pixels, parse_slices
    parser = argparse.ArgumentParser(description='Calculate the Task Transfer Function (TTF) from CT image data.')
    parser.add_argument('--input', required=True, help='Path to image file (.npy, .npz, or .vff)')
    parser.add_argument('--centre_pixels', required=True, type=parse_centre_pixels,
                        help='Centre pixels as semicolon-separated y,x pairs (e.g. "1335,2141;765,1914")')
    parser.add_argument('--radius', required=True, type=int, help='Radius of circular edges in pixels')
    parser.add_argument('--materials', type=str, default=None,
                        help='Comma-separated material names (e.g. "Teflon,HD POLY,Fat,Tissue")')
    parser.add_argument('--slices', type=str, default=None, help='Slice selection (e.g. "8:45")')
    parser.add_argument('--pixel_size', type=float, default=0.05, help='Pixel size in mm (default: 0.05)')
    parser.add_argument('--relative', action='store_true', help='Calculate relative TTF instead of absolute')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory (default: ./results)')
    parser.add_argument('--no_plot', action='store_true', help='Disable plot generation')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    return parser.parse_args()


def main():
    args = parse_args()
    from .helper_scripts.io_utils import load_image_data, ensure_output_dir, parse_slices

    image_data = load_image_data(args.input)
    if args.slices is not None:
        image_data = image_data[parse_slices(args.slices)]

    materials = None
    if args.materials is not None:
        materials = [m.strip() for m in args.materials.split(',')]

    output_dir = ensure_output_dir(args.output_dir)

    _ = get_TTF(image_data, args.centre_pixels, args.radius, materials=materials,
                find_absolute_TTF=not args.relative, pixel_size=args.pixel_size,
                target_directory=output_dir, plot_results=not args.no_plot)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
