# Description: This script calculates the Modulation Transfer Function (MTF) of an image data set
# using a slanted edge test pattern.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def get_MTF(image_data, crop_indices, find_absolute_MTF=True, pixel_size=0.05,
            target_directory=os.getcwd(), plot_results=True, edge_angle=5.0, high_to_low=True, process_LSF=True, return_ERF=False, normalize_MTF=True):
    """
    This function calculates the Modulation Transfer Function (MTF) on image data. Note that number of data points for the fit,
    needs to be a 4x supersampled version of image data according to ISO 12233
    :param image_data: The image data as a 3D numpy array (z, y, x)
    :param crop_indices: Specify the region of interest to crop the images (y1, y2, x1, x2) as pixel indices
    :param find_absolute_MTF: Boolean to calculate the absolute MTF (True) or the relative MTF (False)
    :param pixel_size: The pixel size of the images in mm (set to 1 for relative MTF)
                        (Note: you need to set the pixel size manually!)
    :param target_directory: Where to save the cropped image (if desired)
    :param plot_results: Boolean to plot the results or not
    :param edge_angle: The angle of the edge in degrees (default is 5 degrees)
    :param high_to_low: Boolean to check if the edge goes from high to low (True) or low to high (False)
    :param process_LSF: Boolean to process the LSF or not (i.e Detrending, windowing and centering)
    :param normalize_MTF: Boolean to normalize the MTF by its maximum value (default is True)
    :return: MTF_freq: The frequency axis of the MTF, MTF: The Modulation Transfer Function
    """
    # Check if the LSF processing is required
    if process_LSF:
        from .helper_scripts import lsf_processing as LSF_processing

    # ROI width is taken as half the ROI height (this is ambiguous and changes the MTF shape)
    if crop_indices[1]-crop_indices[0] != 2*(crop_indices[3]-crop_indices[2]):
        difference = (crop_indices[3]-crop_indices[2])-int(0.5*(crop_indices[1]-crop_indices[0]))
        crop_indices[3] -= int(difference/2)
        crop_indices[2] += int(difference/2)


    # For relative MTF calculation it is done per pixel (so =1)
    if find_absolute_MTF==False:
        pixel_size = 1

    # adding padding to the crop indices due to np.roll later which creates problems at the edges
    crop_indices[2] -= int(1.05*(crop_indices[1]-crop_indices[0])*np.tan(edge_angle*np.pi/180))
    crop_indices[3] += int(1.05*(crop_indices[1]-crop_indices[0])*np.tan(edge_angle*np.pi/180))

    # if only one image 2d data is provided, convert it to 3d
    if len(image_data.shape) == 2:
        image_data = image_data[np.newaxis, :, :]

    # now the image_data is cropped
    image_data = image_data[:, crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]].astype(np.float64)


    # Create arrays to store the results
    array_1_length = (4*image_data.shape[2]-int((crop_indices[1]-crop_indices[0])*np.tan(4*edge_angle*np.pi/180)))//4
    arrays_shape_1 = (image_data.shape[0], array_1_length)
    ERF_array = np.zeros(arrays_shape_1)
    LSF_array = np.zeros(arrays_shape_1)
    LSF_x_axis = np.linspace(0, LSF_array.shape[1]*pixel_size, LSF_array.shape[1])
    MTF_array = np.zeros(arrays_shape_1)

    # iterate through the slices and compute the MTF
    for i in range(image_data.shape[0]):

        # Normalise the image data
        image_data[i] = image_data[i] - image_data[i].min()
        image_data[i] = 255*image_data[i] / image_data[i].max()

        # Calculate the shifted edge response function from all rows
        ERF = np.zeros((4*len(image_data[i][0, :])))
        ERF[::4] += np.array(image_data[i][0, :]) # first we fill the first row into every 4th element of the ERF (4x supersampled)

        for row in range(1, image_data[i].shape[0]):
            # shift the row by the angle of the edge to align the edges
            row_array = np.zeros((4*len(image_data[i][row, :])))
            row_array[::4] += image_data[i][row, :]

            if high_to_low: # check if edge goes from high to low
                ERF += np.roll(row_array, -int(row*np.tan(4*edge_angle*np.pi/180)))
            else: # or low to high
                ERF += np.roll(row_array, int(row*np.tan(4*edge_angle*np.pi/180)))

        # Normalise the ERF back to single image
        ERF /= (image_data[i].shape[0]/4)

        # Crop the ERF to the region of interest (omit edge that has roll 'artefact')
        if high_to_low: # check if edge goes from high to low
            ERF = ERF[:-int((crop_indices[1]-crop_indices[0])*np.tan(4*edge_angle*np.pi/180))]
        else: # or low to high
            ERF = ERF[int((crop_indices[1]-crop_indices[0])*np.tan(4*edge_angle*np.pi/180)):]

        # taken from https://stackoverflow.com/questions/30379311/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
        # this is to average the ERF over 4 pixels (otherwise too noisy)
        ERF = np.cumsum(ERF, 0)[4-1::4]/float(4)
        ERF[1:] = ERF[1:] - ERF[:-1]

        # Apply Savitzky-Golay filter to smooth ERF before differentiation
        # Window=11, polyorder=3 preserves edge shape while reducing noise
        if len(ERF) >= 11:
            ERF = savgol_filter(ERF, window_length=11, polyorder=3)

        # calculate the Line Spread Function
        LSF = np.abs(np.gradient(ERF, pixel_size, edge_order=1))

        # process the LSF
        if process_LSF:
            # change the LSF length if processing is required (since a lot is cut from the signal)
            if i == 0:
                LSF_x_axis_original = np.linspace(0, len(LSF)*pixel_size, len(LSF))
                LSF_x_axis, LSF = LSF_processing.process_LSF(LSF_x_axis_original, LSF, pixel_size)

                LSF_array = np.zeros((image_data.shape[0], len(LSF)))
                MTF_array = np.zeros((image_data.shape[0], len(LSF)))
            else:
                # after the first iteration the array is constructed yet the new LSFs still may be a different length
                # this is only a small difference because the arrays are often very similar in size
                _, LSF = LSF_processing.process_LSF(np.linspace(0, len(LSF)*pixel_size, len(LSF)), LSF, pixel_size)

                if LSF_array.shape[1] > len(LSF):
                    # pad the LSF with zeros if new LSF is shorter
                    LSF = np.pad(LSF, (int(np.floor(0.5*(LSF_array.shape[1]-len(LSF)))), int(np.ceil(0.5*(LSF_array.shape[1]-len(LSF))))))
                elif LSF_array.shape[1] < len(LSF):
                    # crop the zeroes of the LSF if new LSF is longer
                    LSF = LSF[int(np.floor(0.5*(len(LSF)-LSF_array.shape[1]))):int(-np.ceil(0.5*(len(LSF)-LSF_array.shape[1])))]

            # apply a Hamming window to it (this was an old processing step instead of the one above (is done by ISO 12233 standards))
            #LSF = LSF*np.hamming(len(LSF))**int(ERF.max()-ERF.min())


        # calculate the Modulation Transfer Function
        MTF = np.abs(np.fft.fftshift(np.fft.fft(LSF)))
        # normalise the MTF
        if normalize_MTF:
            MTF = MTF/MTF.max()

        # store the results
        ERF_array[i] = ERF
        LSF_array[i] = LSF
        MTF_array[i] = MTF

    # Average the results
    ERF = np.mean(ERF_array, axis=0)
    LSF = np.mean(LSF_array, axis=0)
    MTF = np.mean(MTF_array, axis=0)

    if process_LSF:
        # shift LSF x axis to align with highest average gradient (might not be same as from first slice)
        LSF_x_axis += (LSF_x_axis_original[np.argmax(np.abs(np.gradient(ERF)))]-LSF_x_axis[np.argmax(LSF)])

    # calculate the frequency axis
    MTF_freq = np.fft.fftshift(np.fft.fftfreq(len(MTF), d=pixel_size))

    # interpolate the MTF to find MTF50 and MTF10 (makes it more accurate)
    MTF_freq_interpolated = np.linspace(0, np.max(MTF_freq), 1000)
    MTF_interpolated = np.interp(MTF_freq_interpolated, MTF_freq, MTF)

    # find MTF50
    try:
        MTF50_index = np.where((MTF_interpolated < 0.5) & (MTF_freq_interpolated > 0))[0][0]
        MTF50 = MTF_freq_interpolated[MTF50_index]
    except:
        MTF50 = 1/(2*pixel_size) # if MTF50 is not found, set it to Nyquist frequency
        MTF50_index = len(MTF_interpolated)-1

    # find MTF10
    try:
        MTF10_index = np.where((MTF_interpolated < 0.1) & (MTF_freq_interpolated > 0))[0][0]
        MTF10 = MTF_freq_interpolated[MTF10_index]
    except:
        MTF10 = 1/(2*pixel_size) # if MTF10 is not found, set it to Nyquist frequency
        MTF10_index = len(MTF_interpolated)-1

    if plot_results:
        # Plot the results
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plotting the combined images
        ax1 = axs[0, 0]
        im = ax1.imshow(image_data[0], cmap='gray')
        ax1.axis('off')
        ax1.set_title('Cropped phantom scan (first slice)')

        # Plotting the Edge Response Function
        ax2 = axs[0, 1]
        ax2.plot(np.linspace(0, len(ERF)*pixel_size, len(ERF)), ERF, color='orange', alpha=0.5, label='ERF')
        if find_absolute_MTF==False:
            ax2.set_xlabel('Distance perpendicular to edge (pixels)')
        else:
            ax2.set_xlabel('Distance perpendicular to edge (mm)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Edge Response Function')
        ax2.grid(True)
        ax2.legend()

        # Plotting the Line Spread Function
        ax3 = axs[1, 0]
        ax3.plot(LSF_x_axis, LSF, color='green') # x-axis is the same as ERF
        if find_absolute_MTF==False:
            ax3.set_xlabel('Distance perpendicular to edge (pixels)')
        else:
            ax3.set_xlabel('Distance perpendicular to edge (mm)')
        ax3.set_ylabel('Intensity (gradient of the ERF)')
        ax3.set_title('Line Spread Function')
        ax3.set_xlim([0, len(ERF)*pixel_size])
        ax3.grid(True)

        # Plotting the Modulation Transfer Function
        ax4 = axs[1, 1]
        ax4.plot(MTF_freq, MTF, color='red')
        ax4.axvline(MTF50, 0, 0.5, color='black', linestyle='-.', label=f'MTF50 = {MTF50:.2f}')
        ax4.axvline(MTF10, 0, 0.1, color='black', linestyle='--', label=f'MTF10 = {MTF10:.2f}')
        ax4.axhline(0.5, 0, MTF50/(1/(2*pixel_size)), color='black', linestyle='-.')
        ax4.axhline(0.1, 0, MTF10/(1/(2*pixel_size)), color='black', linestyle='--')
        if find_absolute_MTF==False:
            ax4.set_xlabel('Spatial Frequency (Cycles per pixel)')
        else:
            ax4.set_xlabel('Spatial Frequency (lp per mm)')
        ax4.set_xlim([0, (1/(2*pixel_size))])
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Normalised Modulation (i.e. Contrast)')
        ax4.set_title('Modulation Transfer Function')
        ax4.legend()
        ax4.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot
        if find_absolute_MTF == False:
            plt.savefig(target_directory + '/MTF_results_relative.png', dpi=300)
            plt.savefig(target_directory + '/MTF_results_relative.pdf', dpi=300)
        else:
            plt.savefig(target_directory + '/MTF_results_absolute.png', dpi=300)
            plt.savefig(target_directory + '/MTF_results_absolute.pdf', dpi=300)

    if return_ERF:
        return MTF_freq, MTF, ERF
    else:
        return MTF_freq, MTF

if __name__ == '__main__':
    # Load the image data
    from reconstruction.ct_core import vff_io as vff
    image_data = vff.read_vff("data/results/repaint_reconstruction.vff", verbose=False)[1]

    # select only the slices which contain the slanted edge test pattern
    # then define the crop indices (y1, y2, x1, x2)
    #image_data_MTF = image_data[24:67, :, :]

    #crop_indices_MTF = np.array([788, 1988, 1156, 1572])
    #crop_indices_MTF = np.round(crop_indices_MTF/3).astype(int)
    image_data_MTF = image_data[228:229, :, :]
    crop_indices_MTF = [270, 664, 522, 640]

    _ = get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.085,
                target_directory=os.getcwd(), plot_results=True, edge_angle=5.4, high_to_low=True)



'''
For data/results/ground_truth_reconstruction.vff
image_data_MTF = image_data[228:229, :, :]
crop_indices_MTF = [270, 664, 522, 640]
'''
