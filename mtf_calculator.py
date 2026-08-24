# Description: This script calculates the Modulation Transfer Function (MTF) of an image data set
# using a slanted edge test pattern.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import matplotlib.pyplot as plt

def get_MTF(image_data, crop_indices, find_absolute_MTF=True, pixel_size=0.05,
            target_directory=os.getcwd(), plot_results=True, edge_angle=5.0, high_to_low=True, process_LSF=True, return_ERF=False, normalize_MTF=True, **kwargs):
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
    :param edge_angle: The angle of the edge in degrees (default is 5 degrees).
                       May be negative: the sign says which way the edge leans
                       and is handled. What matters is the MAGNITUDE -- below
                       about 1.5 degrees the edge does not cross a whole pixel
                       over the ROI height and the projected-bin method has
                       nothing to supersample.
    :param high_to_low: Boolean to check if the edge goes from high to low (True) or low to high (False)
    :param process_LSF: Boolean to process the LSF or not (i.e Detrending, windowing and centering)
    :param normalize_MTF: Boolean to normalize the MTF by its maximum value (default is True)
    :return: MTF_freq: The frequency axis of the MTF, in mm^-1. TWO-SIDED and
                       fftshifted, i.e. it runs from -Nyquist to +Nyquist with
                       DC in the middle -- MTF_freq[0] is NOT zero frequency.
                       Take MTF_freq >= 0 before normalising or before reading
                       a crossing off the curve.
                       Nyquist here is the PIXEL Nyquist (1 / (2 * pixel_size)),
                       because the 4x oversampled ERF is block-averaged back to
                       pixel spacing before the transform. get_TTF does not do
                       that, so its axis reaches further; the two are directly
                       comparable in value but not in extent.
    :return: MTF: The Modulation Transfer Function
    """
    # Check if the LSF processing is required
    if process_LSF:
        from .helper_scripts import lsf_processing as LSF_processing

    # Work on a COPY. This function reshapes and pads crop_indices, and it used
    # to do so in place on the caller's list — so all_metrics_calculator, which
    # passes the same list to get_MTF and then to get_NEQ, had the second call
    # pad an already-padded crop.
    crop_indices = [int(v) for v in crop_indices]

    # ROI width is taken as half the ROI height (this is ambiguous and changes the MTF shape)
    if crop_indices[1]-crop_indices[0] != 2*(crop_indices[3]-crop_indices[2]):
        difference = (crop_indices[3]-crop_indices[2])-int(0.5*(crop_indices[1]-crop_indices[0]))
        crop_indices[3] -= int(difference/2)
        crop_indices[2] += int(difference/2)


    # For relative MTF calculation it is done per pixel (so =1)
    if find_absolute_MTF==False:
        pixel_size = 1

    # if only one image 2d data is provided, convert it to 3d
    if len(image_data.shape) == 2:
        image_data = image_data[np.newaxis, :, :]

    # Widen the crop sideways to make room for the edge's travel across the ROI.
    #
    # abs(): the padding is a WIDTH, so it has to grow the crop whichever way
    # the edge leans. With the signed tangent a NEGATIVE edge_angle made both
    # lines shrink the crop by the travel instead of growing it, removing the
    # very columns the row alignment below then needed.
    travel_px = int(np.ceil(1.05 * abs(crop_indices[1]-crop_indices[0])
                            * abs(np.tan(np.radians(edge_angle)))))
    crop_indices[0] = max(0, crop_indices[0])
    crop_indices[1] = min(image_data.shape[1], crop_indices[1])
    crop_indices[2] = max(0, crop_indices[2] - travel_px)
    crop_indices[3] = min(image_data.shape[2], crop_indices[3] + travel_px)
    if crop_indices[1]-crop_indices[0] < 4 or crop_indices[3]-crop_indices[2] < 8:
        raise ValueError(
            f"the MTF crop {crop_indices} does not fit inside a "
            f"{image_data.shape[1]}x{image_data.shape[2]} image with room for "
            f"an edge slanted at {edge_angle} degrees")

    # now the image_data is cropped
    image_data = image_data[:, crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]].astype(np.float64)


    # --- ERF construction via interpolation-based row alignment ---
    # Each row of the crop samples the slanted edge at a different sub-pixel
    # position.  We shift each row by the exact fractional offset (using
    # linear interpolation) onto a common 4x-oversampled grid, then average.
    # This avoids the smearing caused by int()-rounding in the old np.roll
    # approach.
    from scipy.interpolate import interp1d

    nrows = image_data.shape[1]
    ncols = image_data.shape[2]
    shift_per_row = np.tan(np.radians(edge_angle))  # pixels per row, SIGNED
    # The step actually applied to the rows below. `high_to_low` flips the
    # direction, so this — not shift_per_row — is what the geometry depends on.
    step_per_row = -shift_per_row if high_to_low else shift_per_row
    travel = abs(step_per_row) * (nrows - 1)   # pixels the edge moves across the ROI

    # 4x oversampled output grid: covers the region common to all shifted rows
    subsample = 4
    erf_pixel_size = pixel_size / subsample
    # Usable range after accounting for the shift across all rows
    usable_cols = ncols - int(np.ceil(travel)) - 1
    if usable_cols < 8:
        raise ValueError(
            f"an edge slanted at {edge_angle} degrees travels {travel:.1f} "
            f"pixels across {nrows} rows, leaving {usable_cols} of {ncols} "
            f"columns common to every row. Use a shorter ROI or a wider crop.")
    n_erf = usable_cols * subsample
    # WHERE the common region starts. Row r is shifted by -step*r, so with a
    # negative step every row moves right and the region common to all of them
    # begins at +travel rather than at 0. Starting at 0 regardless — which is
    # what an unsigned grid does — put most rows outside the interpolation
    # range for a negative angle, where they were discarded as NaN.
    x_start = (travel * pixel_size) if step_per_row < 0 else 0.0
    x_erf_grid = x_start + np.arange(n_erf) * erf_pixel_size  # common grid (mm)

    # Pre-compute for array sizing: final ERF length after block-averaging
    array_1_length = n_erf // subsample
    arrays_shape_1 = (image_data.shape[0], array_1_length)
    ERF_array = np.zeros(arrays_shape_1)
    LSF_array = np.zeros(arrays_shape_1)
    LSF_x_axis = np.linspace(0, array_1_length * pixel_size, array_1_length)
    MTF_array = np.zeros(arrays_shape_1)

    # Collect per-slice results (lengths may vary due to LSF processing)
    _erf_list = []
    _lsf_list = []
    _lsf_x_list = []

    # iterate through the slices and compute the ERF and LSF
    for i in range(image_data.shape[0]):

        sl = image_data[i]
        x_orig = np.arange(ncols) * pixel_size  # original pixel centres (mm)

        # Accumulate aligned rows onto the oversampled grid
        ERF_sum = np.zeros(n_erf)
        ERF_count = np.zeros(n_erf)
        for row in range(sl.shape[0]):
            # Fractional shift for this row (mm), signed
            dx = step_per_row * row * pixel_size
            # Shifted x-coordinates for this row
            x_shifted = x_orig - dx
            # Interpolate onto the common grid
            f = interp1d(x_shifted, sl[row, :], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            vals = f(x_erf_grid)
            valid = ~np.isnan(vals)
            ERF_sum[valid] += vals[valid]
            ERF_count[valid] += 1

        # Average
        ERF_count[ERF_count == 0] = 1
        ERF_4x = ERF_sum / ERF_count

        # Block-average from 4x back to 1x pixel spacing
        n_out = len(ERF_4x) // subsample * subsample
        ERF = ERF_4x[:n_out].reshape(-1, subsample).mean(axis=1)

        # Fit a smoothing spline to the ERF and differentiate analytically.
        # This produces a noise-free LSF while preserving the true edge shape
        # (including filter-induced ringing / sidelobes).
        from scipy.interpolate import UnivariateSpline
        x_erf_1x = np.arange(len(ERF)) * pixel_size
        # Smoothing factor: n * noise_variance (estimate noise from flat plateaus)
        q = max(len(ERF) // 4, 2)
        noise_std = np.mean([np.std(ERF[:q]), np.std(ERF[-q:])])
        s_val = len(ERF) * noise_std**2
        try:
            spline = UnivariateSpline(x_erf_1x, ERF, s=s_val, k=4)
            LSF = spline.derivative()(x_erf_1x)
        except:
            # Fallback to numerical gradient if spline fails
            LSF = np.gradient(ERF, pixel_size, edge_order=1)

        # process the LSF
        if process_LSF:
            LSF_x_tmp = np.linspace(0, len(LSF)*pixel_size, len(LSF))
            LSF_x_tmp, LSF = LSF_processing.process_LSF(LSF_x_tmp, LSF, pixel_size)

        # Store per-slice results in lists (lengths may vary across slices)
        _erf_list.append(ERF)
        _lsf_list.append(LSF)
        _lsf_x_list.append(LSF_x_tmp if process_LSF else np.linspace(0, len(LSF)*pixel_size, len(LSF)))

    # Determine the target length from the median processed LSF length.
    # This prevents a single degenerate slice from collapsing the arrays.
    lsf_lengths = [len(l) for l in _lsf_list]
    target_len = int(np.median(lsf_lengths))
    # Minimum sanity: at least 20 points
    target_len = max(target_len, 20)

    LSF_array = np.zeros((image_data.shape[0], target_len))
    MTF_array = np.zeros((image_data.shape[0], target_len))
    ERF_array = np.zeros((image_data.shape[0], array_1_length))

    for i in range(image_data.shape[0]):
        ERF_array[i] = _erf_list[i]

        lsf_i = _lsf_list[i]
        # Pad or crop to target_len
        if len(lsf_i) < target_len:
            pad_l = (target_len - len(lsf_i)) // 2
            pad_r = target_len - len(lsf_i) - pad_l
            lsf_i = np.pad(lsf_i, (pad_l, pad_r))
        elif len(lsf_i) > target_len:
            crop_l = (len(lsf_i) - target_len) // 2
            lsf_i = lsf_i[crop_l:crop_l + target_len]

        LSF_array[i] = lsf_i

        MTF_i = np.abs(np.fft.fftshift(np.fft.fft(lsf_i)))
        if normalize_MTF and MTF_i.max() > 0:
            MTF_i = MTF_i / MTF_i.max()
        MTF_array[i] = MTF_i

    # Average the results
    ERF = np.mean(ERF_array, axis=0)
    LSF = np.mean(LSF_array, axis=0)
    MTF = np.mean(MTF_array, axis=0)

    if process_LSF:
        LSF_x_axis = np.linspace(0, target_len * pixel_size, target_len)

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
        plt.close('all')

    if return_ERF:
        return MTF_freq, MTF, ERF
    else:
        return MTF_freq, MTF

def parse_args():
    import argparse
    from .helper_scripts.io_utils import parse_int_list, parse_slices
    parser = argparse.ArgumentParser(description='Calculate the Modulation Transfer Function (MTF) from CT image data.')
    parser.add_argument('--input', required=True, help='Path to image file (.npy, .npz, or .vff)')
    parser.add_argument('--crop_indices', required=True, type=parse_int_list,
                        help='Crop region as y1,y2,x1,x2 (e.g. "270,664,522,640")')
    parser.add_argument('--slices', type=str, default=None,
                        help='Slice selection (e.g. "10:160" or "0:30,140:182")')
    parser.add_argument('--pixel_size', type=float, default=0.05, help='Pixel size in mm (default: 0.05)')
    parser.add_argument('--edge_angle', type=float, default=5.0, help='Edge angle in degrees (default: 5.0)')
    parser.add_argument('--high_to_low', action='store_true', default=True,
                        help='Edge goes from high to low intensity (default: True)')
    parser.add_argument('--low_to_high', action='store_true', help='Edge goes from low to high intensity')
    parser.add_argument('--relative', action='store_true', help='Calculate relative MTF instead of absolute')
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

    output_dir = ensure_output_dir(args.output_dir)
    high_to_low = not args.low_to_high

    _ = get_MTF(image_data, args.crop_indices, find_absolute_MTF=not args.relative,
                pixel_size=args.pixel_size, target_directory=output_dir,
                plot_results=not args.no_plot, edge_angle=args.edge_angle, high_to_low=high_to_low)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
