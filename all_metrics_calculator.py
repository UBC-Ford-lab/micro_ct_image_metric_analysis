# Description: This script calculates all the metrics of an image data set: NPS, NEQ, MTF, TTF, and d'
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
from . import nps_calculator as NPS_calculator
from . import neq_calculator as NEQ_calculator
from . import mtf_calculator as MTF_calculator
from . import ttf_calculator as TTF_calculator
from . import d_prime_calculator
from reconstruction.ct_core import vff_io as vff

if __name__ == '__main__':
    # Create the target directory
    # TODO: Update this path to your desired output directory
    target_directory = 'data/results/metric_results'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Calculate the NPS
    # TODO: Update this path to your reconstruction volume
    print("Calculating the NPS")
    image_data = vff.read_vff("data/results/your_reconstruction.vff", verbose=False)[1]
    image_data_NPS = image_data[np.concatenate((np.arange(0, 30), np.arange(140, 182))), :, :]
    ROI_bounds_NPS = np.array([[308, 714, 1116, 1522],[552, 958, 1632, 2038],[1024, 1430, 1896, 2302],[1568, 1974, 1688, 2094],[1892, 2298, 1148, 1554], [1576, 1982, 512, 918], [1040, 1446, 304, 710], [544, 950, 544, 950]])

    _ = NPS_calculator.get_NPS(image_data_NPS, ROI_bounds_NPS, pixel_size=0.025, target_directory=target_directory, plot_results=True)

    # Calculate the MTF
    print("Calculating the MTF")
    image_data = vff.read_vff("data/scans/phantom/Halfscan-100ms/Halfscan-100ms-25um-slanted-edge.vff", verbose=False)[1]
    image_data_MTF = image_data[10:160, :, :]
    crop_indices_MTF = [732, 1940, 1160, 1548] # ymin, ymax, xmin, xmax
    _ = MTF_calculator.get_MTF(image_data_MTF, crop_indices_MTF, find_absolute_MTF=True, pixel_size=0.025,
                               target_directory=target_directory, plot_results=True, edge_angle=5.5, high_to_low=True)

    # Calculate the NEQ
    print("Calculating the NEQ")
    _ = NEQ_calculator.get_NEQ(image_data_MTF, image_data_NPS, crop_indices_MTF, ROI_bounds_NPS, pixel_size=0.025,
                               target_directory=target_directory, plot_results=True)

    # Calculate the TTF
    print("Calculating the TTF")
    image_data = vff.read_vff("data/scans/phantom/Halfscan-100ms/Halfscan-100ms-25um-materials.vff", verbose=False)[1]
    image_data_TTF = image_data[30:130, :, :]
    centre_pixels_TTF = [[1335, 2141], [765, 1914], [528, 1348], [755, 782], [1321, 546], [1890, 773], [2128, 1335], [1330, 1341]]
    radius_TTF = 120
    materials_TTF = ['Teflon', 'HD POLY', 'Fat', 'Tissue', 'Lucite', 'Water', 'SB3', 'Air']
    _ = TTF_calculator.get_TTF(image_data_TTF, centre_pixels_TTF, radius_TTF, materials=materials_TTF, find_absolute_TTF=True,
                               pixel_size=0.025, target_directory=target_directory, plot_results=True)

    # Calculate the detectability index d'
    print("Calculating the detectability index d'")
    task_function_object_size = 0.2
    task_function_data = d_prime_calculator.create_circular_task_function(-160, task_function_object_size, pixel_size=0.025,
                                                                          image_dimension=np.min(image_data.shape[1:]))

    _ = d_prime_calculator.get_d_prime(image_data_TTF, centre_pixels_TTF, radius_TTF, materials_TTF, image_data_NPS, ROI_bounds_NPS,
            task_function_data=task_function_data, task_function_material='Fat', task_function_object_size=task_function_object_size,
            pixel_size=0.025, verbose=True, plot_results=True, target_directory=target_directory)
