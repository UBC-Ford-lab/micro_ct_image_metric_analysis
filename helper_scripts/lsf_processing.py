# Description: This script processes the line spread function (LSF) based on Jeff Siewerdsen's method in
# MTFTools from https://istar.jhu.edu/downloads/
# specifically this is a close adaptation of the matlab script MTFTools/src/+mtf/lsfDetrendWdCenter.m
# Written by Falk Wiegmann at the University of British Columbia in June 2024.

import numpy as np
from scipy.optimize import curve_fit

def process_LSF(LSF_x_axis, LSF, pixel_size):
    # The Gaussian shape fitting function
    def primary_fit_func(x, mu, sigma, A):
        return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

    # Calculate the region of interest given a scale factor and FWHM
    def calcRg_scaleFwhm(sca, fwhm, cen, length):
        lMargin = min(cen, int(sca * fwhm / 2))
        rMargin = min(length - cen, int(sca * fwhm / 2))
        margin_ = min(lMargin, rMargin)
        rg = np.arange(cen - margin_, cen + margin_)
        return rg

    # scaling parameters
    P = {
    'primarySca': 6, # scale factor for number of FWHM to include in processed signal (rest is zeroed)
    'marginSca': 12 # scale factor for number of FWHM to keep (as zeroes). Rest is discarded
    }

    # shift to axis
    LSF -= np.min(LSF)

    # Fit Gaussian to determine FWHM
    try:
        params, _ = curve_fit(primary_fit_func, LSF_x_axis, LSF, p0=[LSF_x_axis[np.argmax(LSF)], pixel_size, np.max(LSF)],
                            maxfev=5000, bounds=([0, 0.5*pixel_size, 0.5*np.max(LSF)], [np.max(LSF_x_axis), 5*pixel_size, np.max(LSF)]))

    except:
        return LSF_x_axis, LSF

    fh_lsf = lambda x: primary_fit_func(x, *params)
    cen = max(np.argmax(fh_lsf(LSF_x_axis)), 1)
    fwhm = 2.355 * params[1] / pixel_size # FWHM since we know sigma from the fit
    # determine region of interest based on primarySca
    rg = calcRg_scaleFwhm(P['primarySca'], fwhm, cen, len(LSF))

    # Detrend data by subtracting 1st order polynomial fit. The signal (based on primarySca) is weighted out of the fitting
    lsfTrend = LSF - fh_lsf(LSF_x_axis)
    weightTrend = np.abs(LSF_x_axis - np.median(LSF_x_axis))
    weightTrend[rg] = 0
    params_trend = np.polyfit(LSF_x_axis, lsfTrend, 1, w=weightTrend)
    fh_Trend = lambda x: np.polyval(params_trend, x)

    # Actual detrending step from above computed trend
    LSF -= fh_Trend(LSF_x_axis)

    # Zero out the marginal region (anything outside of primarySca)
    LSF[:rg[0]] = 0
    LSF[rg[-1]:] = 0

    # Throw away points that are outside the margin (marginSca)
    rg = calcRg_scaleFwhm(P['marginSca'], fwhm, cen, len(LSF))
    LSF_x_axis = LSF_x_axis[rg]
    LSF = LSF[rg]

    return LSF_x_axis, LSF
