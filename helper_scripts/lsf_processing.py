# Description: This script processes the line spread function (LSF) based on Jeff Siewerdsen's method in
# MTFTools from https://istar.jhu.edu/downloads/
# specifically this is a close adaptation of the matlab script MTFTools/src/+mtf/lsfDetrendWdCenter.m
# Written by Falk Wiegmann at the University of British Columbia in June 2024.

import numpy as np
from scipy.optimize import curve_fit


def process_LSF(LSF_x_axis, LSF, pixel_size):
    """Detrend, window and crop a line spread function.

    `pixel_size` is used only as a PRIOR on the physical blur width (it sets
    the starting point and the bounds of the Gaussian fit). The window widths
    are measured in array samples, and that spacing is taken from
    `LSF_x_axis`, which is the only thing that knows it -- callers supersample
    by different factors.
    """
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

    # Fit Gaussian to abs(LSF) to find peak centre and width.
    # Using abs() makes this robust to signed LSFs with negative sidelobes.
    LSF_abs = np.abs(LSF)

    try:
        params, _ = curve_fit(primary_fit_func, LSF_x_axis, LSF_abs,
                            p0=[LSF_x_axis[np.argmax(LSF_abs)], 2*pixel_size, np.max(LSF_abs)],
                            maxfev=5000,
                            bounds=([0, pixel_size, 0.5*np.max(LSF_abs)],
                                    [np.max(LSF_x_axis), 20*pixel_size, np.max(LSF_abs)]))

    except:
        return LSF_x_axis, LSF

    fh_lsf = lambda x: primary_fit_func(x, *params)
    cen = max(np.argmax(fh_lsf(LSF_x_axis)), 1)
    # FWHM in ARRAY SAMPLES, which is what calcRg_scaleFwhm indexes with.
    #
    # This used to divide by pixel_size, i.e. it expressed the width in PIXELS
    # and then used that as a number of samples. The two are the same only when
    # the LSF is sampled once per pixel, which is true for get_MTF (it block-
    # averages its 4x ERF back down first) and false for get_TTF (its radial
    # ERF stays at 1/4-pixel spacing). On the TTF path the window therefore
    # came out 4x too narrow and clipped the LSF tails, which NARROWS the LSF
    # and so RAISES the transfer function it produces.
    #
    # MEASURED 2026-08-24 on a synthetic disc with a 0.30 mm Gaussian blur
    # (true TTF50 1.471 lp/mm): process_LSF=True gave 1.743 (+18%) and
    # process_LSF=False gave 1.437 (-2%). Taking the spacing from the axis
    # itself makes the two agree and is correct for either caller by
    # construction.
    if np.size(LSF_x_axis) > 1:
        sample_spacing = float(np.median(np.diff(np.asarray(LSF_x_axis))))
    else:
        sample_spacing = float(pixel_size)
    if not np.isfinite(sample_spacing) or sample_spacing <= 0:
        sample_spacing = float(pixel_size)
    fwhm = 2.355 * params[1] / sample_spacing # FWHM since we know sigma from the fit
    # Ensure 6×FWHM window is at least 20 pixels to avoid over-cropping
    fwhm = max(fwhm, 20 / 6)
    # determine region of interest based on primarySca
    rg = calcRg_scaleFwhm(P['primarySca'], fwhm, cen, len(LSF))

    # Detrend data by subtracting 1st order polynomial fit. The signal (based on primarySca) is weighted out of the fitting
    lsfTrend = LSF - fh_lsf(LSF_x_axis)
    weightTrend = np.abs(LSF_x_axis - LSF_x_axis[cen])
    weightTrend[rg] = 0
    if np.sum(weightTrend > 0) >= 3:
        try:
            params_trend = np.polyfit(LSF_x_axis, lsfTrend, 1, w=weightTrend)
            fh_Trend = lambda x: np.polyval(params_trend, x)
            # Actual detrending step from above computed trend
            LSF -= fh_Trend(LSF_x_axis)
        except np.linalg.LinAlgError:
            pass

    # Zero out the marginal region (anything outside of primarySca)
    LSF[:rg[0]] = 0
    LSF[rg[-1]:] = 0

    # Throw away points that are outside the margin (marginSca)
    rg = calcRg_scaleFwhm(P['marginSca'], fwhm, cen, len(LSF))
    LSF_x_axis = LSF_x_axis[rg]
    LSF = LSF[rg]

    return LSF_x_axis, LSF
