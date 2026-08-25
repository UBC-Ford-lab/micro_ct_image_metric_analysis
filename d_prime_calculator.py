# Description: This script calculates the Detectability index (d') of an image data set
# using the TTF, NPS, and W (task function)
# Written by Falk Wiegmann at the University of British Columbia in May 2024.
# Updated March 2026: added frequency-domain NPW observer (get_d_prime_npw) for
# pre-computed MTF/NPS curves following AAPM TG-233 (Samei et al., 2019).

import numpy as np
import os
import matplotlib.pyplot as plt
from photutils.profiles import RadialProfile
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import j1
import scipy.integrate
from . import ttf_calculator as TTF_calculator
from . import nps_calculator as NPS_calculator


# =============================================================================
# Frequency-domain NPW observer (AAPM TG-233)
# =============================================================================

# Default disc diameters for micro-CT task-based assessment (mm)
DEFAULT_DISC_DIAMETERS = [0.15, 0.5, 1.0, 3.0]

# Default contrast: soft-tissue lesion (100 HU) — clinically challenging
# scenario for micro-CT mouse imaging.  Other common choices:
#   ~300-700 HU  lung nodule vs parenchyma
#   ~1000 HU     air-water (phantom calibration)
DEFAULT_CONTRAST_HU = 100.0


def disc_task_function(f, radius_mm, contrast_hu):
    """Radial Fourier transform of a uniform disc (jinc function).

    For a disc of radius R and contrast ΔC, the 2D FT evaluated along
    the radial frequency axis is:

        W(f) = ΔC · π R² · 2 J₁(2πRf) / (2πRf)       f > 0
        W(0) = ΔC · π R²                               f = 0

    Args:
        f: 1D array of radial spatial frequencies (lp/mm or mm⁻¹).
        radius_mm: Disc radius in mm.
        contrast_hu: Contrast between disc and background in HU.

    Returns:
        W: 1D array, same shape as *f*, task function amplitude.
    """
    f = np.asarray(f, dtype=np.float64)
    W = np.empty_like(f)
    area = np.pi * radius_mm ** 2

    nonzero = f > 0
    arg = 2.0 * np.pi * radius_mm * f[nonzero]
    W[nonzero] = contrast_hu * area * 2.0 * j1(arg) / arg
    W[~nonzero] = contrast_hu * area
    return W


# Rose criterion.  The classical threshold for a signal-known-exactly
# detection task: d' >= 3 is conventionally read as "reliably detectable".
ROSE_THRESHOLD = 3.0


def curve_diameter_grid(disc_diameters_mm, n=1201):
    """Dense, geometrically spaced diameter grid bracketing *disc_diameters_mm*.

    d' is reported at a handful of nominal disc sizes, but a threshold
    crossing has to be read off a continuous curve, so it is evaluated on a
    grid running from a fifth of the smallest requested disc to five times
    the largest.  Widening it that far is what keeps the crossing inside the
    sampled range across a wide spread of noise levels.

    The default n puts the samples well under a percent apart, so the curve
    is smooth at any plotted scale and the crossing is set by the physics
    rather than by the grid; d' at one diameter costs two 1D integrals, so
    a dense grid is cheap.
    """
    d = np.asarray(disc_diameters_mm, dtype=np.float64)
    return np.geomspace(0.2 * d.min(), 5.0 * d.max(), int(n))


def diameter_at_threshold(diameters, d_prime, threshold=ROSE_THRESHOLD):
    """Disc diameter at which d' reaches *threshold* -- the detectable size.

    d' grows with disc size, so the curve crosses the threshold once, from
    below; the crossing is located by linear interpolation between the two
    bracketing samples (linear in both diameter and d', matching the axes
    the curve is plotted on).  If the curve is not monotonic the *last*
    upward crossing is taken, i.e. the smallest diameter above which d'
    never drops back below the threshold.

    Returns nan when the crossing lies outside the sampled diameters --
    either every sampled disc is already above the threshold, or none of
    them reach it.  That is a real answer ("this grid cannot locate it"),
    not an error; a caller that needs a number must widen the grid.

    Args:
        diameters: 1D array of disc diameters (mm), in any order.
        d_prime: 1D array of d' values, one per diameter.
        threshold: Detectability threshold (default ROSE_THRESHOLD = 3).

    Returns:
        Diameter in mm at the crossing, or nan.
    """
    d = np.asarray(diameters, dtype=np.float64)
    y = np.asarray(d_prime, dtype=np.float64)
    order = np.argsort(d)
    d, y = d[order], y[order]
    keep = np.isfinite(d) & np.isfinite(y)
    d, y = d[keep], y[keep]
    if d.size < 2:
        return float('nan')

    below = np.nonzero(y < threshold)[0]
    if below.size == 0 or below[-1] == d.size - 1:
        return float('nan')

    lo = int(below[-1])
    hi = lo + 1
    span = y[hi] - y[lo]
    if span <= 0:
        return float('nan')
    t = (threshold - y[lo]) / span
    return float(d[lo] + t * (d[hi] - d[lo]))


def _resample_mtf_nps(mtf_freq, mtf, nps_freq, nps, normalize_mtf, n_freq):
    """Put the MTF and the NPS on one frequency grid up to the shared Nyquist."""
    mtf_freq = np.asarray(mtf_freq, dtype=np.float64)
    nps_freq = np.asarray(nps_freq, dtype=np.float64)

    pos_mtf = mtf_freq >= 0
    f_max = min(mtf_freq[pos_mtf].max(), nps_freq.max())
    freq = np.linspace(0, f_max, n_freq)

    mtf_interp = interp1d(mtf_freq[pos_mtf], np.asarray(mtf)[pos_mtf],
                          bounds_error=False, fill_value=0.0)(freq)
    nps_interp = interp1d(nps_freq, nps,
                          bounds_error=False, fill_value='extrapolate')(freq)

    if normalize_mtf:
        mtf_max = mtf_interp.max()
        if mtf_max > 0:
            mtf_interp = mtf_interp / mtf_max

    # Protect against zero/negative NPS
    nps_interp = np.maximum(nps_interp, np.max(nps_interp) * 1e-12)
    return freq, mtf_interp, nps_interp


def _d_prime_for_diameters(freq, mtf, nps, diameters, contrast_hu, eye=None):
    """NPW (eye=None) or NPWE d' for each diameter, plus the task functions.

    See get_d_prime_npw / get_d_prime_npwe for the two integrals; the eye
    filter enters the numerator as E^2 and the denominator as E^4.
    """
    diameters = np.asarray(diameters, dtype=np.float64)
    e_num = 1.0 if eye is None else np.asarray(eye) ** 2
    e_den = 1.0 if eye is None else np.asarray(eye) ** 4

    values = np.empty(len(diameters))
    task_functions = {}
    for i, diam in enumerate(diameters):
        W = disc_task_function(freq, diam / 2.0, contrast_hu)
        task_functions[float(diam)] = W

        common = np.abs(W) ** 2 * mtf ** 2 * freq
        numerator = scipy.integrate.simpson(common * e_num, x=freq)
        denominator = scipy.integrate.simpson(common * e_den * nps, x=freq)

        if denominator > 0 and numerator > 0:
            values[i] = numerator / np.sqrt(denominator)
        else:
            values[i] = 0.0
    return values, task_functions


def get_d_prime_npw(mtf_freq, mtf, nps_freq, nps,
                    disc_diameters_mm=None, contrast_hu=None,
                    normalize_mtf=True, n_freq=500,
                    rose_threshold=ROSE_THRESHOLD, curve_diameters_mm=None,
                    plot_results=True, target_directory=None):
    """Compute NPW-observer detectability index for multiple disc sizes.

    Uses pre-computed 1D MTF and NPS curves and the AAPM TG-233
    non-prewhitening (NPW) matched-filter formulation:

        d'^2 = [integral |W(f)|^2 MTF^2(f) f df]^2
              / [integral |W(f)|^2 MTF^2(f) NPS(f) f df]

    where W(f) is the analytical Bessel-function task function for a
    uniform disc and the integrals run from 0 to the Nyquist frequency.

    Besides d' at each requested diameter, d' is evaluated on a dense
    diameter grid so the *detectable disc size* -- the diameter at which
    d' crosses `rose_threshold` -- can be read off the curve.  That single
    number in mm is the headline result: it says how small an object of
    this contrast the system resolves out of its own noise, and unlike d'
    at a fixed diameter it stays readable when two methods differ by an
    order of magnitude.

    Args:
        mtf_freq: 1D array of MTF frequencies (mm^-1).
        mtf: 1D array of MTF values (absolute or normalized).
        nps_freq: 1D array of NPS frequencies (mm^-1).
        nps: 1D array of NPS values (HU^2 mm^2).
        disc_diameters_mm: List of disc diameters in mm.
            Defaults to DEFAULT_DISC_DIAMETERS [0.15, 0.5, 1.0, 3.0].
        contrast_hu: Contrast between disc and background in HU.
            Defaults to DEFAULT_CONTRAST_HU (100 HU, soft tissue).
        normalize_mtf: If True, normalize MTF to unity at f=0 before
            computing d'.  Recommended when absolute MTF scale varies
            across methods.
        n_freq: Number of frequency samples for integration grid.
        rose_threshold: Detectability threshold for the reported disc size
            (default ROSE_THRESHOLD = 3, the Rose criterion).
        curve_diameters_mm: Diameter grid for the continuous d' curve and
            the threshold crossing.  Defaults to curve_diameter_grid() over
            the requested diameters.
        plot_results: Whether to save a diagnostic plot.
        target_directory: Output directory for the plot.
            If None, uses current working directory.

    Returns:
        dict with keys:
            'disc_diameters_mm': array of requested disc diameters
            'contrast_hu': contrast used
            'd_prime': array of d' values, one per requested diameter
            'curve_diameters_mm': dense diameter grid
            'curve_d_prime': d' on that grid
            'rose_threshold': the threshold used
            'diameter_at_threshold_mm': detectable disc size (mm), or nan
                if the crossing falls outside the grid
            'freq': integration frequency grid
            'task_functions': dict mapping diameter -> W(f) array
    """
    if disc_diameters_mm is None:
        disc_diameters_mm = list(DEFAULT_DISC_DIAMETERS)
    if contrast_hu is None:
        contrast_hu = DEFAULT_CONTRAST_HU
    if target_directory is None:
        target_directory = os.getcwd()

    disc_diameters_mm = np.asarray(disc_diameters_mm, dtype=np.float64)
    if curve_diameters_mm is None:
        curve_diameters_mm = curve_diameter_grid(disc_diameters_mm)
    curve_diameters_mm = np.asarray(curve_diameters_mm, dtype=np.float64)

    freq, mtf_interp, nps_interp = _resample_mtf_nps(
        mtf_freq, mtf, nps_freq, nps, normalize_mtf, n_freq)

    d_prime_values, task_functions = _d_prime_for_diameters(
        freq, mtf_interp, nps_interp, disc_diameters_mm, contrast_hu)
    curve_values, _ = _d_prime_for_diameters(
        freq, mtf_interp, nps_interp, curve_diameters_mm, contrast_hu)
    d_at_threshold = diameter_at_threshold(curve_diameters_mm, curve_values,
                                           rose_threshold)

    if plot_results:
        _plot_d_prime_npw(freq, mtf_interp, nps_interp, task_functions,
                          disc_diameters_mm, d_prime_values, contrast_hu,
                          target_directory,
                          curve_diameters_mm=curve_diameters_mm,
                          curve_d_prime=curve_values,
                          rose_threshold=rose_threshold,
                          diameter_at_threshold_mm=d_at_threshold)

    return {
        'disc_diameters_mm': disc_diameters_mm,
        'contrast_hu': contrast_hu,
        'd_prime': d_prime_values,
        'curve_diameters_mm': curve_diameters_mm,
        'curve_d_prime': curve_values,
        'rose_threshold': float(rose_threshold),
        'diameter_at_threshold_mm': d_at_threshold,
        'freq': freq,
        'task_functions': task_functions,
    }


def _plot_d_prime_npw(freq, mtf, nps, task_functions,
                      diameters, d_prime, contrast_hu, target_directory,
                      curve_diameters_mm=None, curve_d_prime=None,
                      rose_threshold=ROSE_THRESHOLD,
                      diameter_at_threshold_mm=float('nan'),
                      observer_label='NPW',
                      plot_filename='Detectability_NPW_plot.png'):
    """Save diagnostic plot for NPW/NPWE d' calculation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Task functions
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(diameters)))
    for diam, col in zip(diameters, colors):
        W = task_functions[float(diam)]
        W_norm = W / W.max() if W.max() > 0 else W
        ax.plot(freq, W_norm, color=col, linewidth=1.5,
                label=f'{diam:.2f} mm')
    ax.set_xlabel('Spatial Frequency (mm$^{-1}$)')
    ax.set_ylabel('|W(f)| (normalized)')
    ax.set_title('(a) Disc task functions', fontweight='bold')
    ax.set_xlim([0, freq.max()])
    ax.legend(fontsize=8, title='Diameter')
    ax.grid(True, alpha=0.3)

    # (b) MTF and NPS
    ax = axes[1]
    ax.plot(freq, mtf, 'b-', linewidth=1.5, label='MTF (normalized)')
    ax2 = ax.twinx()
    ax2.semilogy(freq, nps, 'r--', linewidth=1.5, label='NPS', alpha=0.8)
    ax.set_xlabel('Spatial Frequency (mm$^{-1}$)')
    ax.set_ylabel('MTF', color='b')
    ax2.set_ylabel('NPS (HU$^2$ mm$^2$)', color='r')
    ax.set_title('(b) System MTF & NPS', fontweight='bold')
    ax.set_xlim([0, freq.max()])
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (c) d' against disc diameter, linear on BOTH axes.  Linear is the point:
    # the quantity being read off is where the curve cuts the threshold, and a
    # log axis distorts both that crossing and the spacing between methods.
    ax = axes[2]
    x_max = float(np.max(diameters))
    if np.isfinite(diameter_at_threshold_mm):
        x_max = max(x_max, diameter_at_threshold_mm)
    x_max *= 1.08

    # The requested diameters are not marked on the curve: they are an
    # arbitrary sample of a continuous quantity, and the reading being made
    # here is the crossing, not the value at any one of them.
    inside = np.asarray(d_prime, dtype=np.float64)
    if curve_diameters_mm is not None and curve_d_prime is not None:
        curve_x = np.asarray(curve_diameters_mm, dtype=np.float64)
        curve_y = np.asarray(curve_d_prime, dtype=np.float64)
        ax.plot(curve_x, curve_y, '-', color='0.35', linewidth=1.5, zorder=2)
        inside = curve_y[curve_x <= x_max]

    ax.axhline(rose_threshold, color='crimson', linestyle=':', linewidth=1.5,
               zorder=1)

    inside = inside[np.isfinite(inside)]
    y_max = float(inside.max()) if inside.size else float(rose_threshold)
    y_max = max(y_max, float(rose_threshold)) * 1.18

    # Both labels hug the right edge, one above the threshold line and one
    # below it: that corner is empty whatever the curve does, while the space
    # around the crossing itself is not.
    ax.text(0.99 * x_max, rose_threshold, f"d'={rose_threshold:g} (Rose) ",
            fontsize=7, color='crimson', ha='right', va='bottom')
    if np.isfinite(diameter_at_threshold_mm):
        # The crossing is where the two dotted lines meet; a marker on top of
        # it only hides the curve at the one place worth looking at.
        ax.vlines(diameter_at_threshold_mm, 0, rose_threshold,
                  color='crimson', linestyle=':', linewidth=1.5, zorder=1)
        label = f'detectable at {diameter_at_threshold_mm:.3f} mm'
    else:
        label = 'threshold not crossed'
    ax.text(0.99 * x_max, 0.94 * rose_threshold, label, fontsize=9,
            color='crimson', fontweight='bold', ha='right', va='top')

    ax.set_xlabel('Disc diameter (mm)')
    ax.set_ylabel("Detectability index d'")
    ax.set_title(f"(c) {observer_label} d' (contrast = {contrast_hu:.0f} HU)",
                 fontweight='bold')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(target_directory, plot_filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  {observer_label} d' plot saved to: {path}")
    if np.isfinite(diameter_at_threshold_mm):
        print(f"  {observer_label} detectable disc size at d'="
              f"{rose_threshold:g}: {diameter_at_threshold_mm:.3f} mm")
    else:
        print(f"  {observer_label} d' never crosses {rose_threshold:g} on the "
              f"sampled diameters -- no detectable size to report")


# =============================================================================
# NPWE observer: NPW matched filter + a human contrast-sensitivity ("eye")
# filter. get_d_prime_npw above (E(f) implicitly = 1) remains the project
# default everywhere else -- this is an explicit alternative for comparing
# how rankings shift under a human-visual-system weighting.
#
# IMPORTANT -- E(f) exponent asymmetry (verified against literature, see
# below): the eye filter enters the numerator as E^2 but the denominator as
# E^4, NOT E^2/E^2. This is because the observer's template is matched to
# the eye-filtered signal (w = W*MTF*E, giving E^2 in the signal term), but
# the *noise* is independently filtered by E before reaching that template
# (its power spectrum scales by E^2), and the matched-filter variance is
# |template|^2 * (filtered NPS) = E^2 * (E^2 * NPS) = E^4 * NPS. Using
# E^2/E^2 throughout (as an earlier version of this function did) is the
# *prewhitening*-eye (PWE) convention, not NPWE -- see Gang et al., "The
# Generalized NEQ and Detectability Index for Tomosynthesis and Cone-Beam
# CT," PMC3845534, Eq. (8)-(9), which states this exact E^2-num/E^4-denom
# split for NPWE vs. E^2/E^2 for PWE.
#
# Burgess-style bandpass eye filter E(f) = f^n * exp(-(f/f_c)^2). n=1.3 and
# a peak at ~4 cycles/deg (50 cm viewing distance) are directly confirmed
# from Gang et al. above (their Eq. 9, citing Burgess, Li & Abbey, "Visual
# Signal Detectability With Two Noise Components," J. Opt. Soc. Am. A
# 14(9):2420-2442, 1997, which extends the original Burgess 1994 eye-filter
# model). f_c below is *derived* from that peak (f_peak = f_c*sqrt(n/2) for
# this f^n*exp(-(f/f_c)^2) form) rather than taken as a literature f_c
# value directly, since sources disagree on whether f_c denotes the peak
# location or an internal rolloff constant -- if exact reproduction of a
# specific paper's f_c is needed, verify against that paper directly.
DEFAULT_EYE_FILTER_N = 1.3
DEFAULT_EYE_FILTER_PEAK_CY_PER_DEG = 4.0
DEFAULT_EYE_FILTER_FC_CY_PER_DEG = (
    DEFAULT_EYE_FILTER_PEAK_CY_PER_DEG / np.sqrt(DEFAULT_EYE_FILTER_N / 2.0)
)

# There is no established clinical viewing protocol for this micro-CT
# phantom data (unlike a diagnostic radiology monitor/reading-distance
# setup), so these are configurable *modeling assumptions*, not measured
# values. Defaults: a generic reading distance (50 cm) at 1x magnification
# (displayed pixel size == object pixel size).
DEFAULT_VIEWING_DISTANCE_MM = 500.0
DEFAULT_MAGNIFICATION = 1.0


def eye_filter(f_mm, n=DEFAULT_EYE_FILTER_N,
               f_c_cy_per_deg=DEFAULT_EYE_FILTER_FC_CY_PER_DEG,
               viewing_distance_mm=DEFAULT_VIEWING_DISTANCE_MM,
               magnification=DEFAULT_MAGNIFICATION):
    """Human contrast-sensitivity ("eye") filter E(f), Burgess-style bandpass.

        E(f) = f_deg^n * exp(-(f_deg / f_c)^2)

    f_mm (object-space spatial frequency, cycles/mm) is converted to visual
    angle frequency f_deg (cycles/degree) assuming the image is viewed from
    `viewing_distance_mm` at `magnification` (displayed size / object
    size), via the small-angle approximation
    f_deg = f_mm * magnification * viewing_distance_mm * tan(1 deg).

    Args:
        f_mm: 1D array of spatial frequencies (mm^-1).
        n: Eye-filter frequency exponent (default 1.3).
        f_c_cy_per_deg: Eye-filter rolloff frequency in cy/deg (default 12.0).
        viewing_distance_mm: Assumed viewing distance (default 500 mm).
        magnification: Displayed size / object size (default 1.0).

    Returns:
        E: 1D array, same shape as f_mm.
    """
    f_mm = np.asarray(f_mm, dtype=np.float64)
    deg_per_mm = magnification * viewing_distance_mm * np.tan(np.pi / 180.0)
    f_deg = f_mm * deg_per_mm
    with np.errstate(over='ignore'):
        E = f_deg ** n * np.exp(-(f_deg / f_c_cy_per_deg) ** 2)
    return E


def get_d_prime_npwe(mtf_freq, mtf, nps_freq, nps,
                     disc_diameters_mm=None, contrast_hu=None,
                     normalize_mtf=True, n_freq=500,
                     rose_threshold=ROSE_THRESHOLD, curve_diameters_mm=None,
                     eye_filter_n=DEFAULT_EYE_FILTER_N,
                     eye_filter_fc_cy_per_deg=DEFAULT_EYE_FILTER_FC_CY_PER_DEG,
                     viewing_distance_mm=DEFAULT_VIEWING_DISTANCE_MM,
                     magnification=DEFAULT_MAGNIFICATION,
                     plot_results=True, target_directory=None):
    """Compute NPWE-observer (non-prewhitening + eye filter) detectability index.

    Same NPW matched-filter structure as get_d_prime_npw, with the human
    eye filter E(f) (see eye_filter()) applied to the template. E(f)
    enters the numerator squared but the denominator to the 4th power --
    see the module-level comment above eye_filter() for the derivation and
    literature citation (this asymmetry, not E^2/E^2, is the NPWE
    convention; E^2/E^2 throughout is the distinct PWE convention):

        d'^2 = [ integral  |W(f)|^2 MTF(f)^2 E(f)^2 f df ]^2
               / integral  |W(f)|^2 MTF(f)^2 E(f)^4 NPS(f) f df

    Also note: this integral runs over radial frequency with an explicit
    f df term, i.e. the 2D frequency integral already collapsed to polar
    coordinates (f df = the polar-coordinates Jacobian, with the constant
    angular factor folded into the overall scale) under the assumption
    that W, MTF, NPS, and E are all radially symmetric. `f_max` /
    `freq[-1]` is therefore the *radial* (isotropic) Nyquist frequency,
    not the axial (row/column) Nyquist.

    get_d_prime_npw (equivalent to E(f) = 1 everywhere) remains the default
    detectability metric used elsewhere in this project; this function is
    an explicit alternative for comparing rankings under a human-visual-
    system-weighted observer. See eye_filter() docstring for the viewing
    distance / magnification assumptions this relies on.

    Args/Returns: identical to get_d_prime_npw -- including the dense d'
    curve and the `rose_threshold` crossing reported as a detectable disc
    size in mm -- plus the eye_filter_n, eye_filter_fc_cy_per_deg,
    viewing_distance_mm, magnification knobs above, and an added
    'eye_filter' key in the returned dict (E(f) on the same 'freq' grid).
    """
    if disc_diameters_mm is None:
        disc_diameters_mm = list(DEFAULT_DISC_DIAMETERS)
    if contrast_hu is None:
        contrast_hu = DEFAULT_CONTRAST_HU
    if target_directory is None:
        target_directory = os.getcwd()

    disc_diameters_mm = np.asarray(disc_diameters_mm, dtype=np.float64)
    if curve_diameters_mm is None:
        curve_diameters_mm = curve_diameter_grid(disc_diameters_mm)
    curve_diameters_mm = np.asarray(curve_diameters_mm, dtype=np.float64)

    freq, mtf_interp, nps_interp = _resample_mtf_nps(
        mtf_freq, mtf, nps_freq, nps, normalize_mtf, n_freq)

    E = eye_filter(freq, n=eye_filter_n, f_c_cy_per_deg=eye_filter_fc_cy_per_deg,
                   viewing_distance_mm=viewing_distance_mm, magnification=magnification)

    d_prime_values, task_functions = _d_prime_for_diameters(
        freq, mtf_interp, nps_interp, disc_diameters_mm, contrast_hu, eye=E)
    curve_values, _ = _d_prime_for_diameters(
        freq, mtf_interp, nps_interp, curve_diameters_mm, contrast_hu, eye=E)
    d_at_threshold = diameter_at_threshold(curve_diameters_mm, curve_values,
                                           rose_threshold)

    if plot_results:
        _plot_d_prime_npw(freq, mtf_interp, nps_interp, task_functions,
                          disc_diameters_mm, d_prime_values, contrast_hu,
                          target_directory,
                          curve_diameters_mm=curve_diameters_mm,
                          curve_d_prime=curve_values,
                          rose_threshold=rose_threshold,
                          diameter_at_threshold_mm=d_at_threshold,
                          observer_label='NPWE',
                          plot_filename='Detectability_NPWE_plot.png')

    return {
        'disc_diameters_mm': disc_diameters_mm,
        'contrast_hu': contrast_hu,
        'd_prime': d_prime_values,
        'curve_diameters_mm': curve_diameters_mm,
        'curve_d_prime': curve_values,
        'rose_threshold': float(rose_threshold),
        'diameter_at_threshold_mm': d_at_threshold,
        'freq': freq,
        'task_functions': task_functions,
        'eye_filter': E,
    }


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

    parser = argparse.ArgumentParser(
        description='Calculate the Detectability Index (d\') from CT image data.')
    sub = parser.add_subparsers(dest='mode', help='Calculation mode')

    # --- TTF-based mode (original) ---
    p_ttf = sub.add_parser('ttf', help='TTF-based d\' from circular phantom inserts')
    p_ttf.add_argument('--input_ttf', required=True, help='Path to TTF image file (.npy, .npz, or .vff)')
    p_ttf.add_argument('--input_nps', required=True, help='Path to NPS image file (.npy, .npz, or .vff)')
    p_ttf.add_argument('--centre_pixels', required=True, type=parse_centre_pixels,
                        help='Centre pixels as semicolon-separated y,x pairs (e.g. "686,398;418,132")')
    p_ttf.add_argument('--radius', required=True, type=int, help='Radius of circular edges in pixels')
    p_ttf.add_argument('--materials', required=True, type=str,
                        help='Comma-separated material names (e.g. "SB3,Teflon,Fat,Tissue")')
    p_ttf.add_argument('--roi_bounds', required=True, type=parse_roi_bounds,
                        help='NPS ROI bounds as semicolon-separated y1,y2,x1,x2 groups')
    p_ttf.add_argument('--task_material', required=True, type=str, help='Task function material (e.g. "Fat")')
    p_ttf.add_argument('--task_contrast', type=float, default=-160, help='Task function contrast in HU (default: -160)')
    p_ttf.add_argument('--task_object_size', type=float, default=0.2, help='Task function object radius in mm (default: 0.2)')
    p_ttf.add_argument('--slices_ttf', type=str, default=None, help='Slice selection for TTF data')
    p_ttf.add_argument('--slices_nps', type=str, default=None, help='Slice selection for NPS data')
    p_ttf.add_argument('--pixel_size', type=float, default=0.05, help='Pixel size in mm (default: 0.05)')
    p_ttf.add_argument('--output_dir', type=str, default='./results', help='Output directory (default: ./results)')
    p_ttf.add_argument('--no_plot', action='store_true', help='Disable plot generation')
    p_ttf.add_argument('--show', action='store_true', help='Display plots interactively')
    p_ttf.add_argument('--quiet', action='store_true', help='Suppress console output')

    # --- NPW mode (frequency-domain, from pre-computed MTF/NPS) ---
    p_npw = sub.add_parser('npw', help='NPW-observer d\' from pre-computed MTF/NPS curves (AAPM TG-233)')
    p_npw.add_argument('--mtf_npz', required=True,
                        help='Path to .npz with "mtf_freq" and "mtf" arrays')
    p_npw.add_argument('--nps_npz', required=True,
                        help='Path to .npz with "nps_freq" and "nps" arrays')
    p_npw.add_argument('--disc_diameters', type=str, default=None,
                        help='Comma-separated disc diameters in mm (default: 0.15,0.5,1.0,3.0)')
    p_npw.add_argument('--contrast', type=float, default=DEFAULT_CONTRAST_HU,
                        help=f'Contrast in HU (default: {DEFAULT_CONTRAST_HU})')
    p_npw.add_argument('--rose_threshold', type=float, default=ROSE_THRESHOLD,
                        help='Detectability threshold at which the detectable '
                             f'disc size is read off (default: {ROSE_THRESHOLD:g}, '
                             'the Rose criterion)')
    p_npw.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    p_npw.add_argument('--no_plot', action='store_true', help='Disable plot generation')
    p_npw.add_argument('--show', action='store_true', help='Display plots interactively')

    # Default to TTF mode for backwards compatibility
    return parser.parse_args()


def main():
    args = parse_args()
    from .helper_scripts.io_utils import load_image_data, ensure_output_dir, parse_slices

    if args.mode == 'npw':
        output_dir = ensure_output_dir(args.output_dir)
        mtf_data = np.load(args.mtf_npz)
        nps_data = np.load(args.nps_npz)

        diameters = None
        if args.disc_diameters:
            diameters = [float(x) for x in args.disc_diameters.split(',')]

        result = get_d_prime_npw(
            mtf_data['mtf_freq'], mtf_data['mtf'],
            nps_data['nps_freq'], nps_data['nps'],
            disc_diameters_mm=diameters,
            contrast_hu=args.contrast,
            rose_threshold=args.rose_threshold,
            plot_results=not args.no_plot,
            target_directory=output_dir,
        )
        dp_header = "d'"
        print(f"\nNPW Detectability Index (ΔC = {result['contrast_hu']:.0f} HU)")
        print(f"{'Diameter (mm)':>14s}  {dp_header:>8s}  {'Detectable':>10s}")
        print('-' * 36)
        threshold = result['rose_threshold']
        for d, dp in zip(result['disc_diameters_mm'], result['d_prime']):
            flag = 'Yes' if dp >= threshold else 'No'
            print(f'{d:>14.2f}  {dp:>8.1f}  {flag:>10s}')
        print('-' * 36)
        size = result['diameter_at_threshold_mm']
        if np.isfinite(size):
            print(f"Detectable disc size at the Rose threshold "
                  f"(d'={threshold:g}): {size:.3f} mm")
        else:
            print(f"d' does not cross the Rose threshold (d'={threshold:g}) "
                  f"anywhere on the sampled diameters")

    else:
        # TTF mode (original behaviour / default)
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
        plt.show()


if __name__ == '__main__':
    main()
