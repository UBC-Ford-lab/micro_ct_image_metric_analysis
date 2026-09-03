"""Azimuthal (radial) averaging of a 2D image over concentric annuli.

``radial_profile`` mirrors ``photutils.profiles.RadialProfile``: ``radii``
are the annulus EDGES, ``.radius`` the annulus centres and ``.profile`` the
area-weighted mean of the pixels inside each annulus. When photutils is
installed it is used, so results are bit-identical to earlier versions;
otherwise a NumPy implementation samples every pixel on a fine sub-grid to
estimate the pixel/annulus overlap area. The two agree to well under 1 %
on the profiles the calculators use (see tests).
"""

from __future__ import annotations

import numpy as np

try:  # optional: exact circle/pixel overlap geometry
    from photutils.profiles import RadialProfile as _PhotutilsRadialProfile
except Exception:  # pragma: no cover - depends on the environment
    _PhotutilsRadialProfile = None

#: Sub-samples per pixel edge for the overlap estimate (100 per pixel).
SUBPIXELS = 10
#: Upper bound on sub-samples held in memory at once.
_CHUNK = 4_000_000


class RadialProfile:
    """``radius`` and ``profile`` for an image about ``centre``.

    :param image: 2D array.
    :param centre: ``(x, y)`` pixel coordinates of the centre, photutils order.
    :param radii: 1D increasing array of annulus edges in pixels.
    """

    def __init__(self, image, centre, radii):
        image = np.asarray(image, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError("image must be 2D")
        if radii.ndim != 1 or radii.size < 2 or np.any(np.diff(radii) <= 0):
            raise ValueError("radii must be a 1D increasing array of at least 2 edges")
        self.radii = radii
        self.radius = 0.5 * (radii[:-1] + radii[1:])
        self.profile = _annulus_means(image, float(centre[0]), float(centre[1]), radii)


def _annulus_means(image, cx, cy, radii):
    """Area-weighted mean per annulus by sub-pixel sampling, chunked by rows."""
    ny, nx = image.shape
    n_ann = radii.size - 1
    r_max = radii[-1]
    # Only pixels within r_max + 1 of the centre can overlap any annulus.
    y_lo = max(0, int(np.floor(cy - r_max - 1)))
    y_hi = min(ny, int(np.ceil(cy + r_max + 1)) + 1)
    x_lo = max(0, int(np.floor(cx - r_max - 1)))
    x_hi = min(nx, int(np.ceil(cx + r_max + 1)) + 1)
    if y_hi <= y_lo or x_hi <= x_lo:
        return np.full(n_ann, np.nan)

    s = SUBPIXELS
    offs = (np.arange(s) + 0.5) / s - 0.5              # sub-sample offsets in a pixel
    dx_sub = (np.arange(x_lo, x_hi)[:, None] + offs[None, :]).ravel() - cx   # (W*s,)
    sums = np.zeros(n_ann)
    counts = np.zeros(n_ann)
    rows_per_chunk = max(1, _CHUNK // max(1, dx_sub.size * s))
    for y0 in range(y_lo, y_hi, rows_per_chunk):
        y1 = min(y_hi, y0 + rows_per_chunk)
        dy_sub = (np.arange(y0, y1)[:, None] + offs[None, :]).ravel() - cy    # (H*s,)
        r = np.hypot(dy_sub[:, None], dx_sub[None, :])                        # (H*s, W*s)
        bins = np.searchsorted(radii, r.ravel(), side='right') - 1
        vals = np.repeat(np.repeat(image[y0:y1, x_lo:x_hi], s, axis=0), s, axis=1).ravel()
        inside = (bins >= 0) & (bins < n_ann)
        sums += np.bincount(bins[inside], weights=vals[inside], minlength=n_ann)
        counts += np.bincount(bins[inside], minlength=n_ann)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(counts > 0, sums / counts, np.nan)


def radial_profile(image, centre, radii, *, exact=None):
    """Radial profile of ``image`` about ``centre`` over the annulus edges ``radii``.

    Uses photutils when it is installed (``exact=None``), or when ``exact``
    is True; ``exact=False`` forces the NumPy implementation.
    """
    use_photutils = _PhotutilsRadialProfile is not None if exact is None else bool(exact)
    if use_photutils:
        if _PhotutilsRadialProfile is None:
            raise ImportError("photutils is not installed; pip install photutils")
        return _PhotutilsRadialProfile(np.asarray(image, dtype=np.float64), tuple(centre),
                                       np.asarray(radii, dtype=np.float64))
    return RadialProfile(image, centre, radii)


def photutils_available() -> bool:
    return _PhotutilsRadialProfile is not None
