"""Load a volume as a ``(z, y, x)`` array together with its :class:`Frame`.

Accepted inputs: ``.npy``, ``.npz`` (first array) and GE ``.vff``. The
voxel size comes from, in order of precedence: the ``pixel_size`` argument,
a ``<volume>.json`` sidecar (``voxel_size_mm``/``volume_shape``/``vol_origin_mm``
as written by the eXplore CT 120 reconstruction package), or the VFF
``elementsize`` header. Without any of these the loader refuses rather than
guessing, because every metric is in physical units.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .helpers.io_utils import load_image_data
from .helpers.vff_io import read_vff_header
from .rois import Frame


def sidecar_path(path) -> Path:
    return Path(path).with_suffix('.json')


def load_sidecar(path) -> dict:
    p = sidecar_path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def load_volume(path, *, pixel_size=None, slice_thickness=None, flip=(),
                origin_mm=None, dtype=np.float32):
    """Return ``(image_zyx, frame, info)`` for a volume file.

    :param pixel_size: in-plane voxel size in mm; overrides any metadata.
    :param slice_thickness: z spacing in mm (default: the pixel size).
    :param flip: axes to reverse after loading, any of ``'x'``, ``'y'``, ``'z'``.
    :param origin_mm: ``(x, y, z)`` position of the volume centre
        (default: from the sidecar, else 0).
    :param dtype: the array is cast to this (memmaps become arrays).
    """
    path = Path(path)
    image = np.asarray(load_image_data(str(path)))
    if image.ndim == 2:
        image = image[np.newaxis]
    if image.ndim != 3:
        raise ValueError(f"{path}: expected a 2D or 3D array, got {image.ndim}D")
    image = image.astype(dtype, copy=False)

    side = load_sidecar(path)
    header = read_vff_header(path) if path.suffix.lower() == '.vff' else {}
    info = {'path': str(path), 'sidecar': bool(side), 'source': None}

    dx, dz = pixel_size, slice_thickness
    if dx is None and side.get('voxel_size_mm'):
        vs = side['voxel_size_mm']
        if isinstance(vs, dict):
            dx = float(vs.get('xy', vs.get('x')))
            dz = dz if dz is not None else float(vs.get('z', dx))
        else:
            dx = float(np.atleast_1d(vs)[0])
        info['source'] = 'sidecar'
    if dx is None and header.get('elementsize'):
        dx = float(header['elementsize'])
        info['source'] = 'vff header'
    if dx is None and header.get('spacing'):
        toks = [float(t) for t in str(header['spacing']).split()]
        if toks and toks[0] not in (0.0, 1.0):
            dx = toks[0]
            dz = dz if dz is not None else (toks[2] if len(toks) > 2 else dx)
            info['source'] = 'vff header'
    if dx is None:
        raise ValueError(
            f"{path}: no voxel size found (no sidecar, no VFF elementsize). "
            f"Pass pixel_size (mm) explicitly.")
    if pixel_size is not None:
        info['source'] = 'argument'
    if dz is None:
        dz = dx

    if origin_mm is None:
        o = side.get('vol_origin_mm')
        origin_mm = tuple(float(v) for v in o) if o else (0.0, 0.0, 0.0)

    for ax in flip:
        image = np.flip(image, axis={'z': 0, 'y': 1, 'x': 2}[ax])
    image = np.ascontiguousarray(image)

    frame = Frame.for_image(image, dx, dz, origin_mm=origin_mm)
    info.update({'pixel_size_mm': dx, 'slice_thickness_mm': dz,
                 'origin_mm': list(origin_mm), 'shape_zyx': list(image.shape),
                 'units': side.get('units', header.get('modality', 'unknown'))})
    return image, frame, info
