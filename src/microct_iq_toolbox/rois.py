"""Metric regions in millimetres, resolved per volume.

The calculators take their regions as pixel indices, which are only
meaningful on one grid. Two reconstructions of the same phantom at different
voxel sizes need different indices for the same piece of phantom, and a
hand-written index list silently measures a different part of it in each.

So a region is specified ONCE in isocentre-centred millimetres
(:class:`PhantomROIs`) and converted per volume through that volume's own
geometry (:class:`Frame`). Positions are ``(x, y)`` in the plane and ``z``
along the axis; the calculators' images are ``(z, y, x)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Geometry: millimetres <-> voxel indices
# --------------------------------------------------------------------------

def index_of_mm(mm, n: int, spacing: float, origin: float) -> np.ndarray:
    """Fractional voxel index of a position in mm, on one axis.

    Voxel centre ``i`` sits at ``(i - (n-1)/2) * spacing + origin``.
    """
    return (np.asarray(mm, dtype=np.float64) - float(origin)) / float(spacing) \
        + (int(n) - 1) / 2.0


def mm_of_index(idx, n: int, spacing: float, origin: float) -> np.ndarray:
    """Position in mm of a fractional voxel index, on one axis."""
    return (np.asarray(idx, dtype=np.float64) - (int(n) - 1) / 2.0) \
        * float(spacing) + float(origin)


def nearest_index(value) -> int:
    """A fractional index as a whole one, resolving an exact tie HALF UP.

    Not ``round``: Python rounds halves to even, so the answer would depend
    on the parity of the number being rounded, and an ROI centre lands on an
    exact half whenever the grid has an even voxel count. Half-up is
    shift-equivariant on the lattice, so the ROI follows a crop exactly.
    """
    return int(np.floor(np.float64(value) + 0.5))


class Frame:
    """One volume's mm <-> index mapping.

    ``shape`` is ``(nx, ny, nz)``, ``origin`` the mm position of the volume
    centre, ``dx`` the in-plane voxel size and ``dz`` the slice spacing.
    Every index this class returns is in the calculators' ``(z, y, x)``
    order.

    Construct from a geometry dict (``vol_shape``/``vol_origin``/``dx``/``dz``,
    arrays in ``(x, y, z)``), or with :meth:`for_image` from a ``(z, y, x)``
    array and a pixel size.
    """

    def __init__(self, geometry: dict):
        self.shape = tuple(int(v) for v in geometry['vol_shape'])   # (Nx,Ny,Nz)
        self.origin = tuple(float(v) for v in geometry.get('vol_origin', (0.0, 0.0, 0.0)))
        self.dx = float(geometry['dx'])
        self.dz = float(geometry.get('dz', geometry['dx']))

    @classmethod
    def for_image(cls, image_zyx, pixel_size: float, slice_thickness=None,
                  origin_mm=(0.0, 0.0, 0.0)) -> "Frame":
        """A frame for a ``(z, y, x)`` array with the origin at its centre."""
        nz, ny, nx = np.shape(image_zyx)[-3:]
        return cls({'vol_shape': (nx, ny, nz), 'vol_origin': origin_mm,
                    'dx': float(pixel_size),
                    'dz': float(slice_thickness if slice_thickness else pixel_size)})

    @property
    def image_shape(self) -> tuple:
        """``(nz, ny, nx)``."""
        return (self.shape[2], self.shape[1], self.shape[0])

    def as_dict(self) -> dict:
        return {'vol_shape': list(self.shape), 'vol_origin': list(self.origin),
                'dx': self.dx, 'dz': self.dz}

    # -- per-axis ------------------------------------------------------
    def ix(self, x_mm):
        return index_of_mm(x_mm, self.shape[0], self.dx, self.origin[0])

    def iy(self, y_mm):
        return index_of_mm(y_mm, self.shape[1], self.dx, self.origin[1])

    def iz(self, z_mm):
        return index_of_mm(z_mm, self.shape[2], self.dz, self.origin[2])

    def x_mm(self, ix):
        return mm_of_index(ix, self.shape[0], self.dx, self.origin[0])

    def y_mm(self, iy):
        return mm_of_index(iy, self.shape[1], self.dx, self.origin[1])

    def z_mm(self, iz):
        return mm_of_index(iz, self.shape[2], self.dz, self.origin[2])

    def px(self, mm) -> int:
        """A LENGTH in mm as a whole number of in-plane pixels, at least 1."""
        return max(1, int(round(float(mm) / self.dx)))

    # -- regions -------------------------------------------------------
    def to_image(self, volume) -> np.ndarray:
        """``(x, y, z)`` -> the calculators' ``(z, y, x)``."""
        vol = np.asarray(volume)
        if tuple(vol.shape) != self.shape:
            raise ValueError(
                f"volume shape {tuple(vol.shape)} does not match the geometry "
                f"it was given ({self.shape})")
        return vol.transpose(2, 1, 0)

    def check_image(self, image) -> np.ndarray:
        """Assert a ``(z, y, x)`` array matches this frame and return it."""
        img = np.asarray(image)
        if tuple(img.shape) != self.image_shape:
            raise ValueError(
                f"image shape {tuple(img.shape)} does not match the frame's "
                f"(z, y, x) shape {self.image_shape}")
        return img

    def z_slice(self, z_mm_range) -> slice:
        """The z slices spanning a mm interval, clipped to the volume.

        Rounded OUTWARD (floor/ceil) so a section is never silently trimmed by
        a fraction of a slice on a coarse grid.
        """
        lo, hi = (float(v) for v in z_mm_range)
        a = int(np.floor(self.iz(min(lo, hi))))
        b = int(np.ceil(self.iz(max(lo, hi)))) + 1
        a, b = max(0, a), min(self.shape[2], b)
        if b <= a:
            raise ValueError(
                f"z range {z_mm_range} mm lies outside this volume, which "
                f"spans {self.z_mm(0):.2f} to {self.z_mm(self.shape[2]-1):.2f} mm")
        return slice(a, b)

    def box(self, centre_mm, size_mm) -> list:
        """A rectangle as ``[y1, y2, x1, x2]``, the calculators' order.

        ``size_mm`` is ``(width_x, height_y)``. The extents are rounded to
        whole pixels FIRST and the box then centred, so two volumes get boxes
        that differ by at most half a voxel in placement rather than
        accumulating a rounding error in the size.
        """
        cx, cy = (float(v) for v in centre_mm)
        wx, hy = (float(v) for v in np.atleast_1d(size_mm) * np.ones(2))
        nx, ny = self.px(wx), self.px(hy)
        x0 = nearest_index(self.ix(cx) - nx / 2.0)
        y0 = nearest_index(self.iy(cy) - ny / 2.0)
        return [y0, y0 + ny, x0, x0 + nx]

    def square(self, centre_mm, side_mm) -> list:
        """A square ROI, what ``get_NPS`` requires of every one of its ROIs."""
        n = self.px(side_mm)
        cx, cy = (float(v) for v in centre_mm)
        x0 = nearest_index(self.ix(cx) - n / 2.0)
        y0 = nearest_index(self.iy(cy) - n / 2.0)
        return [y0, y0 + n, x0, x0 + n]

    def centre_px(self, centre_mm) -> list:
        """A point as ``[y, x]`` whole pixels, the calculators' order."""
        cx, cy = (float(v) for v in centre_mm)
        return [nearest_index(self.iy(cy)), nearest_index(self.ix(cx))]

    def contains_box(self, box) -> bool:
        y1, y2, x1, x2 = box
        return (0 <= y1 < y2 <= self.shape[1]
                and 0 <= x1 < x2 <= self.shape[0])


# --------------------------------------------------------------------------
# The ROI table
# --------------------------------------------------------------------------

@dataclass
class PhantomROIs:
    """Where each metric's target sits in the phantom, in mm.

    Written once per phantom (by hand or by ``ct-find-rois``) and reused for
    every volume of that phantom. Positions are isocentre-centred
    millimetres, so they survive any change of voxel size or grid.

    Sections (each optional):

    ``mtf``: ``z_mm`` [lo, hi], ``centre_mm`` [x, y], ``size_mm`` [w, h],
    ``edge_angle_deg``, ``high_to_low``.
    ``nps``: ``z_mm``, ``roi_centres_mm`` [[x, y], ...], ``size_mm``.
    ``ttf``: ``z_mm``, ``centres_mm`` [[x, y], ...], ``radius_mm``,
    ``materials`` [names].
    ``d_prime``: ``disc_diameters_mm``, ``contrast_hu``, ``rose_threshold``.
    """

    scan: str = ""
    reference_volume: str = ""
    created: str = ""
    notes: str = ""
    mtf: dict = field(default_factory=dict)
    nps: dict = field(default_factory=dict)
    ttf: dict = field(default_factory=dict)
    d_prime: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "PhantomROIs":
        record = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in record.items() if k in known})

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    # -- resolution against one volume ---------------------------------
    def resolve(self, frame: Frame) -> dict:
        """Every region as pixel indices on ``frame``, plus what was dropped.

        A region that does not fit inside the volume is REPORTED, not
        clamped: a clamped NPS ROI would change size and a clamped MTF crop
        would move the edge, and either would produce a number that looks
        like a measurement.
        """
        out = {'skipped': []}

        if self.mtf:
            box = frame.box(self.mtf['centre_mm'], self.mtf['size_mm'])
            # get_MTF requires the y extent to be exactly twice the x extent
            # and silently reshapes the crop when it is not. Do it here so
            # the crop that gets measured is the one recorded.
            y1, y2, x1, x2 = box
            width = x2 - x1
            y_mid = (y1 + y2) // 2
            box = [y_mid - width, y_mid + width, x1, x2]
            if frame.contains_box(box):
                out['mtf'] = {
                    'crop_indices': box,
                    'slices': frame.z_slice(self.mtf['z_mm']),
                    'edge_angle': float(self.mtf['edge_angle_deg']),
                    'high_to_low': bool(self.mtf.get('high_to_low', True)),
                }
            else:
                out['skipped'].append(('mtf', f"crop {box} outside the volume"))

        if self.nps:
            boxes, dropped = [], 0
            for c in self.nps['roi_centres_mm']:
                b = frame.square(c, self.nps['size_mm'])
                if frame.contains_box(b):
                    boxes.append(b)
                else:
                    dropped += 1
            if boxes:
                out['nps'] = {
                    'roi_bounds': np.array(boxes, dtype=int),
                    'slices': frame.z_slice(self.nps['z_mm']),
                    'n_dropped': dropped,
                }
                if dropped:
                    out['skipped'].append(
                        ('nps', f"{dropped} of {len(self.nps['roi_centres_mm'])} "
                                f"ROIs outside the volume"))
            else:
                out['skipped'].append(('nps', "every ROI outside the volume"))

        if self.ttf:
            radius = frame.px(self.ttf['radius_mm'])
            centres, materials = [], []
            names = self.ttf.get('materials') or [
                f"insert {i+1}" for i in range(len(self.ttf['centres_mm']))]
            for c, m in zip(self.ttf['centres_mm'], names):
                cy, cx = frame.centre_px(c)
                # get_TTF crops a square of side 2*radius around each centre.
                if (radius <= cx < frame.shape[0] - radius
                        and radius <= cy < frame.shape[1] - radius):
                    centres.append([cy, cx])
                    materials.append(m)
            if centres:
                out['ttf'] = {
                    'centre_pixels': centres,
                    'radius': int(radius),
                    'materials': materials,
                    'slices': frame.z_slice(self.ttf['z_mm']),
                }
                if len(centres) < len(self.ttf['centres_mm']):
                    out['skipped'].append(
                        ('ttf', f"{len(self.ttf['centres_mm']) - len(centres)}"
                                f" inserts too close to the volume edge"))
            else:
                out['skipped'].append(('ttf', "every insert outside the volume"))

        return out
