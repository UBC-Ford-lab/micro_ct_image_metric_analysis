"""Read and write GE ``.vff`` volumes with plain NumPy.

The format is an ASCII header (``key=value;`` lines) terminated by a form
feed, followed by big-endian integer voxels in z-major order. Files written
here use GE's ``ncaa`` header dialect so they load in MicroView / Amalytics.
"""

import os

import numpy as np


def read_vff_header(filename, verbose=False):
    """Parse the ASCII header of a VFF file into a dict of strings."""
    header = {}
    with open(filename, 'rb') as file:
        content = file.read(1000).decode('latin-1')
    try:
        header_content, _ = content.split('\f', 1)
    except ValueError:  # no form feed in the first kilobyte
        header_content = content
    for line in header_content.splitlines():
        if '=' not in line:
            continue
        key, value = line.strip().split('=', 1)
        header[key.strip()] = value.strip().rstrip(';')
    if verbose:
        for key, value in header.items():
            print(f"{key}: {value}")
    return header


def read_vff_data(filename, header, verbose=False, memmap_above_gb=0.3):
    """Read the voxel payload described by ``header`` as a ``(z, y, x)`` array.

    Volumes larger than ``memmap_above_gb`` are memory-mapped (copy-on-write)
    instead of being read into RAM.
    """
    size = [int(s) for s in header['size'].split()]
    xdim, ydim = size[0], size[1]
    zdim = size[2] if len(size) > 2 else 1
    bits = int(header['bits'])
    if bits not in (8, 16):
        raise ValueError(f"unsupported bits per voxel: {bits}")
    data_type = np.dtype('>b') if bits == 8 else np.dtype('>h')
    data_size = xdim * ydim * zdim * (bits // 8)
    offset = os.path.getsize(filename) - data_size
    if offset < 0:
        raise ValueError(f"{filename}: header claims {data_size} bytes of voxels "
                         f"but the file only has {os.path.getsize(filename)}")
    shape = (zdim, ydim, xdim)
    if data_size > memmap_above_gb * 1024 ** 3:
        if verbose:
            print(f"memory-mapping {data_size / 1024 ** 3:.2f} GB of voxels")
        return np.memmap(filename, dtype=data_type, mode='c', offset=offset, shape=shape)
    return np.fromfile(filename, dtype=data_type, offset=offset).reshape(shape)


def read_vff(filename, verbose=False):
    """Return ``(header, data)`` for a VFF file; ``data`` is shaped ``(z, y, x)``."""
    header = read_vff_header(filename, verbose=verbose)
    data = read_vff_data(filename, header, verbose=verbose)
    if verbose:
        print("loaded", filename, "shape", data.shape)
    return header, data


def write_vff(filename, header, data, verbose=False):
    """Write a 2D or 3D array as a VFF file with a GE ``ncaa`` header.

    Recognised ``header`` keys: ``bits`` (8 or 16, default 16),
    ``elementsize`` (isotropic voxel size in mm) or ``spacing`` (``"dx dy dz"``,
    first value used), ``water`` and ``air`` (HU calibration anchors,
    default 0 and -1000 so stored values are read back as HU unchanged).
    The size is always taken from ``data``. Float input is rounded and
    clipped to the integer range, never truncated or wrapped.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"data must be 2D or 3D, got {arr.ndim}D")
    zdim, ydim, xdim = arr.shape

    bits = int(header.get('bits', 16))
    if bits == 8:
        dtype = np.dtype('>u1')
    elif bits == 16:
        dtype = np.dtype('>i2')
    else:
        raise ValueError("bits must be 8 or 16")

    if 'elementsize' in header:
        elementsize = float(header['elementsize'])
    elif 'spacing' in header:
        toks = str(header['spacing']).split()
        elementsize = float(toks[0])
        if verbose and len(toks) >= 3 and any(abs(float(t) - elementsize) > 1e-9 for t in toks):
            print(f"warning: anisotropic spacing {toks}; VFF elementsize is a scalar, "
                  f"writing {elementsize} for all axes")
    else:
        elementsize = 1.0

    water = float(header.get('water', 0.0))
    air = float(header.get('air', -1000.0))

    if np.issubdtype(arr.dtype, np.floating):
        info = np.iinfo(dtype)
        n_clip = int(np.count_nonzero((arr < info.min) | (arr > info.max)))
        if n_clip and verbose:
            print(f"warning: {n_clip} voxel(s) outside [{info.min}, {info.max}] clipped")
        arr_be = np.rint(np.clip(arr, info.min, info.max)).astype(dtype)
    else:
        arr_be = np.ascontiguousarray(arr, dtype=dtype)

    dmin = float(arr_be.min()) if arr_be.size else 0.0
    dmax = float(arr_be.max()) if arr_be.size else 0.0
    lines = [
        "ncaa",
        "rank=3;",
        "type=raster;",
        "modality=CT;",
        f"size={xdim} {ydim} {zdim};",
        "origin=0.0 0.0 0.0;",
        "bands=1;",
        f"bits={bits};",
        "format=slice;",
        "spacing=1.00 1.00 1.00;",
        f"elementsize={elementsize:.6f};",
        f"min={dmin:.6f};",
        f"max={dmax:.6f};",
        f"water={water:.6f};",
        f"air={air:.6f};",
    ]
    header_text = "\n".join(lines) + "\n\f\n"
    if verbose:
        print(f"writing {filename}: shape {arr_be.shape}, {bits}-bit, range [{dmin:.0f}, {dmax:.0f}]")
    with open(filename, 'wb') as f:
        f.write(header_text.encode('latin-1'))
        arr_be.tofile(f)
