# Description: Shared I/O utilities for loading image data and parsing CLI arguments.

import os
import numpy as np


def load_image_data(filepath):
    """Load image data from .npy, .npz, or .vff files.

    :param filepath: Path to the image file
    :return: Image data as a numpy array
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npy':
        return np.load(filepath)
    elif ext == '.npz':
        data = np.load(filepath)
        # Return the first array in the archive
        return data[list(data.keys())[0]]
    elif ext == '.vff':
        from . import vff_io
        return vff_io.read_vff(filepath, verbose=False)[1]
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .npy, .npz, or .vff")


def ensure_output_dir(path):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def parse_int_list(s):
    """Parse a comma-separated string of integers.

    Example: "270,664,522,640" -> [270, 664, 522, 640]
    """
    return [int(x.strip()) for x in s.split(',')]


def parse_float_list(s):
    """Parse a comma-separated string of floats.

    Example: "0.1,0.2,0.3" -> [0.1, 0.2, 0.3]
    """
    return [float(x.strip()) for x in s.split(',')]


def parse_roi_bounds(s):
    """Parse ROI bounds from semicolon-separated groups of 4 integers.

    Example: "178,294,510,626;258,374,750,866" -> Nx4 numpy array
    """
    rows = []
    for group in s.split(';'):
        rows.append([int(x.strip()) for x in group.split(',')])
    return np.array(rows)


def parse_centre_pixels(s):
    """Parse centre pixel coordinates from semicolon-separated y,x pairs.

    Example: "1335,2141;765,1914" -> [[1335, 2141], [765, 1914]]
    """
    centres = []
    for pair in s.split(';'):
        coords = [int(x.strip()) for x in pair.split(',')]
        centres.append(coords)
    return centres


def parse_slices(s):
    """Parse slice specification into a concatenated numpy index array.

    Example: "0:30,140:182" -> np.concatenate((np.arange(0,30), np.arange(140,182)))
    Example: "10:160" -> np.arange(10, 160)
    """
    arrays = []
    for part in s.split(','):
        if ':' in part:
            start, stop = part.split(':')
            arrays.append(np.arange(int(start.strip()), int(stop.strip())))
        else:
            arrays.append(np.array([int(part.strip())]))
    return np.concatenate(arrays)
