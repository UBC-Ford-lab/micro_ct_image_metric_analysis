# Description: This script reads a VFF file and returns the header and the data.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
from pathlib import Path

# key insights into what a vff file looks like came from https://imagej.net/ij/plugins/download/Multi_VFF_Opener.java

def read_vff_header(filename, verbose=True):
    '''
    This function reads the header of a VFF file and returns a dictionary with the header data.
    Don't call this on its own (use read_vff() instead).
    :param filename: The path to the VFF file (including the file name.vff)
    :return: A dictionary with the header data
    '''
    header = {}
    with open(filename, 'rb') as file:
        # Using 'latin-1' to avoid decoding issues
        content = file.read(1000).decode('latin-1') # Read the first 1000 bytes (header is < 1000 bytes)

        # Split on form feed, take the first part
        try:
            header_content, _ = content.split('\f', 1)
        except ValueError: # sometimes there is no form feed
            header_content = content
        lines = header_content.splitlines()
        for line in lines:
            if '=' in line:  # Simple check to filter out irrelevant lines
                key, value = line.strip().split('=')
                header[key.strip()] = value.strip()[:-1]

        # Print the header data if verbose is set to True
        if verbose:
            print("--------------------------Header Data----------------------------")
            for key, value in header.items():
                print(f"{key}: {value}")
            print("-----------------------End of Header Data-------------------------")

    return header

def read_vff_data(filename, header, verbose=True):
    '''
    This function reads the data of a VFF file and returns a 3D numpy array.
    Don't call this on its own (use read_vff() instead).
    :param filename: The path to the VFF file (including the file name.vff)
    :param header: The header data of the VFF file (to get the dimensions and data type)
    :return: A 3D numpy array with the voxel data
    '''
    SYSTEM_MEMORY = 0.3 # in GB, change if larger arrays should be loaded into memory instead of using memmap

    xdim = int(header['size'].split()[0])
    ydim = int(header['size'].split()[1])
    if len(header['size'].split()) == 2:
        zdim = 1
    else:
        zdim = int(header['size'].split()[2])
    bits = int(header['bits'])

    # Assuming 8 or 16 bits per voxel
    data_type = np.dtype('>b') if bits == 8 else np.dtype('>h')
    data_size = xdim * ydim * zdim * int(bits/8)

    if data_size > SYSTEM_MEMORY * 1024**3:
        if verbose:
            print(f"Data size is {data_size/1024**3:.3f} GB, which is larger than the specified system memory allocation of {SYSTEM_MEMORY} GB. Loading data as memory-mapped file.")
        data = np.memmap(filename, dtype=data_type, mode='c', offset=os.path.getsize(filename)-data_size, shape=(zdim, ydim, xdim))

    else:
        data = np.fromfile(filename, dtype=data_type, offset=os.path.getsize(filename)-data_size).reshape(zdim, ydim, xdim)

    return data

def read_vff(filename, verbose=True):
    '''
    This function reads a VFF file and returns the header and the data.
    This is the main function that should be called.
    Set verbose to False to suppress the print output of the header.
    :return: header: A dictionary with the header data,
             data: A 3D numpy array with the voxel data (z, y, x)
    '''

    header = read_vff_header(filename, verbose=verbose)
    data = read_vff_data(filename, header, verbose=verbose)

    if verbose:
        print("Data loaded successfully, data shape:", data.shape)

    return header, data

def write_vff(filename, header, data, verbose=True):
    """
    Write a VFF file from a header dict and 2D/3D NumPy array.

    Delegates to `reconstruction.ct_core.vff_io.write_vff`, which is the single
    writer for the project. This function USED to emit a minimal
    `size/bits/format/endian` header; GE / Amalytics rejects those files
    ("ncaa is missing"), which is what scripts/fix_vff_header.py existed to
    repair after the fact. Delegating means there is no longer any writer in
    the tree that can produce an unloadable file.

    :param filename: Path to output .vff file
    :param header: Dict with metadata keys (e.g., 'bits'); size is inferred from `data`
    :param data: 3D NumPy array shaped (z, y, x) or 2D array (y, x)
    """
    # Import lazily and path-tolerantly: this module is imported both as
    # `metric_calculators.helper_scripts.vff_io` (package context) and as a
    # bare `vff_io` after its directory is put on sys.path (see
    # inr_pipeline/fdk_reference.py, ct_core/tiff_converter.py), so a package-
    # relative import would fail in the latter case.
    import sys
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from reconstruction.ct_core.vff_io import write_vff as _write_vff

    return _write_vff(filename, header, data, verbose=verbose)


class VFFDataset:
    def __init__(self, folder: str, xml_file: str, save_headers: bool = False, tensor_projections: bool = False,
                 paths_str: str = "acq*", exclude_pred_paths: bool = False, projection_spacing = None,
                 sub_scan='-00-'):
        import xmltodict
        import torch

        self.folder = folder
        self.paths = sorted(Path(folder).glob(paths_str))
        if exclude_pred_paths is True:
            self.paths = [p for p in self.paths if not p.name.endswith("_pred.vff")]

        # Apply sub-scan filter only to acquisition files (not sequential proj-* files)
        if sub_scan:
            self.paths = [p for p in self.paths
                          if p.name.startswith('proj-') or sub_scan in str(p)]

        with open(xml_file, 'r') as f:
            header = xmltodict.parse(f.read())
        sp = header['Series']['SeriesParams']
        self.num_projections = len(self.paths) #int(sp['ViewCount'])
        if projection_spacing is not None:
            self.imaging_angle = projection_spacing * self.num_projections
            print(f"Using custom projection spacing: {projection_spacing} degrees per projection, total angle: {self.imaging_angle:.2f} degrees")
        else:
            self.imaging_angle = float(sp['IncrementAngle']) * self.num_projections
            print(f"Using default projection spacing: {sp['IncrementAngle']} degrees per projection, total angle: {self.imaging_angle:.2f} degrees")

        self.starting_angle_offset = float(header['Series']['AngleOffset']) + 120

        # Read first frame for shape/dtype
        hdr, data = self._read_vff(self.paths[0])
        data = data.squeeze(0)
        self.det_rows, self.det_cols = data.shape

        # Create memmap for projections
        shape = (len(self.paths), self.det_rows, self.det_cols)
        memmap_path = os.path.join(folder, 'detector_values.dat')
        self.projections = np.memmap(memmap_path,
                                     dtype=data.dtype,
                                     mode='w+',
                                     shape=shape)
        self.projection_angles = np.zeros(len(self.paths), dtype=np.float32)

        if save_headers:
            self.headers = []

        # Fill memmap and angles
        for idx, p in enumerate(self.paths):
            hdr, dat = self._read_vff(p)
            arr = dat.squeeze(0)
            # Reverse byteswap to original order before writing
            original = arr.byteswap().view(arr.dtype.newbyteorder())
            self.projections[idx] = original
            self.projection_angles[idx] = float(hdr['gantryPosition'])

            if save_headers:
                self.headers.append(hdr)

        # Convert memmap to native byte order so torch can load it
        if self.projections.dtype.byteorder not in ('=', '<'):  # not native little endian
            # numpy 2.0 dropped ndarray.newbyteorder(), use view with new dtype
            new_dtype = self.projections.dtype.newbyteorder()
            self.projections = self.projections.byteswap().view(new_dtype)

        # Convert projections to float32 tensor if requested
        if tensor_projections:
            self.projections = torch.from_numpy(np.asarray(self.projections, dtype=np.float32))

        # Prepare angles in radians
        angles = torch.linspace(
            self.starting_angle_offset,
            self.starting_angle_offset + self.imaging_angle,
            steps=self.num_projections,
            dtype=torch.float32
        ) % 360
        self.angles_rad = torch.deg2rad(angles)

    def _read_vff(self, path: Path):
        # Use same header parsing as standalone functions
        header = {}
        with open(path, 'rb') as f:
            raw = f.read(1000).decode('latin-1', errors='ignore')
        try:
            header_content, _ = raw.split('\f', 1)
        except ValueError:
            header_content = raw
        for line in header_content.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                header[key.strip()] = value.strip().rstrip(',;')

        # Determine dtype from header
        bits = int(header.get('bits', '16'))
        dtype = np.dtype('>b' if bits == 8 else '>h')
        dims = [int(x) for x in header['size'].split()]
        if len(dims) == 2:
            z, y, x = 1, dims[1], dims[0]
        else:
            z, y, x = dims[2], dims[1], dims[0]
        total_bytes = x * y * z * bits // 8
        offset = os.path.getsize(path) - total_bytes
        data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(z, y, x))
        return header, data


if __name__ == '__main__':
    read_vff(filename = "/Users/falk/Downloads/Shelley phantom full scan 75um 16ms.vff", verbose=True)
