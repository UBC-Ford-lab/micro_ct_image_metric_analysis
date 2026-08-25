# Micro-CT Image Metric Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CT image quality metric calculators for micro-CT reconstructed volumes: Modulation Transfer Function (MTF), Noise Power Spectrum (NPS), Noise-Equivalent Quanta (NEQ), Task Transfer Function (TTF), and Detectability Index (d').

## Quick Start

```bash
# Install
git clone https://github.com/UBC-Ford-lab/micro_ct_image_metric_analysis.git
cd micro_ct_image_metric_analysis
pip install -e .

# Run MTF calculation on a .npy file
ct-mtf --input phantom.npy --crop_indices 270,664,522,640 --pixel_size 0.085 --edge_angle 5.4

# Run all metrics from a config file
ct-metrics --config example_config.json
```

## Overview

This package provides a complete suite of spatial-frequency-domain image quality metrics for evaluating CT reconstructions, following standards from AAPM Report 233 and ISO 12233. Each metric can be computed independently or all together via the `all_metrics_calculator` module.

### Metrics

| Metric | Description | Input |
|--------|-------------|-------|
| **MTF** | Modulation Transfer Function -- spatial resolution via slanted-edge method | Slanted edge phantom slices |
| **NPS** | Noise Power Spectrum -- noise texture characterization via radial averaging of 2D power spectra | Homogeneous phantom slices with multiple ROIs |
| **NEQ** | Noise-Equivalent Quanta -- combines MTF and NPS into a single signal-to-noise metric | MTF + NPS inputs |
| **TTF** | Task Transfer Function -- contrast-dependent resolution via circular edge inserts | Circular insert phantom slices |
| **d'** | Detectability Index -- task-based detectability combining TTF, NPS, and a task function | TTF + NPS inputs + task function |
| **d' (NPW)** | NPW-observer Detectability Index -- frequency-domain d' for disc objects of multiple sizes, from pre-computed MTF and NPS (AAPM TG-233) | MTF + NPS frequency curves |

## Package Structure

```
metric_calculators/              # Package root (also repo root)
├── __init__.py
├── all_metrics_calculator.py   # Run all metrics in sequence
├── mtf_calculator.py           # Modulation Transfer Function
├── nps_calculator.py           # Noise Power Spectrum
├── neq_calculator.py           # Noise-Equivalent Quanta
├── ttf_calculator.py           # Task Transfer Function
├── d_prime_calculator.py       # Detectability Index (d')
├── helper_scripts/
│   ├── __init__.py
│   ├── io_utils.py             # Shared I/O utilities and CLI argument parsers
│   ├── vff_io.py               # VFF file reader/writer (bundled from muPIU-Net)
│   ├── lsf_processing.py       # Line Spread Function processing (detrend, window, center)
│   ├── all_models_comparison_plot.py      # Multi-model MTF/NPS/NEQ comparison figure
│   └── plot_reconstruction_comparison.py  # Visual slice comparison figure
├── example_config.json         # Example config for all_metrics_calculator
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

### As a standalone package

```bash
git clone https://github.com/UBC-Ford-lab/micro_ct_image_metric_analysis.git
cd micro_ct_image_metric_analysis
pip install -e .
```

### As a git submodule

When used inside a parent project (e.g., [muPIU-Net](https://github.com/UBC-Ford-lab/muPIU-Net-microCT-sinogram-infilling-network)):

```bash
git submodule add https://github.com/UBC-Ford-lab/micro_ct_image_metric_analysis.git metric_calculators
```

No `pip install` needed -- just ensure the parent repo root is on `sys.path` (e.g., via `pip install -e .` on the parent).

## Input Data Format

Image data should be 3D numpy arrays with shape `(z, y, x)` where `z` is the slice axis. Supported formats:

- **`.npy`** (recommended): Save with `np.save('volume.npy', image_array)`
- **`.npz`**: The first array in the archive is used. Save with `np.savez('volume.npz', image_array)`
- **`.vff`**: Natively supported via the bundled `vff_io` reader (no external dependencies needed)

To convert from other formats:

```python
import numpy as np

# From TIFF stack
from skimage import io
volume = io.imread('stack.tif')  # shape: (z, y, x)
np.save('volume.npy', volume)

# From DICOM
import pydicom, glob
files = sorted(glob.glob('dicom_dir/*.dcm'))
volume = np.stack([pydicom.dcmread(f).pixel_array for f in files])
np.save('volume.npy', volume)
```

## CLI Reference

All commands are available after `pip install -e .`. Use `--help` on any command for full details.

### ct-mtf -- Modulation Transfer Function

```bash
ct-mtf --input volume.npy \
       --crop_indices 270,664,522,640 \
       --pixel_size 0.085 \
       --edge_angle 5.4 \
       --slices 228:229 \
       --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to image file (.npy, .npz, or .vff) |
| `--crop_indices` | Yes | Crop region as y1,y2,x1,x2 |
| `--pixel_size` | No | Pixel size in mm (default: 0.05) |
| `--edge_angle` | No | Slanted edge angle in degrees (default: 5.0) |
| `--slices` | No | Slice selection (e.g. "10:160" or "0:30,140:182") |
| `--high_to_low` | No | Edge goes from high to low (default) |
| `--low_to_high` | No | Edge goes from low to high |
| `--relative` | No | Calculate relative MTF instead of absolute |
| `--output_dir` | No | Output directory (default: ./results) |
| `--no_plot` | No | Disable plot generation |
| `--show` | No | Display plots interactively |

### ct-nps -- Noise Power Spectrum

```bash
ct-nps --input volume.npy \
       --roi_bounds "178,294,510,626;258,374,750,866;310,426,328,444" \
       --pixel_size 0.085 \
       --slices "0:30,140:182" \
       --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to image file |
| `--roi_bounds` | Yes | ROI bounds as semicolon-separated y1,y2,x1,x2 groups |
| `--pixel_size` | Yes | Pixel size in mm |
| `--slices` | No | Slice selection |
| `--filter_low_freq` | No | Filter low frequency components |
| `--output_dir` | No | Output directory (default: ./results) |
| `--no_plot` | No | Disable plot generation |
| `--show` | No | Display plots interactively |

### ct-neq -- Noise-Equivalent Quanta

```bash
ct-neq --input_mtf mtf_volume.npy \
       --input_nps nps_volume.npy \
       --crop_indices 270,664,522,640 \
       --roi_bounds "178,294,510,626;258,374,750,866" \
       --pixel_size 0.085 \
       --slices_mtf 228:229 \
       --slices_nps "0:30,140:182" \
       --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_mtf` | Yes | Path to MTF image file |
| `--input_nps` | Yes | Path to NPS image file |
| `--crop_indices` | Yes | MTF crop region as y1,y2,x1,x2 |
| `--roi_bounds` | Yes | NPS ROI bounds |
| `--pixel_size` | Yes | Pixel size in mm |
| `--slices_mtf` | No | Slice selection for MTF data |
| `--slices_nps` | No | Slice selection for NPS data |
| `--low_to_high` | No | MTF edge goes from low to high |
| `--edge_angle` | No | MTF edge angle in degrees (default: 5.5). **Must match what you pass to `ct-mtf`**, or the NEQ is built on a different MTF than the one you reported. |
| `--output_dir` | No | Output directory (default: ./results) |
| `--no_plot` | No | Disable plot generation |
| `--show` | No | Display plots interactively |

### ct-ttf -- Task Transfer Function

```bash
ct-ttf --input volume.npy \
       --centre_pixels "1335,2141;765,1914;528,1348" \
       --radius 120 \
       --materials "Teflon,HD POLY,Fat" \
       --pixel_size 0.025 \
       --slices 30:130 \
       --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to image file |
| `--centre_pixels` | Yes | Centre pixels as semicolon-separated y,x pairs |
| `--radius` | Yes | Radius of circular edges in pixels |
| `--materials` | No | Comma-separated material names |
| `--slices` | No | Slice selection |
| `--pixel_size` | No | Pixel size in mm (default: 0.05) |
| `--relative` | No | Calculate relative TTF |
| `--output_dir` | No | Output directory (default: ./results) |
| `--no_plot` | No | Disable plot generation |
| `--show` | No | Display plots interactively |

### ct-dprime -- Detectability Index (d')

Two modes are available: **TTF-based** (original, from circular phantom inserts) and **NPW** (frequency-domain, from pre-computed MTF/NPS curves).

#### TTF mode (circular inserts)

```bash
ct-dprime ttf --input_ttf ttf_volume.npy \
              --input_nps nps_volume.npy \
              --centre_pixels "686,398;418,132;153,398;229,586" \
              --radius 30 \
              --materials "SB3,Teflon,Fat,Tissue" \
              --roi_bounds "194,400,175,381;440,646,175,381" \
              --task_material Fat \
              --task_contrast -160 \
              --task_object_size 0.2 \
              --pixel_size 0.075 \
              --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_ttf` | Yes | Path to TTF image file |
| `--input_nps` | Yes | Path to NPS image file |
| `--centre_pixels` | Yes | Centre pixels as semicolon-separated y,x pairs |
| `--radius` | Yes | Radius of circular edges in pixels |
| `--materials` | Yes | Comma-separated material names |
| `--roi_bounds` | Yes | NPS ROI bounds |
| `--task_material` | Yes | Task function material name |
| `--task_contrast` | No | Task function contrast in HU (default: -160) |
| `--task_object_size` | No | Task function object radius in mm (default: 0.2) |
| `--slices_ttf` | No | Slice selection for TTF data |
| `--slices_nps` | No | Slice selection for NPS data |
| `--pixel_size` | No | Pixel size in mm (default: 0.05) |
| `--output_dir` | No | Output directory (default: ./results) |
| `--no_plot` | No | Disable plot generation |
| `--show` | No | Display plots interactively |
| `--quiet` | No | Suppress console output |

#### NPW mode (AAPM TG-233, from MTF/NPS curves)

Computes the non-prewhitening matched-filter observer d' for multiple uniform disc sizes
using the analytical Bessel-function task function (Samei et al., 2019).

```bash
ct-dprime npw --mtf_npz metrics_mtf.npz \
              --nps_npz metrics_nps.npz \
              --disc_diameters "0.15,0.5,1.0,3.0" \
              --contrast 100 \
              --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--mtf_npz` | Yes | .npz file with `mtf_freq` and `mtf` arrays |
| `--nps_npz` | Yes | .npz file with `nps_freq` and `nps` arrays |
| `--disc_diameters` | No | Comma-separated disc diameters in mm (default: 0.15,0.5,1.0,3.0) |
| `--contrast` | No | Contrast in HU (default: 100, soft tissue) |
| `--rose_threshold` | No | Threshold at which the detectable disc size is read off (default: 3, the Rose criterion) |
| `--output_dir` | No | Output directory (default: ./results) |
| `--no_plot` | No | Disable plot generation |
| `--show` | No | Display plots interactively |

Default disc diameters for micro-CT: **0.15 mm** (finest resolvable detail), **0.5 mm** (bronchioles), **1.0 mm** (small lesions), **3.0 mm** (larger structures).
A feature is considered detectable when d' ≥ 3 (Rose criterion).

**The headline number is a size, not an index.** Alongside d' at each requested
diameter, d' is evaluated on a dense diameter grid and the run prints (and
plots) the **detectable disc size**: the diameter at which the curve crosses
the threshold, in mm. d' at a fixed disc size spans orders of magnitude between
systems and is hard to read comparatively; the diameter at a fixed d' does not.
The crossing is `nan` when it falls outside the sampled grid -- either every
sampled disc is already above the threshold or none of them reach it -- rather
than being extrapolated.

### ct-metrics -- All Metrics at Once

```bash
ct-metrics --config example_config.json --output_dir ./results
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to JSON configuration file |
| `--output_dir` | No | Override output directory from config |
| `--show` | No | Display plots interactively |

## Configuration File

The `ct-metrics` command reads all parameters from a JSON config file. See [`example_config.json`](example_config.json) for the full template. The config has a top-level `pixel_size` and `output_dir`, plus sections for each metric (`nps`, `mtf`, `ttf`, `d_prime`). NEQ is computed automatically from the MTF and NPS data.

## Usage

### Python API

#### MTF (Modulation Transfer Function)

```python
from metric_calculators import mtf_calculator

# image_data: 3D numpy array (z, y, x) of slanted-edge phantom slices
# crop_indices: [y1, y2, x1, x2] pixel coordinates of the edge region
mtf_freq, mtf = mtf_calculator.get_MTF(
    image_data,
    crop_indices=[270, 664, 522, 640],
    find_absolute_MTF=True,
    pixel_size=0.085,           # mm
    edge_angle=5.4,             # degrees
    high_to_low=True,
    plot_results=True,
    target_directory='results/'
)
```

#### NPS (Noise Power Spectrum)

```python
from metric_calculators import nps_calculator
import numpy as np

# image_data: 3D numpy array (z, y, x) of homogeneous phantom slices
# ROI_bounds: Nx4 array of [y1, y2, x1, x2] for each ROI
ROI_bounds = np.array([
    [178, 294, 510, 626],
    [258, 374, 750, 866],
    # ... more ROIs
])

nps_freq, nps = nps_calculator.get_NPS(
    image_data,
    ROI_bounds,
    pixel_size=0.085,
    plot_results=True,
    target_directory='results/'
)
```

#### NEQ (Noise-Equivalent Quanta)

```python
from metric_calculators import neq_calculator

neq_freq, neq = neq_calculator.get_NEQ(
    image_data_MTF,       # slanted-edge slices
    image_data_NPS,       # homogeneous slices
    crop_indices_MTF,
    ROI_bounds_NPS,
    pixel_size=0.085,
    plot_results=True,
    target_directory='results/',
    edge_angle=5.5,       # must match the angle you gave get_MTF
)
```

If you have already measured the MTF and the NPS -- to report them in the same
table -- build the NEQ from those curves instead. `get_NEQ` measures both
itself, so calling it as well costs a second measurement AND can hand you an
NEQ built on a different MTF than the one you published:

```python
neq_freq, neq = neq_calculator.neq_from_curves(mtf_freq, mtf, nps_freq, nps)
```

Only the overlap of the two frequency axes is used; neither curve is
extrapolated past its own Nyquist.

#### TTF (Task Transfer Function)

```python
from metric_calculators import ttf_calculator

# centre_pixels: list of [y, x] centres for each circular insert
# radius: radius of the circular inserts in pixels
ttf_freq, ttf_array, cnr_array = ttf_calculator.get_TTF(
    image_data,
    centre_pixels=[[1335, 2141], [765, 1914], ...],
    radius=120,
    materials=['Teflon', 'HD POLY', 'Fat', ...],
    pixel_size=0.025,
    plot_results=True,
    target_directory='results/'
)
```

#### Detectability Index (d') -- TTF-based

```python
from metric_calculators import d_prime_calculator

# Create a circular task function
task_function = d_prime_calculator.create_circular_task_function(
    contrast=-160,          # HU
    radius=0.2,             # mm
    pixel_size=0.025,
    image_dimension=1000
)

d_prime = d_prime_calculator.get_d_prime(
    image_data_TTF, centre_pixels, radius, materials,
    image_data_NPS, ROI_bounds,
    task_function_data=task_function,
    task_function_material='Fat',
    task_function_object_size=0.2,
    pixel_size=0.025,
    plot_results=True,
    target_directory='results/'
)
```

#### Detectability Index (d') -- NPW observer (AAPM TG-233)

Compute d' for multiple disc sizes from pre-computed MTF and NPS curves.
Uses the non-prewhitening matched-filter (NPW) observer with analytical
Bessel-function task functions for uniform discs.

```python
from metric_calculators import d_prime_calculator

# From pre-computed MTF and NPS (e.g., from mtf_calculator / nps_calculator)
result = d_prime_calculator.get_d_prime_npw(
    mtf_freq, mtf,                              # 1D arrays from get_MTF()
    nps_freq, nps,                              # 1D arrays from get_NPS()
    disc_diameters_mm=[0.15, 0.5, 1.0, 3.0],   # micro-CT task sizes
    contrast_hu=100.0,                          # soft-tissue contrast
    normalize_mtf=True,
    rose_threshold=3.0,                         # Rose criterion
    plot_results=True,
    target_directory='results/',
)

# result['d_prime']                  -- array of d' values per disc diameter
# result['disc_diameters_mm']        -- corresponding diameters
# result['contrast_hu']              -- contrast used
# result['curve_diameters_mm']       -- dense grid the crossing is read off
# result['curve_d_prime']            -- d' on that grid
# result['diameter_at_threshold_mm'] -- DETECTABLE DISC SIZE in mm, or nan
# result['rose_threshold']           -- the threshold used
```

`d_prime_calculator.diameter_at_threshold(diameters, d_prime, threshold)`
locates the crossing on any pair of arrays, so a caller that already has a
d' curve does not have to recompute one. `get_d_prime_npwe` returns the same
keys with a human contrast-sensitivity filter applied.

#### All Metrics at Once

```python
# Via CLI:
#   ct-metrics --config example_config.json

# Via Python:
from metric_calculators.all_metrics_calculator import run_all_metrics
import json

with open('example_config.json') as f:
    config = json.load(f)
run_all_metrics(config)
```

### CLI -- Comparison Plots

```bash
# MTF/NPS/NEQ comparison across all models
python -m metric_calculators.helper_scripts.all_models_comparison_plot \
    --scan_name Scan_1681 \
    --results_dir data/results \
    --output_dir ./figures

# Visual slice comparison
python -m metric_calculators.helper_scripts.plot_reconstruction_comparison \
    --scan_name Scan_1681 \
    --slice_idx 150 \
    --output_dir ./figures
```

## Conventions worth knowing

**The returned frequency axis is two-sided.** `get_MTF` and `get_TTF` both
return `fftshift(fftfreq(...))`, which runs from -Nyquist to +Nyquist with DC
in the middle. `freq[0]` is therefore NOT zero frequency, and normalising a
curve by `curve[0]` or reading a crossing off it from the start of the array
gives a number from the negative half. Take `freq >= 0` first.

**The two reach different Nyquists.** `get_MTF` block-averages its 4x
oversampled ESF back to pixel spacing before the transform, so its axis stops
at the pixel Nyquist `1/(2*pixel_size)`. `get_TTF` keeps its radial profile at
`sampling_pixel_increment` (1/4) of a pixel, so its axis reaches four times
further. The VALUES are comparable; the extents are not. The block average
also costs the MTF about 2% at a typical MTF50 (it is a 1-pixel boxcar applied
on top of the sampling that already happened) in exchange for lower noise.

**`edge_angle` may be negative.** The sign says which way the edge leans and is
handled. What matters is the magnitude: below roughly 1.5 degrees the edge does
not cross a whole pixel over the ROI height and the projected-bin method has
nothing to supersample.

**Regions are pixel indices.** `crop_indices`, `roi_bounds`, `centre_pixels`,
`radius` and `slices` are all in pixels on the grid of the volume you passed.
Comparing reconstructions on different grids means converting a region
specified once in millimetres, per volume -- the indices themselves are not
portable.

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib
- photutils

VFF file support is included out of the box. The `VFFDataset` class (for raw projection loading) additionally requires `xmltodict` and `torch`.

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wiegmann2026metrics,
  author = {Wiegmann, Falk},
  title = {Micro-CT Image Metric Analysis},
  year = {2026},
  url = {https://github.com/UBC-Ford-lab/micro_ct_image_metric_analysis}
}
```
