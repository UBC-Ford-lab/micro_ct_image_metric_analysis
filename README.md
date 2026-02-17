# Micro-CT Image Metric Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CT image quality metric calculators for micro-CT reconstructed volumes: Modulation Transfer Function (MTF), Noise Power Spectrum (NPS), Noise-Equivalent Quanta (NEQ), Task Transfer Function (TTF), and Detectability Index (d').

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

## Package Structure

```
metric_calculators/              # Package root (also repo root)
├── __init__.py
├── all_metrics_calculator.py    # Run all metrics in sequence
├── mtf_calculator.py            # Modulation Transfer Function
├── nps_calculator.py            # Noise Power Spectrum
├── neq_calculator.py            # Noise-Equivalent Quanta
├── ttf_calculator.py            # Task Transfer Function
├── d_prime_calculator.py        # Detectability Index (d')
├── helper_scripts/
│   ├── __init__.py
│   ├── lsf_processing.py       # Line Spread Function processing (detrend, window, center)
│   ├── all_models_comparison_plot.py      # Multi-model MTF/NPS/NEQ comparison figure
│   └── plot_reconstruction_comparison.py  # Visual slice comparison figure
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
    target_directory='results/'
)
```

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

#### Detectability Index (d')

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

#### All Metrics at Once

```python
# Run from parent project root:
python -m metric_calculators.all_metrics_calculator
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

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib
- photutils

When used as a submodule within muPIU-Net, the `reconstruction` package (for VFF I/O) is provided by the parent project.

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
