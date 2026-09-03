# Micro-CT Image Quality Toolbox

[![PyPI](https://img.shields.io/pypi/v/micro-ct-image-quality-toolbox)](https://pypi.org/project/micro-ct-image-quality-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

MTF, NPS, NEQ, TTF and detectability index (d') for reconstructed CT
volumes, following AAPM Report 233 and ISO 12233.

![MTF, NPS and d' of a micro-CT phantom reconstruction](https://raw.githubusercontent.com/UBC-Ford-lab/micro_ct_image_quality_toolbox/master/assets/scan_1988_vendor_fdk_iq.png)

| Metric | Method | Needs |
|---|---|---|
| MTF | Slanted edge (ISO 12233) | An edge tilted 2 to 10° |
| NPS | Radially averaged 2D power spectra of ROIs | A uniform region |
| NEQ | MTF² / NPS | Both of the above |
| TTF | Radial edge profile of circular inserts | Cylindrical inserts |
| d' | NPW observer, disc tasks (AAPM TG-233) | MTF and NPS |

Volumes are `(z, y, x)` arrays from `.npy`, `.npz` or GE `.vff` files.
Regions are given in millimetres, so one ROI table works at any voxel size.

## Install

```bash
pip install micro_ct_image_quality_toolbox
```

Python 3.10+. Import name `microct_iq_toolbox`. Add `[photutils]` for
exact circle/pixel overlap in the radial averages (the NumPy default agrees
to <1 %).

## Usage

```bash
ct-demo --out demo/                                        # synthetic phantom, full pipeline, ~10 s

ct-find-rois --template seed.json                          # edit the rough positions
ct-find-rois reference.vff --seed seed.json --out rois.json
ct-report volume.vff --rois rois.json --out results/
```

`ct-find-rois` measures the edge angle, insert centres and noise ROIs once
per phantom and writes `rois.json` with a verification figure. `ct-report`
runs every metric on a volume and writes the curves (`.npz` + `.json`),
`metrics.json` and the figure above. Voxel size is read from the VFF header
or a `.json` sidecar; use `--pixel-size` for a bare `.npy`.

Single metrics with pixel regions:

```bash
ct-mtf --input edge.npy --crop_indices 270,664,522,640 --pixel_size 0.075 --edge_angle 5.4 --output_dir r/
ct-nps --input flat.npy --roi_bounds "178,294,510,626;258,374,750,866" --pixel_size 0.075 --output_dir r/
ct-neq --mtf r/mtf.npz --nps r/nps.npz --output_dir r/
ct-dprime npw --mtf r/mtf.npz --nps r/nps.npz --contrast 100 --output_dir r/
ct-ttf --input inserts.npy --centre_pixels "1335,2141;765,1914" --radius 120 --pixel_size 0.075
ct-metrics --config config.json
```

From Python:

```python
import microct_iq_toolbox as iq

r = iq.mtf(vol, [270, 664, 522, 640], pixel_size=0.075, edge_angle=5.4)
r.x, r.y, r.summary["mtf50"]

rois = iq.PhantomROIs.load("rois.json")
frame = iq.Frame.for_image(vol, pixel_size=0.075)
metrics = iq.run_metrics(vol, frame, rois, out_dir="results")
```

Each function returns a `MetricResult` with the curve, a `summary` dict and
`save()`. The raw calculators (`iq.get_MTF`, ...) are still available.

Notes:

- The edge angle is an input; a wrong value biases the MTF.
- TTF inserts below CNR 5 are reported but excluded from the average.
- The detectable disc size is where d' crosses 3 (Rose). It is `nan` if
  the curve never gets there.
- Regions that do not fit the volume are listed under `skipped`, never
  clamped.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
