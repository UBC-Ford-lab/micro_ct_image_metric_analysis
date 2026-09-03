# Micro-CT Image Quality Toolbox

[![PyPI](https://img.shields.io/pypi/v/micro_ct_image_quality_toolbox.svg)](https://pypi.org/project/micro-ct-image-quality-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Image quality metrics for reconstructed CT volumes: MTF, NPS, NEQ, TTF and
the detectability index d'. Point it at a phantom scan and get the curves,
the headline numbers and a figure back.

![MTF, NPS and detectability index of a micro-CT phantom reconstruction](https://raw.githubusercontent.com/UBC-Ford-lab/micro_ct_image_quality_toolbox/master/assets/scan_1988_vendor_fdk_iq.png)

*What `ct-report` draws: the MTF with its 50 % and 10 % crossings, the noise
power spectrum with its peak frequency and pixel noise, and d' against disc
diameter with the smallest disc that reaches the Rose criterion.*

| Metric | Method | Phantom feature |
|---|---|---|
| **MTF** | Slanted edge, 4x oversampled ERF, ISO 12233 | A straight edge tilted 2 to 10 degrees |
| **NPS** | 2D power spectra of detrended ROIs, radially averaged | A uniform region |
| **NEQ** | MTF² / NPS on a shared frequency axis | The two above |
| **TTF** | Radial ERF of circular inserts, per material, with a CNR floor | Cylindrical inserts |
| **d'** | Non-prewhitening observer, disc tasks of several sizes, AAPM TG-233 | From the MTF and NPS curves |

Nothing here depends on a scanner. Volumes are `(z, y, x)` arrays from
`.npy`, `.npz` or GE `.vff` files, the pixel size is in mm, and every
region is specified in millimetres so the same table works on any grid.

## Install

```bash
pip install micro_ct_image_quality_toolbox
# or the development version
pip install git+https://github.com/UBC-Ford-lab/micro_ct_image_quality_toolbox.git
```

Python 3.10+, NumPy, SciPy, Matplotlib. The import name is `microct_iq_toolbox`.
`pip install "micro_ct_image_quality_toolbox[photutils]"` adds photutils for
exact circle/pixel overlap in the radial averages; the built-in NumPy
version agrees with it to better than 1 %.

## Try it in ten seconds

```bash
ct-demo --out demo/
```

builds a synthetic phantom with a known blur and noise level, runs the whole
pipeline on it and prints the measured numbers next to the true ones. Look
in `demo/` to see every file the real workflow produces.

## Two commands for a real phantom

**1. Measure where things are, once per phantom.**

```bash
ct-find-rois --template seed.json        # edit the approximate positions
ct-find-rois reference.vff --seed seed.json --out rois.json
```

The seed holds rough positions in mm: which z range holds the edge slab,
roughly where the inserts sit, the ring the noise ROIs go on. The fitter
finds the slab, fits the edge angle to a hundredth of a degree, refines
every insert centre with a circle fit, checks each noise ROI for structure,
and writes `rois.json` plus `rois.png`. Look at the figure: a region that
has drifted onto the wrong feature still produces numbers.

**2. Measure any volume of that phantom.**

```bash
ct-report volume.vff --rois rois.json --out results/
```

```
  MTF50                                 0.939 lp/mm
  MTF10                                 2.851 lp/mm
  noise sigma                          362.4 HU
  NPS peak frequency                    1.441 lp/mm
  TTF50 (mean of usable inserts)        0.912 lp/mm
  detectable disc at Rose d'            0.908 mm
```

`results/` gets one `.npz` + `.json` per metric, a combined `metrics.json`,
and `iq_report.png` / `.pdf`. Because the ROI table is in millimetres,
`ct-report` on a 75 µm and a 100 µm reconstruction of the same scan measures
the same piece of phantom. The voxel size comes from the VFF header or a
`.json` sidecar; pass `--pixel-size` for a bare `.npy`.

## One metric at a time

Each metric is also its own command, taking pixel-index regions on the
volume you pass in. Every command writes `<metric>.npz` (the curve) and
`<metric>.json` (the numbers), and the later ones read what the earlier ones
wrote:

```bash
ct-mtf --input edge.npy --crop_indices 270,664,522,640 --pixel_size 0.075 --edge_angle 5.4 --output_dir r/
ct-nps --input flat.npy --roi_bounds "178,294,510,626;258,374,750,866" --pixel_size 0.075 --output_dir r/
ct-neq --mtf r/mtf.npz --nps r/nps.npz --output_dir r/
ct-dprime npw --mtf r/mtf.npz --nps r/nps.npz --contrast 100 --output_dir r/
ct-ttf --input inserts.npy --centre_pixels "1335,2141;765,1914" --radius 120 --materials "Teflon,Fat" --pixel_size 0.075
ct-metrics --config config.json          # any subset of the above from one JSON file
```

`crop_indices` is `y1,y2,x1,x2` around the edge (height twice the width),
`roi_bounds` a `;`-separated list of square ROIs, `centre_pixels` `y,x` per
insert, `--slices` the z range (`10:160` or `0:30,140:182`). `--help` on
any command lists every flag; [`example_config.json`](example_config.json)
shows the `ct-metrics` layout.

## From Python

```python
import numpy as np
import microct_iq_toolbox as iq

vol = np.load("edge.npy")                                   # (z, y, x)
r = iq.mtf(vol, [270, 664, 522, 640], pixel_size=0.075, edge_angle=5.4)
r.x, r.y                                                    # frequency (mm⁻¹), MTF
r.summary["mtf50"], r.summary["mtf10"]
r.save("results/mtf")                                       # .npz + .json

n = iq.nps(np.load("flat.npy"), [[178, 294, 510, 626]], pixel_size=0.075)
d = iq.detectability(r, n, contrast_hu=100)
d.summary["detectable_size_mm"]

# or everything from an ROI table in mm
rois = iq.PhantomROIs.load("rois.json")
frame = iq.Frame.for_image(vol, pixel_size=0.075)
metrics = iq.run_metrics(vol, frame, rois, out_dir="results")
iq.iq_figure(metrics, "results/iq_report")
```

Every function returns a `MetricResult`: a non-negative frequency axis,
the curve (one row per insert for the TTF), `summary` numbers and `meta`
describing what was measured. The raw calculators (`iq.get_MTF`, ...) are
still there for anyone who wants the two-sided spectra.

## Things worth knowing

- **The edge angle is an input.** `mtf()` trusts `edge_angle` and a wrong
  value biases the curve; that is what `ct-find-rois` measures. Negative
  angles mean the edge leans the other way.
- **TTF carries a CNR floor.** Inserts below `DEFAULT_CNR_THRESHOLD` (5)
  get a NaN TTF50 and stay out of the average, so a low-contrast insert
  cannot masquerade as the sharpest one.
- **d' is reported two ways.** d' at each requested disc size, and the
  diameter at which the dense curve crosses the Rose threshold of 3. The
  crossing is `nan` when it does not exist, never an extrapolated number.
- **Nothing is skipped silently.** A region the volume cannot hold, or an
  insert that fails the CNR floor, is listed under `skipped` with a reason.

## Layout

```
src/microct_iq_toolbox/
  api.py           mtf / nps / neq / ttf / detectability / run_metrics
  results.py       MetricResult, crossing, nps_summary
  rois.py          Frame (mm <-> voxels), PhantomROIs
  roi_fit.py       ct-find-rois
  report.py        ct-report and the three-panel figure
  demo.py          ct-demo and the synthetic phantom
  volumes.py       load_volume (.npy / .npz / .vff + sidecar)
  *_calculator.py  the calculators and their ct-* commands
  helpers/         VFF I/O, CLI parsers, LSF processing, radial averaging
```

## Development

```bash
git clone https://github.com/UBC-Ford-lab/micro_ct_image_quality_toolbox.git
cd micro_ct_image_quality_toolbox
pip install -e ".[dev]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).
