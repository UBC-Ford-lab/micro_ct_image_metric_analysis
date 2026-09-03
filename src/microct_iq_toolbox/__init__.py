"""Image quality metrics for reconstructed CT volumes.

MTF (slanted edge), NPS, NEQ, TTF (circular inserts) and the detectability
index d' following AAPM Report 233 and ISO 12233. Every calculator takes a
``(z, y, x)`` NumPy array and a pixel size in mm; nothing here depends on a
particular scanner.

High-level API (one function per metric, one result type)::

    import microct_iq_toolbox as iq
    r = iq.mtf(volume, [y1, y2, x1, x2], pixel_size=0.075, edge_angle=5.4)
    r.summary['mtf50']; r.x; r.y; r.save('results/mtf')

Regions in millimetres, measured once and reused on any grid::

    rois = iq.PhantomROIs.load('rois.json')          # from ct-find-rois
    frame = iq.Frame.for_image(volume, pixel_size=0.075)
    metrics = iq.run_metrics(volume, frame, rois, out_dir='results')

The raw calculators (``get_MTF`` and friends) remain available and return
their two-sided frequency axes unchanged.
"""

__version__ = "0.2.0"

from . import mtf_calculator
from . import nps_calculator
from . import neq_calculator
from . import ttf_calculator
from . import d_prime_calculator
from . import all_metrics_calculator
from .mtf_calculator import get_MTF
from .nps_calculator import get_NPS
from .neq_calculator import get_NEQ, neq_from_curves
from .ttf_calculator import get_TTF, DEFAULT_CNR_THRESHOLD
from .d_prime_calculator import (get_d_prime, get_d_prime_npw, get_d_prime_npwe,
                                 create_circular_task_function, ROSE_THRESHOLD)
from .all_metrics_calculator import run_all_metrics
from .helpers.io_utils import load_image_data
from .helpers.vff_io import read_vff, write_vff
from .helpers.radial import radial_profile
from .results import MetricResult, crossing, nps_summary
from .rois import Frame, PhantomROIs, index_of_mm, mm_of_index, nearest_index
from .volumes import load_volume
from .api import (mtf, nps, neq, ttf, detectability, run_metrics, save_metrics,
                  load_metrics, summary_table)
from .roi_fit import fit_rois, verification_figure, seed_template, ROIFitError
from .report import iq_figure, run_report
from .demo import synthetic_phantom, run_demo

__all__ = [
    "__version__",
    # high-level API
    "mtf", "nps", "neq", "ttf", "detectability", "run_metrics", "save_metrics",
    "load_metrics", "summary_table", "MetricResult", "crossing", "nps_summary",
    # regions in mm
    "Frame", "PhantomROIs", "index_of_mm", "mm_of_index", "nearest_index",
    "fit_rois", "verification_figure", "seed_template", "ROIFitError",
    # I/O, report, demo
    "load_volume", "load_image_data", "read_vff", "write_vff", "iq_figure", "run_report",
    "synthetic_phantom", "run_demo", "radial_profile",
    # raw calculators
    "mtf_calculator", "nps_calculator", "neq_calculator", "ttf_calculator",
    "d_prime_calculator", "all_metrics_calculator",
    "get_MTF", "get_NPS", "get_NEQ", "neq_from_curves", "get_TTF",
    "get_d_prime", "get_d_prime_npw", "get_d_prime_npwe",
    "create_circular_task_function", "run_all_metrics",
    "DEFAULT_CNR_THRESHOLD", "ROSE_THRESHOLD",
]
