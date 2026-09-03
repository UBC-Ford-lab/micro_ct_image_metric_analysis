"""One result type for every metric, plus the curve summaries.

Every calculator hands back a frequency axis and a curve, but with three
different signatures and a two-sided (fftshifted) axis. ``MetricResult`` is
the one shape the high-level API and the commands use: a non-negative
frequency axis, the curve(s), the scalar summaries a table wants, and
``save``/``load`` to ``.npz`` + ``.json`` so a command's output can be read
by the next command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0 and removed in 2.3.
trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# --------------------------------------------------------------------------
# Curve summaries
# --------------------------------------------------------------------------

def crossing(freq, curve, level: float) -> float:
    """First frequency where a normalised curve falls through ``level``.

    Linear interpolation between the bracketing samples. NaN when the curve
    never gets there, which is a real answer: an MTF still above 10 % at
    Nyquist has no MTF10 to report, and extrapolating one would flatter the
    sharpest volume most.
    """
    f = np.asarray(freq, dtype=np.float64)
    c = np.asarray(curve, dtype=np.float64)
    order = np.argsort(f)
    f, c = f[order], c[order]
    if c.size == 0 or not np.isfinite(c[0]) or c[0] <= 0:
        return float('nan')
    c = c / c[0]
    below = np.where(c < level)[0]
    if below.size == 0:
        return float('nan')
    i = int(below[0])
    if i == 0:
        return float('nan')
    c0, c1 = c[i - 1], c[i]
    if c0 == c1:
        return float(f[i])
    return float(f[i - 1] + (c0 - level) / (c0 - c1) * (f[i] - f[i - 1]))


def nps_summary(freq, nps) -> dict:
    """Total noise power, its mean frequency, and where it peaks.

    The mean frequency is the first moment of the NPS: the one number that
    says whether the noise is fine-grained or blotchy.
    """
    f = np.asarray(freq, dtype=np.float64)
    p = np.asarray(nps, dtype=np.float64)
    good = np.isfinite(f) & np.isfinite(p) & (f >= 0)
    f, p = f[good], p[good]
    if f.size < 2 or p.sum() <= 0:
        return {'nps_total': float('nan'), 'nps_f_mean': float('nan'),
                'nps_f_peak': float('nan')}
    total = float(trapezoid(p, f))
    return {
        'nps_total': total,
        'nps_f_mean': float(trapezoid(p * f, f) / total) if total > 0 else float('nan'),
        'nps_f_peak': float(f[int(np.argmax(p))]),
    }


def positive_half(freq, *curves):
    """Drop the negative half of a two-sided (fftshifted) axis.

    Returns ``(freq, curve, ...)`` sorted by frequency with ``f >= 0`` only.
    2D curves are trimmed along their last axis.
    """
    f = np.asarray(freq, dtype=np.float64)
    keep = np.where(f >= 0)[0]
    keep = keep[np.argsort(f[keep])]
    out = [f[keep]]
    for c in curves:
        c = np.asarray(c, dtype=np.float64)
        out.append(c[..., keep])
    return tuple(out)


# --------------------------------------------------------------------------
# The result type
# --------------------------------------------------------------------------

@dataclass
class MetricResult:
    """A measured curve with its summary numbers.

    :param kind: ``'mtf'``, ``'nps'``, ``'neq'``, ``'ttf'`` or ``'d_prime'``.
    :param x: the abscissa, spatial frequency in mm^-1 (or disc diameter
        in mm for ``d_prime``), non-negative and increasing.
    :param y: the curve; shape ``(n,)`` or ``(n_series, n)`` for TTF.
    :param labels: one label per series when ``y`` is 2D (TTF materials).
    :param summary: scalar numbers (MTF50, noise sigma, detectable size, ...).
    :param meta: what was measured and how (pixel size, regions, settings).
    """

    kind: str
    x: np.ndarray
    y: np.ndarray
    labels: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # further arrays (e.g. the dense d' curve)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.labels = list(self.labels)

    # -- convenience --------------------------------------------------------
    @property
    def freq(self) -> np.ndarray:
        return self.x

    @property
    def values(self) -> np.ndarray:
        return self.y

    def series(self, label) -> np.ndarray:
        """One row of a multi-series result by label."""
        if self.y.ndim == 1:
            return self.y
        return self.y[self.labels.index(label)]

    def __getitem__(self, key):
        return self.summary[key]

    # -- persistence ----------------------------------------------------------
    def save(self, stem) -> tuple:
        """Write ``<stem>.npz`` (arrays) and ``<stem>.json`` (numbers).

        ``stem`` may carry an extension; it is replaced.
        """
        stem = Path(stem)
        if stem.suffix in ('.npz', '.json'):
            stem = stem.with_suffix('')
        stem.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'x': self.x, 'y': self.y, 'labels': np.asarray(self.labels, dtype=str)}
        for k, v in self.extra.items():
            arrays[f'extra_{k}'] = np.asarray(v)
        # legacy key names, so older scripts reading mtf_freq/mtf, nps_freq/nps keep working
        if self.kind in ('mtf', 'nps', 'neq') and self.y.ndim == 1:
            arrays[f'{self.kind}_freq'] = self.x
            arrays[self.kind] = self.y
        np.savez(stem.with_suffix('.npz'), **arrays)
        record = {'kind': self.kind, 'summary': _jsonable(self.summary),
                  'meta': _jsonable(self.meta), 'labels': self.labels}
        stem.with_suffix('.json').write_text(json.dumps(record, indent=2))
        return stem.with_suffix('.npz'), stem.with_suffix('.json')

    @classmethod
    def load(cls, stem) -> "MetricResult":
        """Read a result written by :meth:`save`, or a legacy ``.npz``."""
        stem = Path(stem)
        if stem.suffix in ('.npz', '.json'):
            stem = stem.with_suffix('')
        npz = np.load(stem.with_suffix('.npz'))
        js = stem.with_suffix('.json')
        record = json.loads(js.read_text()) if js.exists() else {}
        if 'x' in npz:
            x, y = npz['x'], npz['y']
            labels = [str(v) for v in npz['labels']] if 'labels' in npz else []
            kind = record.get('kind', 'mtf')
        else:  # legacy: mtf_freq/mtf or nps_freq/nps
            kind = next(k for k in ('mtf', 'nps', 'neq') if k in npz)
            x, y = npz[f'{kind}_freq'], npz[kind]
            labels = []
        extra = {k[6:]: npz[k] for k in npz.files if k.startswith('extra_')}
        return cls(kind=kind, x=x, y=y, labels=labels or record.get('labels', []),
                   summary=record.get('summary', {}), meta=record.get('meta', {}),
                   extra=extra)

    def as_dict(self) -> dict:
        """JSON-friendly dict with the curve included (for a combined report)."""
        d = {'kind': self.kind, 'x': self.x.tolist(), 'y': self.y.tolist(),
             'labels': self.labels, 'summary': _jsonable(self.summary),
             'meta': _jsonable(self.meta)}
        if self.extra:
            d['extra'] = {k: np.asarray(v).tolist() for k, v in self.extra.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MetricResult":
        return cls(kind=d['kind'], x=np.asarray(d['x']), y=np.asarray(d['y']),
                   labels=list(d.get('labels', [])), summary=dict(d.get('summary', {})),
                   meta=dict(d.get('meta', {})),
                   extra={k: np.asarray(v) for k, v in d.get('extra', {}).items()})

    def describe(self) -> str:
        """The summary as aligned ``key: value`` lines."""
        if not self.summary:
            return f"{self.kind}: no summary"
        width = max(len(k) for k in self.summary)
        lines = [f"{self.kind}"]
        for k, v in self.summary.items():
            if isinstance(v, float):
                lines.append(f"  {k:<{width}}  {v:.4g}")
            else:
                lines.append(f"  {k:<{width}}  {v}")
        return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, slice):
        return [obj.start, obj.stop]
    if isinstance(obj, Path):
        return str(obj)
    return obj
