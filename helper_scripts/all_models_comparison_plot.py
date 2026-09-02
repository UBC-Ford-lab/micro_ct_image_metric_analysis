"""
Comprehensive MTF, NPS, NEQ comparison across all inpainting models.

Creates a 2x3 figure for academic publication:
- Row 1: MTF, NPS, NEQ comparison (all models)
- Row 2: Residuals vs Ground Truth for each metric

Author: Generated with assistance from Claude Code
Date: January 2026

Expected input files after running the full pipeline:
  U-Net Pipeline Outputs (data/results/):
    {scan_name}_gt_recon.vff      - Ground Truth reconstruction
    {scan_name}_under_recon.vff   - Undersampled reconstruction
    {scan_name}_unet_recon.vff    - U-Net reconstruction

  Base Model Outputs (base_models/models/{model}/results/):
    reconstructed_volume.vff      - Reconstructed volume for each model
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from pathlib import Path

from explore_ct120_recon.ct_core import vff_io as vff
from .. import mtf_calculator as MTF_calculator
from .. import nps_calculator as NPS_calculator
from .. import neq_calculator as NEQ_calculator


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Compare MTF/NPS/NEQ across all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python all_models_comparison_plot.py

  # Specify scan name and directories
  python all_models_comparison_plot.py --scan_name Scan_1681 --results_dir data/results

  # Custom output directory
  python all_models_comparison_plot.py --output_dir ./figures
        """
    )
    p.add_argument('--scan_name', type=str, default='Scan_1681',
                   help='Scan name prefix for reconstruction files (default: Scan_1681)')
    p.add_argument('--results_dir', type=str, default='data/results',
                   help='Directory containing GT/Under/U-Net reconstructions (default: data/results)')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory for figures (default: same as script location)')
    return p.parse_args()


def get_vff_files(args):
    """Build VFF file paths dynamically based on arguments."""
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / args.results_dir
    base_models_dir = project_root / "base_models" / "models"

    return {
        'Ground Truth': results_dir / f'{args.scan_name}_gt_recon.vff',
        'Undersampled': results_dir / f'{args.scan_name}_under_recon.vff',
        'U-Net': results_dir / f'{args.scan_name}_unet_recon.vff',
        'LaMa': base_models_dir / 'lama' / 'results' / 'reconstructed_volume.vff',
        'MAT': base_models_dir / 'mat' / 'results' / 'reconstructed_volume.vff',
        'DeepFill v2': base_models_dir / 'deepfill' / 'results' / 'reconstructed_volume.vff',
        'RePaint': base_models_dir / 'repaint' / 'results' / 'reconstructed_volume.vff',
    }


# =============================================================================
# Configuration
# =============================================================================

# Color scheme - colorblind-friendly and distinguishable in grayscale
# Using a categorical palette with distinct hues
COLORS = {
    'Ground Truth': '#000000',   # Black - reference
    'Undersampled': '#D55E00',   # Vermillion/red-orange
    'U-Net': '#E69F00',          # Orange
    'LaMa': '#56B4E9',           # Sky blue
    'MAT': '#0072B2',            # Blue (colorblind-friendly)
    'DeepFill v2': '#009E73',    # Bluish green
    'RePaint': '#CC79A7',        # Reddish purple
}

# Line styles - simple solid lines for all (academic paper style)
LINE_STYLES = {
    'Ground Truth': '-',
    'Undersampled': '-',
    'U-Net': '-',
    'LaMa': '-',
    'MAT': '-',
    'DeepFill v2': '-',
    'RePaint': '-',
}

# Line widths
LINE_WIDTHS = {
    'Ground Truth': 2.0,
    'Undersampled': 1.8,
    'U-Net': 1.5,
    'LaMa': 1.5,
    'MAT': 1.5,
    'DeepFill v2': 1.5,
    'RePaint': 1.5,
}

# Markers - disabled for clean academic paper style (differentiate by color only)
MARKERS = {
    'Ground Truth': None,
    'Undersampled': None,
    'U-Net': None,
    'LaMa': None,
    'MAT': None,
    'DeepFill v2': None,
    'RePaint': None,
}

# Marker spacing (plot every Nth point)
MARKER_EVERY = 15

# Alpha (opacity) values - more transparent for overlapping visibility
ALPHAS = {
    'Ground Truth': 0.9,
    'Undersampled': 0.7,
    'U-Net': 0.7,
    'LaMa': 0.7,
    'MAT': 0.7,
    'DeepFill v2': 0.7,
    'RePaint': 0.7,
}

# Measurement parameters
PIXEL_SIZE = 0.085  # mm

# MTF measurement region
MTF_SLICE_RANGE = [228, 229]
CROP_INDICES_MTF = [270, 664, 522, 640]  # [y1, y2, x1, x2]
EDGE_ANGLE = 5.4  # degrees

# NPS measurement regions
NPS_SLICE_RANGE = np.concatenate((np.arange(204, 208), np.arange(216, 226)))
ROI_BOUNDS_NPS = np.array([
    [178, 294, 510, 626],
    [258, 374, 750, 866],
    [310, 426, 328, 444],
    [432, 548, 830, 946],
    [488, 604, 248, 364],
    [580, 696, 730, 846],
    [624, 740, 414, 530],
    [724, 840, 598, 714]
])


def compute_metrics(vff_file, name):
    """
    Compute MTF, NPS, and NEQ for a given VFF reconstruction file.

    Parameters
    ----------
    vff_file : str
        Path to the VFF reconstruction file
    name : str
        Name of the model (for logging)

    Returns
    -------
    dict
        Dictionary containing frequencies and values for MTF, NPS, NEQ
    """
    print(f"Processing {name}...")

    # Load VFF data
    header, image_data = vff.read_vff(vff_file, verbose=False)

    # Extract slice ranges
    image_data_mtf = image_data[MTF_SLICE_RANGE[0]:MTF_SLICE_RANGE[1], :, :]
    image_data_nps = image_data[NPS_SLICE_RANGE, :, :]

    # Calculate MTF (unnormalized for global normalization later)
    print(f"  Computing MTF...")
    mtf_freq, mtf = MTF_calculator.get_MTF(
        image_data_mtf,
        CROP_INDICES_MTF.copy(),  # Copy since function modifies in place
        find_absolute_MTF=True,
        pixel_size=PIXEL_SIZE,
        target_directory=os.getcwd(),
        plot_results=False,
        edge_angle=EDGE_ANGLE,
        high_to_low=True,
        process_LSF=True,
        normalize_MTF=False  # Global normalization applied later
    )

    # Calculate NPS
    print(f"  Computing NPS...")
    nps_freq, nps = NPS_calculator.get_NPS(
        image_data_nps,
        ROI_BOUNDS_NPS,
        pixel_size=PIXEL_SIZE,
        target_directory=os.getcwd(),
        plot_results=False,
        filter_low_freq=False
    )

    # Calculate NEQ
    print(f"  Computing NEQ...")
    neq_freq, neq = NEQ_calculator.get_NEQ(
        image_data_mtf,
        image_data_nps,
        CROP_INDICES_MTF.copy(),
        ROI_BOUNDS_NPS,
        pixel_size=PIXEL_SIZE,
        target_directory=os.getcwd(),
        plot_results=False,
        high_to_low_MTF=True
    )

    return {
        'mtf_freq': mtf_freq,
        'mtf': mtf,
        'nps_freq': nps_freq,
        'nps': nps,
        'neq_freq': neq_freq,
        'neq': neq
    }


def interpolate_to_common_grid(freq_list, value_list, n_points=500):
    """
    Interpolate multiple curves to a common frequency grid.

    Parameters
    ----------
    freq_list : list of ndarray
        List of frequency arrays
    value_list : list of ndarray
        List of value arrays corresponding to frequencies
    n_points : int
        Number of points in common grid

    Returns
    -------
    common_freq : ndarray
        Common frequency grid
    interpolated_values : list of ndarray
        Values interpolated to common grid
    """
    # Find overlapping frequency range (positive frequencies only)
    min_freq = 0
    max_freq = min(np.max(np.abs(f)) for f in freq_list)

    common_freq = np.linspace(min_freq, max_freq, n_points)

    interpolated_values = []
    for freq, val in zip(freq_list, value_list):
        # Handle bidirectional frequencies (take positive side)
        if np.any(freq < 0):
            pos_mask = freq >= 0
            freq_pos = freq[pos_mask]
            val_pos = val[pos_mask]
        else:
            freq_pos = freq
            val_pos = val

        # Sort by frequency
        sort_idx = np.argsort(freq_pos)
        freq_sorted = freq_pos[sort_idx]
        val_sorted = val_pos[sort_idx]

        # Interpolate
        val_interp = np.interp(common_freq, freq_sorted, val_sorted)
        interpolated_values.append(val_interp)

    return common_freq, interpolated_values


def create_comparison_figure(results, output_dir):
    """
    Create the comprehensive 2x3 comparison figure.

    Parameters
    ----------
    results : dict
        Dictionary of results keyed by model name
    output_dir : str
        Directory to save the output figure
    """
    # Set up matplotlib for publication quality
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'text.usetex': False,  # Set True if LaTeX is available
    })

    # Create figure with 2 rows, 3 columns
    # Larger figure to prevent text overlap
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))

    model_names = list(results.keys())
    gt_name = 'Ground Truth'

    # Normalize each MTF curve individually to unity
    for name in model_names:
        results[name]['mtf_normalized'] = results[name]['mtf'] / results[name]['mtf'].max()

    # Interpolate all metrics to common grids for residual calculation
    mtf_freqs = [results[n]['mtf_freq'] for n in model_names]
    mtf_vals = [results[n]['mtf_normalized'] for n in model_names]
    common_mtf_freq, interp_mtf = interpolate_to_common_grid(mtf_freqs, mtf_vals)

    nps_freqs = [results[n]['nps_freq'] for n in model_names]
    nps_vals = [results[n]['nps'] for n in model_names]
    common_nps_freq, interp_nps = interpolate_to_common_grid(nps_freqs, nps_vals)

    neq_freqs = [results[n]['neq_freq'] for n in model_names]
    neq_vals = [results[n]['neq'] for n in model_names]
    common_neq_freq, interp_neq = interpolate_to_common_grid(neq_freqs, neq_vals)

    # Get ground truth index for residual calculation
    gt_idx = model_names.index(gt_name)

    # ==========================================================================
    # Row 1: Absolute metrics
    # ==========================================================================

    # --- MTF Plot ---
    ax_mtf = axes[0, 0]
    nyquist = 1 / (2 * PIXEL_SIZE)

    for i, name in enumerate(model_names):
        mtf_freq = results[name]['mtf_freq']
        mtf = results[name]['mtf_normalized']

        # Plot only positive frequencies
        pos_mask = mtf_freq >= 0
        ax_mtf.plot(
            mtf_freq[pos_mask], mtf[pos_mask],
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            linewidth=LINE_WIDTHS[name],
            label=name,
            alpha=ALPHAS[name],
            marker=MARKERS[name],
            markevery=MARKER_EVERY,
            markersize=4,
            markerfacecolor='white' if MARKERS[name] else None,
            markeredgecolor=COLORS[name] if MARKERS[name] else None,
            markeredgewidth=1.2
        )

    ax_mtf.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_mtf.axhline(0.1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_mtf.text(0.98, 0.52, 'MTF50', fontsize=7, color='gray', ha='right', transform=ax_mtf.get_yaxis_transform())
    ax_mtf.text(0.98, 0.12, 'MTF10', fontsize=7, color='gray', ha='right', transform=ax_mtf.get_yaxis_transform())

    ax_mtf.set_xlabel('Spatial Frequency (lp/mm)')
    ax_mtf.set_ylabel('MTF')
    ax_mtf.set_title('(a) MTF', fontweight='bold', loc='left')
    ax_mtf.set_xlim([0, nyquist])
    ax_mtf.set_ylim([0, 1.05])
    ax_mtf.xaxis.set_minor_locator(AutoMinorLocator())
    ax_mtf.yaxis.set_minor_locator(AutoMinorLocator())
    ax_mtf.grid(True, alpha=0.3, linewidth=0.5)
    ax_mtf.grid(True, which='minor', alpha=0.15, linewidth=0.3)

    # --- NPS Plot ---
    ax_nps = axes[0, 1]

    for i, name in enumerate(model_names):
        nps_freq = results[name]['nps_freq']
        nps = results[name]['nps']

        # Avoid log(0)
        nps_safe = np.maximum(nps, np.max(nps) * 1e-8)

        ax_nps.semilogy(
            nps_freq, nps_safe,
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            linewidth=LINE_WIDTHS[name],
            label=name,
            alpha=ALPHAS[name],
            marker=MARKERS[name],
            markevery=MARKER_EVERY,
            markersize=4,
            markerfacecolor='white' if MARKERS[name] else None,
            markeredgecolor=COLORS[name] if MARKERS[name] else None,
            markeredgewidth=1.2
        )

    ax_nps.set_xlabel('Spatial Frequency (lp/mm)')
    ax_nps.set_ylabel('NPS (HU$^2$ mm$^2$)')
    ax_nps.set_title('(b) NPS', fontweight='bold', loc='left')
    ax_nps.set_xlim([0, np.max(results[gt_name]['nps_freq'])])
    ax_nps.xaxis.set_minor_locator(AutoMinorLocator())
    ax_nps.grid(True, alpha=0.3, linewidth=0.5)
    ax_nps.grid(True, which='minor', alpha=0.15, linewidth=0.3)

    # --- NEQ Plot ---
    ax_neq = axes[0, 2]

    for i, name in enumerate(model_names):
        neq_freq = results[name]['neq_freq']
        neq = results[name]['neq']

        # Filter invalid values
        valid_mask = np.isfinite(neq) & (neq > 0)

        if np.any(valid_mask):
            ax_neq.plot(
                neq_freq[valid_mask], neq[valid_mask],
                color=COLORS[name],
                linestyle=LINE_STYLES[name],
                linewidth=LINE_WIDTHS[name],
                label=name,
                alpha=ALPHAS[name],
                marker=MARKERS[name],
                markevery=MARKER_EVERY,
                markersize=4,
                markerfacecolor='white' if MARKERS[name] else None,
                markeredgecolor=COLORS[name] if MARKERS[name] else None,
                markeredgewidth=1.2
            )

    ax_neq.set_xlabel('Spatial Frequency (lp/mm)')
    ax_neq.set_ylabel('NEQ (mm$^{-2}$)')
    ax_neq.set_title('(c) NEQ', fontweight='bold', loc='left')
    ax_neq.set_xlim([0, np.max(results[gt_name]['neq_freq'])])
    ax_neq.set_ylim(bottom=0)
    ax_neq.xaxis.set_minor_locator(AutoMinorLocator())
    ax_neq.yaxis.set_minor_locator(AutoMinorLocator())
    ax_neq.grid(True, alpha=0.3, linewidth=0.5)
    ax_neq.grid(True, which='minor', alpha=0.15, linewidth=0.3)

    # Add text box with peak NEQ values (sorted descending)
    peak_neq_data = []
    for name in model_names:
        neq = results[name]['neq']
        valid_mask = np.isfinite(neq) & (neq > 0)
        if np.any(valid_mask):
            peak_val = np.max(neq[valid_mask])
            peak_neq_data.append((name, peak_val))
    peak_neq_data.sort(key=lambda x: x[1], reverse=True)
    peak_neq_lines = [f"{name}: {val:.1f}" for name, val in peak_neq_data]
    peak_neq_text = "Peak NEQ (mm$^{-2}$)\n" + "\n".join(peak_neq_lines)

    textbox_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='gray', alpha=0.9)
    ax_neq.text(0.97, 0.97, peak_neq_text, transform=ax_neq.transAxes,
                fontsize=7, verticalalignment='top', horizontalalignment='right',
                bbox=textbox_props, family='monospace')

    # ==========================================================================
    # Row 2: Residuals vs Ground Truth
    # ==========================================================================

    gt_mtf = interp_mtf[gt_idx]
    gt_nps = interp_nps[gt_idx]
    gt_neq = interp_neq[gt_idx]

    # --- MTF Residuals ---
    ax_mtf_res = axes[1, 0]

    for i, name in enumerate(model_names):
        if name == gt_name:
            continue  # Skip ground truth (residual would be zero)

        residual = interp_mtf[i] - gt_mtf

        ax_mtf_res.plot(
            common_mtf_freq, residual,
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            linewidth=LINE_WIDTHS[name],
            label=name,
            alpha=ALPHAS[name],
            marker=MARKERS[name],
            markevery=MARKER_EVERY,
            markersize=4,
            markerfacecolor='white' if MARKERS[name] else None,
            markeredgecolor=COLORS[name] if MARKERS[name] else None,
            markeredgewidth=1.2
        )

    ax_mtf_res.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_mtf_res.set_xlabel('Spatial Frequency (lp/mm)')
    ax_mtf_res.set_ylabel('$\Delta$MTF (Model $-$ GT)')
    ax_mtf_res.set_title('(d) MTF Residual', fontweight='bold', loc='left')
    ax_mtf_res.set_xlim([0, nyquist])
    ax_mtf_res.xaxis.set_minor_locator(AutoMinorLocator())
    ax_mtf_res.yaxis.set_minor_locator(AutoMinorLocator())
    ax_mtf_res.grid(True, alpha=0.3, linewidth=0.5)
    ax_mtf_res.grid(True, which='minor', alpha=0.15, linewidth=0.3)

    # --- NPS Residuals (ratio, log scale) ---
    ax_nps_res = axes[1, 1]

    for i, name in enumerate(model_names):
        if name == gt_name:
            continue

        # Use ratio for NPS (Model/GT) - works well with log scale
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = interp_nps[i] / gt_nps
            # Replace invalid values
            ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)

        ax_nps_res.semilogy(
            common_nps_freq, ratio,
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            linewidth=LINE_WIDTHS[name],
            label=name,
            alpha=ALPHAS[name],
            marker=MARKERS[name],
            markevery=MARKER_EVERY,
            markersize=4,
            markerfacecolor='white' if MARKERS[name] else None,
            markeredgecolor=COLORS[name] if MARKERS[name] else None,
            markeredgewidth=1.2
        )

    # Reference line at ratio = 1 (identical to GT)
    ax_nps_res.axhline(1, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_nps_res.set_xlabel('Spatial Frequency (lp/mm)')
    ax_nps_res.set_ylabel('NPS Ratio (Model / GT)')
    ax_nps_res.set_title('(e) NPS Ratio', fontweight='bold', loc='left')
    ax_nps_res.set_xlim([0, np.max(common_nps_freq)])
    ax_nps_res.xaxis.set_minor_locator(AutoMinorLocator())
    ax_nps_res.grid(True, alpha=0.3, linewidth=0.5)
    ax_nps_res.grid(True, which='minor', alpha=0.15, linewidth=0.3)

    # --- NEQ Residuals ---
    ax_neq_res = axes[1, 2]

    for i, name in enumerate(model_names):
        if name == gt_name:
            continue

        # Handle potential inf/nan in NEQ
        neq_model = interp_neq[i].copy()
        neq_gt = gt_neq.copy()

        valid_mask = np.isfinite(neq_model) & np.isfinite(neq_gt) & (neq_gt > 0)

        if np.any(valid_mask):
            residual = neq_model[valid_mask] - neq_gt[valid_mask]
            freq_valid = common_neq_freq[valid_mask]

            ax_neq_res.plot(
                freq_valid, residual,
                color=COLORS[name],
                linestyle=LINE_STYLES[name],
                linewidth=LINE_WIDTHS[name],
                label=name,
                alpha=ALPHAS[name],
                marker=MARKERS[name],
                markevery=MARKER_EVERY,
                markersize=4,
                markerfacecolor='white' if MARKERS[name] else None,
                markeredgecolor=COLORS[name] if MARKERS[name] else None,
                markeredgewidth=1.2
            )

    ax_neq_res.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_neq_res.set_xlabel('Spatial Frequency (lp/mm)')
    ax_neq_res.set_ylabel('$\Delta$NEQ (mm$^{-2}$)')
    ax_neq_res.set_title('(f) NEQ Residual', fontweight='bold', loc='left')
    ax_neq_res.set_xlim([0, np.max(common_neq_freq)])
    ax_neq_res.xaxis.set_minor_locator(AutoMinorLocator())
    ax_neq_res.yaxis.set_minor_locator(AutoMinorLocator())
    ax_neq_res.grid(True, alpha=0.3, linewidth=0.5)
    ax_neq_res.grid(True, which='minor', alpha=0.15, linewidth=0.3)

    # ==========================================================================
    # Legend and layout
    # ==========================================================================

    # Create a single legend for the entire figure
    # Place legend below the figure to avoid overlap
    handles, labels = ax_mtf.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,  # 2 rows of 3 models each
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=9,
        columnspacing=1.5,
        handlelength=2.5
    )

    # Adjust layout with proper spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.3)

    # Save figures
    os.makedirs(output_dir, exist_ok=True)

    # PNG for quick viewing
    png_path = os.path.join(output_dir, 'all_models_MTF_NPS_NEQ_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {png_path}")

    # PDF for publication
    pdf_path = os.path.join(output_dir, 'all_models_MTF_NPS_NEQ_comparison.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")

    # Also save as EPS for some journals
    eps_path = os.path.join(output_dir, 'all_models_MTF_NPS_NEQ_comparison.eps')
    try:
        plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        print(f"Saved: {eps_path}")
    except Exception as e:
        print(f"Could not save EPS (may require additional backends): {e}")

    return fig


def print_summary_statistics(results):
    """
    Print summary statistics for all models.
    """
    print("\n" + "=" * 80)
    print("QUANTITATIVE METRICS SUMMARY")
    print("=" * 80)

    model_names = list(results.keys())

    for name in model_names:
        print(f"\n{name.upper()}")
        print("-" * 40)

        # MTF metrics (individually normalized to unity)
        mtf_freq = results[name]['mtf_freq']
        mtf = results[name]['mtf'] / results[name]['mtf'].max()

        pos_mask = mtf_freq >= 0
        mtf_freq_pos = mtf_freq[pos_mask]
        mtf_pos = mtf[pos_mask]

        # Interpolate for accurate threshold finding
        mtf_freq_interp = np.linspace(0, np.max(mtf_freq_pos), 1000)
        mtf_interp = np.interp(mtf_freq_interp, mtf_freq_pos, mtf_pos)

        try:
            mtf50_idx = np.where((mtf_interp < 0.5) & (mtf_freq_interp > 0))[0][0]
            mtf50 = mtf_freq_interp[mtf50_idx]
        except IndexError:
            mtf50 = np.nan

        try:
            mtf10_idx = np.where((mtf_interp < 0.1) & (mtf_freq_interp > 0))[0][0]
            mtf10 = mtf_freq_interp[mtf10_idx]
        except IndexError:
            mtf10 = np.nan

        print(f"  MTF50: {mtf50:.3f} lp/mm")
        print(f"  MTF10: {mtf10:.3f} lp/mm")

        # NPS metrics
        nps_freq = results[name]['nps_freq']
        nps = results[name]['nps']
        noise_variance = np.trapz(nps, nps_freq)
        peak_nps = np.max(nps)
        peak_nps_freq = nps_freq[np.argmax(nps)]

        print(f"  Noise Variance (σ²): {noise_variance:.2f} HU²")
        print(f"  Peak NPS: {peak_nps:.4f} HU² mm² at {peak_nps_freq:.2f} lp/mm")

        # NEQ metrics
        neq_freq = results[name]['neq_freq']
        neq = results[name]['neq']
        valid_mask = np.isfinite(neq) & (neq > 0)

        if np.any(valid_mask):
            neq_valid = neq[valid_mask]
            neq_freq_valid = neq_freq[valid_mask]

            peak_neq = np.max(neq_valid)
            peak_neq_freq = neq_freq_valid[np.argmax(neq_valid)]
            integrated_neq = np.trapz(neq_valid, neq_freq_valid)

            print(f"  Peak NEQ: {peak_neq:.3f} mm⁻² at {peak_neq_freq:.2f} lp/mm")
            print(f"  Integrated NEQ: {integrated_neq:.3f} mm⁻¹")
        else:
            print(f"  NEQ: No valid data")

    print("\n" + "=" * 80)


def main():
    """
    Main function to generate the comprehensive comparison figure.
    """
    args = parse_args()

    print("=" * 80)
    print("All Models MTF/NPS/NEQ Comparison Plot Generator")
    print("=" * 80)
    print(f"Scan name: {args.scan_name}")
    print(f"Results directory: {args.results_dir}")

    # Get VFF file paths based on arguments
    vff_files = get_vff_files(args)

    # Compute metrics for all available models
    results = {}
    available_models = []

    for name, vff_path in vff_files.items():
        vff_path_str = str(vff_path)
        if os.path.exists(vff_path_str):
            try:
                results[name] = compute_metrics(vff_path_str, name)
                available_models.append(name)
            except Exception as e:
                print(f"Warning: Could not process {name}: {e}")
        else:
            print(f"Warning: VFF file not found for {name}: {vff_path}")

    if len(available_models) < 2:
        print("Error: Need at least 2 models for comparison")
        return None, None

    print(f"\nProcessed {len(available_models)} models: {', '.join(available_models)}")

    # Print summary statistics
    print_summary_statistics(results)

    # Create the comparison figure
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    fig = create_comparison_figure(results, output_dir)

    print("\nFigure generation complete!")

    return fig, results


if __name__ == '__main__':
    fig, results = main()
