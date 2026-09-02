#!/usr/bin/env python3
"""
Publication-ready figures comparing reconstructed CT slices from all models.

Creates two figures:
1. all_models_reconstruction_comparison: 2x4 grid showing all 7 models
2. all_models_reconstruction_comparison_with_labels: 2x4 grid with labeled boxes
   highlighting streaking reduction in lower left corner for GT, Undersampled, U-Net

Output formats: PNG (preview), EPS (journal), PDF (alternative)

Expected input files after running the full pipeline:
  U-Net Pipeline Outputs (data/results/):
    {scan_name}_gt_recon.vff      - Ground Truth reconstruction
    {scan_name}_under_recon.vff   - Undersampled reconstruction
    {scan_name}_unet_recon.vff    - U-Net reconstruction

  Base Model Outputs (base_models/models/{model}/results/):
    reconstructed_volume.vff      - Reconstructed volume for each model
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from explore_ct120_recon.ct_core.vff_io import read_vff

# Configure matplotlib for publication quality
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['figure.dpi'] = 300


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Compare reconstruction slices visually across all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python plot_reconstruction_comparison.py

  # Specify scan name and slice index
  python plot_reconstruction_comparison.py --scan_name Scan_1681 --slice_idx 150

  # Custom output directory
  python plot_reconstruction_comparison.py --output_dir ./figures
        """
    )
    p.add_argument('--scan_name', type=str, default='Scan_1681',
                   help='Scan name prefix for reconstruction files (default: Scan_1681)')
    p.add_argument('--results_dir', type=str, default='data/results',
                   help='Directory containing GT/Under/U-Net reconstructions (default: data/results)')
    p.add_argument('--slice_idx', type=int, default=150,
                   help='Slice index to display (default: 150)')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory for figures (default: same as script location)')
    return p.parse_args()


def load_slice(vff_path: Path, slice_idx: int = 150, verbose: bool = False) -> np.ndarray:
    """Load a single slice from a VFF file."""
    header, data = read_vff(str(vff_path), verbose=verbose)
    # Data shape is (z, y, x) - extract the specified slice
    slice_data = np.array(data[slice_idx, :, :])
    return slice_data


def create_all_models_figure(slices, vmin, vmax, output_dir):
    """Create the 2x4 grid figure showing all 7 models (one empty cell)."""

    display_order = [
        ["Ground Truth", "Undersampled", "U-Net", None],
        ["LaMa", "MAT", "DeepFill v2", "RePaint"],
    ]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))

    for row_idx, row_names in enumerate(display_order):
        for col_idx, name in enumerate(row_names):
            ax = axes[row_idx, col_idx]

            if name is None:
                # Empty cell - hide it
                ax.axis('off')
                continue

            # Display the slice with consistent windowing
            ax.imshow(
                slices[name],
                cmap='gray',
                vmin=vmin,
                vmax=vmax,
                aspect='equal'
            )

            ax.set_title(name, fontsize=10, fontweight='normal')
            ax.axis('off')

    # Adjust layout
    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)

    # Save outputs
    base_name = "all_models_reconstruction_comparison"

    png_path = output_dir / f"{base_name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PNG: {png_path}")

    eps_path = output_dir / f"{base_name}.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Saved EPS: {eps_path}")

    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")

    plt.close(fig)


def create_labeled_figure(slices, vmin, vmax, output_dir):
    """Create the 2x4 figure with all models, adding labeled boxes to GT, Undersampled, U-Net."""

    display_order = [
        ["Ground Truth", "Undersampled", "U-Net", None],
        ["LaMa", "MAT", "DeepFill v2", "RePaint"],
    ]

    # Models that get labeled boxes
    models_with_boxes = {"Ground Truth": "i", "Undersampled": "ii", "U-Net": "iii"}

    # Box parameters - lower left corner region showing streaking
    # Image is 1100x1100, lower left would be high y values (bottom), low x values (left)
    box_x = 30       # x position (left edge of box)
    box_y = 870      # y position (top edge of box) - moved further down
    box_width = 200  # width of box
    box_height = 200 # height of box

    # Color for boxes and labels (a visible color that contrasts with grayscale)
    highlight_color = '#E63946'  # Red color for visibility

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))

    for row_idx, row_names in enumerate(display_order):
        for col_idx, name in enumerate(row_names):
            ax = axes[row_idx, col_idx]

            if name is None:
                # Empty cell - hide it
                ax.axis('off')
                continue

            # Display the slice with consistent windowing
            ax.imshow(
                slices[name],
                cmap='gray',
                vmin=vmin,
                vmax=vmax,
                aspect='equal'
            )

            # Add box and label only for specified models
            if name in models_with_boxes:
                label = models_with_boxes[name]

                # Add the box rectangle
                rect = patches.Rectangle(
                    (box_x, box_y),
                    box_width,
                    box_height,
                    linewidth=2,
                    edgecolor=highlight_color,
                    facecolor='none'
                )
                ax.add_patch(rect)

                # Add the label next to the box (to the right of the box)
                ax.text(
                    box_x + box_width + 15,  # x position: right of box
                    box_y + box_height / 2,   # y position: vertically centered with box
                    label,
                    color=highlight_color,
                    fontsize=14,
                    fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='left'
                )

            ax.set_title(name, fontsize=10, fontweight='normal')
            ax.axis('off')

    # Adjust layout
    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)

    # Save outputs
    base_name = "all_models_reconstruction_comparison_with_labels"

    png_path = output_dir / f"{base_name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PNG: {png_path}")

    eps_path = output_dir / f"{base_name}.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Saved EPS: {eps_path}")

    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")

    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 80)
    print("Reconstruction Comparison Plot Generator")
    print("=" * 80)
    print(f"Scan name: {args.scan_name}")
    print(f"Results directory: {args.results_dir}")
    print(f"Slice index: {args.slice_idx}")

    # Define paths to all reconstruction files
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.results_dir
    base_model_dir = project_root / "base_models" / "models"

    reconstruction_files = {
        "Ground Truth": data_dir / f"{args.scan_name}_gt_recon.vff",
        "Undersampled": data_dir / f"{args.scan_name}_under_recon.vff",
        "U-Net": data_dir / f"{args.scan_name}_unet_recon.vff",
        "LaMa": base_model_dir / "lama" / "results" / "reconstructed_volume.vff",
        "MAT": base_model_dir / "mat" / "results" / "reconstructed_volume.vff",
        "DeepFill v2": base_model_dir / "deepfill" / "results" / "reconstructed_volume.vff",
        "RePaint": base_model_dir / "repaint" / "results" / "reconstructed_volume.vff",
    }

    print("\nLoading reconstruction volumes...")

    # Load slices that exist, warn about missing ones
    slices = {}
    for name, path in reconstruction_files.items():
        if not path.exists():
            print(f"WARNING: Missing file for {name}: {path}")
        else:
            print(f"  Loading {name}...")
            slices[name] = load_slice(path, args.slice_idx, verbose=False)

    if "Ground Truth" not in slices:
        print("ERROR: Ground Truth file is required for windowing reference")
        return

    if len(slices) < 2:
        print("ERROR: Need at least 2 models for comparison")
        return

    # Determine consistent windowing from ground truth
    gt_slice = slices["Ground Truth"]

    # Use percentile-based windowing to handle outliers
    vmin = np.percentile(gt_slice, 1)
    vmax = np.percentile(gt_slice, 99)

    print(f"\nWindowing: [{vmin:.1f}, {vmax:.1f}] (1st-99th percentile of GT)")

    # Determine output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create both figures
    print("\n--- Creating all models comparison figure ---")
    create_all_models_figure(slices, vmin, vmax, output_dir)

    print("\n--- Creating labeled comparison figure (GT, Undersampled, U-Net) ---")
    create_labeled_figure(slices, vmin, vmax, output_dir)

    print("\nFigure generation complete.")
    print(f"Slice index: {args.slice_idx}")
    print(f"Image dimensions: {gt_slice.shape}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
