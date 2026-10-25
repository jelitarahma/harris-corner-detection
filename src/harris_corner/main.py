"""Main entry point and CLI runner for Harris Corner Detector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from harris_corner.detector import (
    DEFAULT_K,
    DEFAULT_NMS_KSIZE,
    DEFAULT_THRESHOLD_RATIO,
    DEFAULT_U,
    DEFAULT_V,
    DEFAULT_WINDOW_SIZE,
    corner_score,
    detect_corners,
    harris_detector,
)

logger = logging.getLogger("harris_corner")


def _save_summary_figure(
    original_bgr: np.ndarray,
    gray: np.ndarray,
    score_map: np.ndarray,
    corners_overlay_bgr: np.ndarray,
    corner_count: int,
    out_path: Path,
) -> None:
    """Save clean 2x2 summary figure using Matplotlib Agg backend."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Harris Corner Detection Summary", fontsize=16, fontweight="bold")

    # (a) Original Color Image
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("(a) Original Image", fontsize=12)
    axes[0, 0].axis("off")

    # (b) Grayscale
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("(b) Grayscale Image", fontsize=12)
    axes[0, 1].axis("off")

    # (c) Corner Score SSD heatmap
    im_cs = axes[1, 0].imshow(score_map, cmap="hot")
    axes[1, 0].set_title("(c) Corner Score E(u,v) (SSD)", fontsize=12)
    axes[1, 0].axis("off")
    fig.colorbar(im_cs, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (d) Harris Corners Overlay
    overlay_rgb = cv2.cvtColor(corners_overlay_bgr, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title(f"(d) Detected Corners (count={corner_count})", fontsize=12)
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_single_heatmap(
    arr: np.ndarray,
    title: str,
    cmap: str,
    out_path: Path,
    norm: matplotlib.colors.Normalize | None = None,
) -> None:
    """Save single normalized heatmap figure with colorbar."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    filename: str = "data/building.jpg",
    u: int = DEFAULT_U,
    v: int = DEFAULT_V,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
    k: float = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD_RATIO,
    output_dir: str = "outputs",
    nms_ksize: int = DEFAULT_NMS_KSIZE,
    circle_radius: int = 1,
) -> dict[str, Any]:
    file_path = Path(filename)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Input image file does not exist: {file_path.resolve()}"
        )

    original_bgr = cv2.imread(str(file_path))
    if original_bgr is None:
        raise ValueError(
            f"Failed to decode image from '{file_path}'. File may be corrupt or an unsupported format."
        )

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    logger.info("Loaded image '%s' with shape %s", file_path.name, gray.shape)

    # 1. Compute exact SSD corner score
    logger.info("Computing corner_score (u=%d, v=%d, window=%s)...", u, v, window_size)
    score_map = corner_score(gray, u=u, v=v, window_size=window_size)

    # 2. Compute Harris response map
    logger.info("Computing harris_detector (window=%s, k=%.4f)...", window_size, k)
    response_map = harris_detector(gray, window_size=window_size, k=k)

    # 3. Detect corners with relative thresholding and NMS
    corners = detect_corners(
        response_map, threshold_ratio=threshold, nms_ksize=nms_ksize
    )
    corner_count = len(corners)
    logger.info(
        "Detected %d corners with threshold %.4f * max(R).", corner_count, threshold
    )

    # 4. Create corners overlay, precise red dots on original color image.
    # radius=1 with thickness=-1 produces a minimal 3x3 pixel dot (or exact 1-pixel
    # if radius <= 0) so the actual corner location remains clear and unobstructed.
    overlay_bgr = original_bgr.copy()
    for row, col in corners:
        r_int, c_int = int(row), int(col)
        if circle_radius <= 0:
            overlay_bgr[r_int, c_int] = [0, 0, 255]
        else:
            cv2.circle(
                overlay_bgr,
                (c_int, r_int),
                radius=circle_radius,
                color=(0, 0, 255),
                thickness=-1,
            )

    # 5. Prepare output paths
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_gray = out_dir / "grayscale.png"
    p_cs_png = out_dir / "corner_score.png"
    p_cs_npy = out_dir / "corner_score.npy"
    p_hr_png = out_dir / "harris_response.png"
    p_hr_npy = out_dir / "harris_response.npy"
    p_corners = out_dir / "harris_corners.png"
    p_summary = out_dir / "summary.png"

    # Save arrays
    np.save(str(p_cs_npy), score_map)
    np.save(str(p_hr_npy), response_map)

    # Save PNG images
    cv2.imwrite(str(p_gray), gray)
    cv2.imwrite(str(p_corners), overlay_bgr)

    # Save normalized heatmap PNGs
    _save_single_heatmap(
        score_map,
        title=f"Corner Score E(u={u}, v={v}) [SSD]",
        cmap="hot",
        out_path=p_cs_png,
    )

    r_min = float(np.percentile(response_map, 0.5))
    r_max = float(np.percentile(response_map, 99.5))
    if r_min >= 0:
        r_min = -1.0
    if r_max <= 0:
        r_max = 1.0
    r_norm = TwoSlopeNorm(vcenter=0.0, vmin=r_min, vmax=r_max)
    _save_single_heatmap(
        response_map,
        title=f"Harris Response R = det(M) - {k}*trace(M)^2",
        cmap="coolwarm",
        out_path=p_hr_png,
        norm=r_norm,
    )

    # Save 2x2 summary figure
    _save_summary_figure(
        original_bgr=original_bgr,
        gray=gray,
        score_map=score_map,
        corners_overlay_bgr=overlay_bgr,
        corner_count=corner_count,
        out_path=p_summary,
    )

    logger.info("All outputs successfully saved to '%s'", out_dir.resolve())

    return {
        "original": original_bgr,
        "grayscale": gray,
        "corner_score": score_map,
        "harris_response": response_map,
        "corners": corners,
        "corner_count": corner_count,
        "paths": {
            "grayscale": p_gray,
            "corner_score_png": p_cs_png,
            "corner_score_npy": p_cs_npy,
            "harris_response_png": p_hr_png,
            "harris_response_npy": p_hr_npy,
            "harris_corners": p_corners,
            "summary": p_summary,
        },
    }


def cli() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Harris Corner Detector from scratch in Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="data/building.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--u",
        type=int,
        default=DEFAULT_U,
        help="Horizontal shift offset for corner_score",
    )
    parser.add_argument(
        "--v",
        type=int,
        default=DEFAULT_V,
        help="Vertical shift offset for corner_score",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=list(DEFAULT_WINDOW_SIZE),
        metavar=("ROWS", "COLS"),
        help="Summation window size",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help="Harris detector sensitivity parameter",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_RATIO,
        help="Threshold ratio relative to maximum response",
    )
    parser.add_argument(
        "--nms-ksize",
        type=int,
        default=DEFAULT_NMS_KSIZE,
        help="Kernel size for non-maximum suppression (e.g. 3 for 3x3, <=1 to disable)",
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=1,
        help="Radius of circle markers drawn on detected corners (<=0 for 1-pixel dot)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory to save figures and arrays",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    main(
        filename=args.filename,
        u=args.u,
        v=args.v,
        window_size=tuple(args.window),
        k=args.k,
        threshold=args.threshold,
        output_dir=args.output_dir,
        nms_ksize=args.nms_ksize,
        circle_radius=args.circle_radius,
    )


if __name__ == "__main__":
    cli()
