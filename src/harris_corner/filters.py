"""Spatial filtering and derivative operations for Harris Corner Detection."""

from __future__ import annotations

import cv2
import numpy as np


def compute_sobel_derivatives(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatial derivatives Ix and Iy using 3x3 Sobel operators.

    Parameters
    ----------
    image : np.ndarray
        2-D float64 grayscale image with shape (H, W).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Horizontal (Ix) and vertical (Iy) gradient images of shape (H, W).
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2-D grayscale array, got shape {image.shape}"
        )

    img_f64 = image.astype(np.float64, copy=False)

    # Use BORDER_REPLICATE for derivatives to avoid artificial edge spikes at boundaries
    ix = cv2.Sobel(
        img_f64,
        ddepth=cv2.CV_64F,
        dx=1,
        dy=0,
        ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )
    iy = cv2.Sobel(
        img_f64,
        ddepth=cv2.CV_64F,
        dx=0,
        dy=1,
        ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )
    return ix, iy


def box_sum(
    arr: np.ndarray,
    window_size: tuple[int, int],
    border_type: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Compute unnormalized sliding window summation over a 2-D array.

    Parameters
    ----------
    arr : np.ndarray
        2-D array of shape (H, W).
    window_size : tuple[int, int]
        (height, width) of the summation window. Must be positive odd integers.
    border_type : int, default=cv2.BORDER_REPLICATE
        OpenCV border extrapolation mode (e.g. cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE).

    Returns
    -------
    np.ndarray
        Summed 2-D array of shape (H, W), float64.
    """
    if arr.ndim != 2:
        raise ValueError(f"arr must be a 2-D array, got shape {arr.shape}")

    wh, ww = window_size
    if wh <= 0 or ww <= 0:
        raise ValueError(f"window_size dimensions must be positive, got ({wh}, {ww})")
    if wh % 2 == 0 or ww % 2 == 0:
        raise ValueError(
            f"window_size dimensions must be odd integers, got ({wh}, {ww})"
        )

    arr_f64 = arr.astype(np.float64, copy=False)
    # ksize in OpenCV boxFilter is (width, height)
    result = cv2.boxFilter(
        arr_f64,
        ddepth=cv2.CV_64F,
        ksize=(ww, wh),
        normalize=False,
        borderType=border_type,
    )
    return result


def shift_zero_pad(image: np.ndarray, u: int, v: int) -> np.ndarray:
    """Shift a 2-D image so that at pixel (y, x), output holds I(y + v, x + u).

    Pixels outside boundaries are zero-padded without toroidal wrapping.

    Parameters
    ----------
    image : np.ndarray
        2-D float64 array of shape (H, W).
    u : int
        Horizontal shift in columns.
    v : int
        Vertical shift in rows.

    Returns
    -------
    np.ndarray
        Shifted image of shape (H, W), with zero-padded boundaries.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")

    h, w = image.shape
    shifted = np.zeros((h, w), dtype=np.float64)

    # If shift exceeds or equals image dimensions, result is entirely zero
    if abs(v) >= h or abs(u) >= w:
        return shifted

    if v >= 0:
        r_dst = slice(0, h - v)
        r_src = slice(v, h)
    else:
        r_dst = slice(-v, h)
        r_src = slice(0, h + v)

    if u >= 0:
        c_dst = slice(0, w - u)
        c_src = slice(u, w)
    else:
        c_dst = slice(-u, w)
        c_src = slice(0, w + u)

    shifted[r_dst, c_dst] = image[r_src, c_src]
    return shifted
