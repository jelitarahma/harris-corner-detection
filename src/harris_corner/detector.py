from __future__ import annotations

import cv2
import numpy as np

from harris_corner.filters import box_sum, compute_sobel_derivatives, shift_zero_pad

DEFAULT_U = 5
DEFAULT_V = 5
DEFAULT_WINDOW_SIZE = (5, 5)
DEFAULT_K = 0.04
DEFAULT_THRESHOLD_RATIO = 0.01
DEFAULT_NMS_KSIZE = 3


def _validate_image_and_window(
    image: np.ndarray,
    window_size: tuple[int, int],
) -> None:
    """Validate 2-D image and window size parameters."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy ndarray, got {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2-D grayscale array, got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")

    wh, ww = window_size
    if wh <= 0 or ww <= 0:
        raise ValueError(f"window_size dimensions must be positive, got ({wh}, {ww})")
    if wh % 2 == 0 or ww % 2 == 0:
        raise ValueError(
            f"window_size dimensions must be odd integers, got ({wh}, {ww})"
        )


def corner_score(
    image: np.ndarray,
    u: int = DEFAULT_U,
    v: int = DEFAULT_V,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
) -> np.ndarray:
    """Calculate the exact SSD-based corner score map for all image pixels.

    For each pixel (i, j), the corner response is defined as:
        E(u, v) = sum_{(x,y) in W(i,j)} [I(x+u, y+v) - I(x, y)]^2

    where W(i, j) is a window of size `window_size` centered at (i, j),
    u is horizontal shift (columns), and v is vertical shift (rows).

    Boundary Handling
    -----------------
    Per assignment specifications:
    1. The shift I(x+u, y+v) uses explicit zero-padding (pixels outside image are 0).
    2. The summation window uses zero-padding (cv2.BORDER_CONSTANT, value=0).

    Parameters
    ----------
    image : np.ndarray
        2-D array of shape (H, W).
    u : int, default=5
        Horizontal column shift.
    v : int, default=5
        Vertical row shift.
    window_size : tuple[int, int], default=(5, 5)
        (height, width) of the summation window; must be positive odd integers.

    Returns
    -------
    np.ndarray
        Corner score map of shape (H, W), dtype float64.
    """
    _validate_image_and_window(image, window_size)

    img_f64 = image.astype(np.float64, copy=False)

    # Explicit zero-padded shift (no np.roll wrap-around)
    shifted = shift_zero_pad(img_f64, u=u, v=v)

    # Pixel-wise squared difference
    diff_sq = (shifted - img_f64) ** 2

    # Windowed summation with ZERO-padding at boundaries
    score_map = box_sum(
        diff_sq, window_size=window_size, border_type=cv2.BORDER_CONSTANT
    )
    return score_map


def corner_score_taylor(
    image: np.ndarray,
    x: int = DEFAULT_U,
    y: int = DEFAULT_V,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
) -> np.ndarray:
    """Calculate first-order Taylor approximation of the SSD corner score map.

    Under the Taylor series expansion:
        I(p + d) - I(p) ≈ d^T ∇I(p) = x * Ix(p) + y * Iy(p)

    The squared difference integrand becomes:
        [I(p + d) - I(p)]^2 ≈ (x * Ix + y * Iy)^2
                            = x^2 * Ix^2 + 2 * x * y * Ix * Iy + y^2 * Iy^2

    Parameters
    ----------
    image : np.ndarray
        2-D array of shape (H, W).
    x : int, default=5
        Horizontal shift magnitude (column direction).
    y : int, default=5
        Vertical shift magnitude (row direction).
    window_size : tuple[int, int], default=(5, 5)
        (height, width) of summation window; must be positive odd integers.

    Returns
    -------
    np.ndarray
        Taylor-approximated corner score map of shape (H, W), dtype float64.
    """
    _validate_image_and_window(image, window_size)

    img_f64 = image.astype(np.float64, copy=False)
    ix, iy = compute_sobel_derivatives(img_f64)

    # Normalize Sobel derivative by its kernel scale factor (8.0 for 3x3 Sobel)
    # to convert integer filter response to true spatial gradient dI/dx, dI/dy
    gx = ix / 8.0
    gy = iy / 8.0

    # First-order Taylor integrand (per-pixel): (x * dI/dx + y * dI/dy)^2
    integrand = (float(x) * gx + float(y) * gy) ** 2

    # Windowed summation matching corner_score zero-padded boundary mode
    score_map = box_sum(
        integrand, window_size=window_size, border_type=cv2.BORDER_CONSTANT
    )
    return score_map


def harris_detector(
    image: np.ndarray,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
    k: float = DEFAULT_K,
) -> np.ndarray:
    """Compute the Harris corner response map R for every image pixel.

    Algorithm
    ---------
    1. Compute spatial gradients Ix, Iy using 3x3 Sobel filters.
    2. Compute second moments: Ixx = Ix^2, Iyy = Iy^2, Ixy = Ix * Iy.
    3. Construct structure tensor M of shape (H, W, 2, 2) using BORDER_REPLICATE
       padding for window summation.
    4. Compute scalar corner response:
           R = det(M) - k * (trace(M))^2
           det(M) = M[0,0] * M[1,1] - M[0,1]^2
           trace(M) = M[0,0] + M[1,1]

    Parameters
    ----------
    image : np.ndarray
        2-D array of shape (H, W).
    window_size : tuple[int, int], default=(5, 5)
        (height, width) of summation window; must be positive odd integers.
    k : float, default=0.04
        Harris sensitivity constant. Typically 0.04 - 0.15 per Lecture Notes LN 07.

    Returns
    -------
    np.ndarray
        Harris response map R of shape (H, W), dtype float64.
    """
    _validate_image_and_window(image, window_size)

    if k <= 0:
        raise ValueError(f"k must be strictly positive (k > 0), got {k}")

    img_f64 = image.astype(np.float64, copy=False)

    # 1. Turunan spasial
    Ix, Iy = compute_sobel_derivatives(img_f64)

    # 2. Komponen turunan kuadrat
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 3. Structure tensor M per piksel, shape (H, W, 2, 2), same-padding
    M = np.empty(image.shape + (2, 2), dtype=np.float64)
    M[..., 0, 0] = box_sum(Ixx, window_size, cv2.BORDER_REPLICATE)
    M[..., 1, 1] = box_sum(Iyy, window_size, cv2.BORDER_REPLICATE)
    M[..., 0, 1] = M[..., 1, 0] = box_sum(Ixy, window_size, cv2.BORDER_REPLICATE)

    # 4. Respons Harris: R = det(M) - k * trace(M)^2
    det_m = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    trace_m = M[..., 0, 0] + M[..., 1, 1]
    return det_m - k * trace_m**2


def detect_corners(
    response: np.ndarray,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    nms_ksize: int = DEFAULT_NMS_KSIZE,
) -> np.ndarray:
    """Detect corner coordinates using relative thresholding and NMS.

    Parameters
    ----------
    response : np.ndarray
        2-D float64 Harris response map of shape (H, W).
    threshold_ratio : float, default=0.01
        Corner threshold as a fraction of maximum response:
        threshold = threshold_ratio * max(R) (per OpenCV tutorial standard).
    nms_ksize : int, default=3
        Kernel size for non-maximum suppression (e.g. 3 for 3x3 local maxima).
        If <= 1, NMS is bypassed.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing (row, col) integer coordinates of detected corners.
    """
    if response.ndim != 2:
        raise ValueError(f"response must be a 2-D array, got shape {response.shape}")

    r_max = float(response.max())
    if r_max <= 0:
        return np.empty((0, 2), dtype=np.int32)

    thresh = threshold_ratio * r_max
    mask = response > thresh

    if nms_ksize > 1:
        # Non-maximum suppression via dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (nms_ksize, nms_ksize))
        dilated = cv2.dilate(response.astype(np.float64), kernel)
        mask = mask & (response >= (dilated - 1e-8))

    corners = np.argwhere(mask)
    return corners.astype(np.int32)
