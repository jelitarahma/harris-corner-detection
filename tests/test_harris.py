from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from harris_corner.detector import (
    corner_score,
    corner_score_taylor,
    detect_corners,
    harris_detector,
)
from harris_corner.filters import shift_zero_pad
from harris_corner.main import main


@pytest.fixture
def synthetic_square() -> np.ndarray:
    """Create a 64x64 synthetic image with a white square on a black background."""
    img = np.zeros((64, 64), dtype=np.float64)
    img[16:48, 16:48] = 255.0
    return img


@pytest.fixture
def smooth_image() -> np.ndarray:
    """Create a smooth 64x64 image (2D Gaussian bell) for Taylor expansion testing."""
    y, x = np.mgrid[-2:2:64j, -2:2:64j]
    img = 200.0 * np.exp(-(x**2 + y**2))
    return img.astype(np.float64)


@pytest.fixture
def building_path() -> str:
    """Return valid absolute path to data/building.jpg."""
    p = Path(__file__).resolve().parents[1] / "data" / "building.jpg"
    assert p.is_file(), f"Missing test fixture: {p.resolve()}"
    return str(p)


def test_synthetic_square_corner_edge_flat_response(
    synthetic_square: np.ndarray,
) -> None:
    R = harris_detector(synthetic_square, window_size=(5, 5), k=0.04)

    # Coordinates of the 4 outer corner points of the 16:48 square
    # Since square is [16:48, 16:48], corner transitions occur around rows/cols 16 and 47
    corners_coords = [(16, 16), (16, 47), (47, 16), (47, 47)]
    for r, c in corners_coords:
        # Check window around corner has strongly positive peak
        patch = R[r - 2 : r + 3, c - 2 : c + 3]
        assert (
            patch.max() > 1e6
        ), f"Expected positive corner response near ({r},{c}), got {patch.max()}"

    # Edge points (middle of top, bottom, left, right edges)
    edge_coords = [(16, 32), (47, 32), (32, 16), (32, 47)]
    for r, c in edge_coords:
        assert (
            R[r, c] < 0
        ), f"Expected negative edge response at ({r},{c}), got {R[r, c]}"

    # Flat points (far background and square center)
    flat_coords = [(5, 5), (32, 32), (58, 58)]
    for r, c in flat_coords:
        assert (
            abs(R[r, c]) < 1e-3
        ), f"Expected near-zero response on flat region at ({r},{c}), got {R[r, c]}"


def test_output_shape_and_dtype(synthetic_square: np.ndarray) -> None:
    """Verify output shape equals input shape and dtype is float64."""
    cs = corner_score(synthetic_square, u=5, v=5, window_size=(5, 5))
    assert cs.shape == synthetic_square.shape
    assert cs.dtype == np.float64

    cst = corner_score_taylor(synthetic_square, x=5, y=5, window_size=(5, 5))
    assert cst.shape == synthetic_square.shape
    assert cst.dtype == np.float64

    hd = harris_detector(synthetic_square, window_size=(5, 5), k=0.04)
    assert hd.shape == synthetic_square.shape
    assert hd.dtype == np.float64


def test_corner_score_zero_shift_all_zeros(synthetic_square: np.ndarray) -> None:
    """Shift of u=0, v=0 must produce identically zero SSD across all pixels."""
    cs = corner_score(synthetic_square, u=0, v=0, window_size=(5, 5))
    np.testing.assert_allclose(cs, 0.0, atol=1e-12)


def test_corner_score_zero_padding_against_naive_loop() -> None:
    """Verify vectorized corner_score zero-padding on a tiny 6x6 image against naive loop."""
    np.random.seed(42)
    img = np.random.uniform(10.0, 200.0, size=(6, 6))
    u, v = 2, 1
    wh, ww = 3, 3
    rad_h, rad_w = wh // 2, ww // 2

    # Naive per-pixel loop with explicit zero-padding
    H, W = img.shape
    naive_cs = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            val_sum = 0.0
            for wi in range(i - rad_h, i + rad_h + 1):
                for wj in range(j - rad_w, j + rad_w + 1):
                    # Check window bounds (zero-padded window)
                    if 0 <= wi < H and 0 <= wj < W:
                        i_orig = img[wi, wj]
                        # Shifted position: (wi + v, wj + u)
                        si = wi + v
                        sj = wj + u
                        if 0 <= si < H and 0 <= sj < W:
                            i_shifted = img[si, sj]
                        else:
                            i_shifted = 0.0  # Zero-padded shift
                        val_sum += (i_shifted - i_orig) ** 2
                    else:
                        pass  # Outside window is 0
            naive_cs[i, j] = val_sum

    vec_cs = corner_score(img, u=u, v=v, window_size=(wh, ww))
    np.testing.assert_allclose(vec_cs, naive_cs, atol=1e-10)


def test_taylor_approximates_exact_ssd_for_small_shift(
    smooth_image: np.ndarray,
) -> None:
    """First-order Taylor approximation must closely track exact SSD for 1-pixel shift."""
    # Test 1-pixel horizontal shift (u=1, v=0)
    exact_ssd = corner_score(smooth_image, u=1, v=0, window_size=(5, 5))
    taylor_approx = corner_score_taylor(smooth_image, x=1, y=0, window_size=(5, 5))

    # Evaluate on interior to ignore boundary truncation
    interior_slice = (slice(10, -10), slice(10, -10))
    e_int = exact_ssd[interior_slice]
    t_int = taylor_approx[interior_slice]

    # Relative difference on significant points
    mask = e_int > 1.0
    rel_err = np.abs(e_int[mask] - t_int[mask]) / e_int[mask]

    # Median relative error should be small on smooth image (< 15%)
    assert (
        np.median(rel_err) < 0.15
    ), f"Taylor median relative error too high: {np.median(rel_err)}"


def test_vectorized_harris_equals_naive_loop() -> None:
    """Verify vectorized harris_detector equals naive per-pixel loop on small random image."""
    np.random.seed(123)
    img = np.random.uniform(0.0, 255.0, size=(12, 12))
    k = 0.04
    wh, ww = 3, 3
    rad_h, rad_w = wh // 2, ww // 2

    # Compute gradients with Sobel
    ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)

    H, W = img.shape
    naive_r = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            sIxx, sIyy, sIxy = 0.0, 0.0, 0.0
            for wi in range(i - rad_h, i + rad_h + 1):
                for wj in range(j - rad_w, j + rad_w + 1):
                    # BORDER_REPLICATE clamp
                    ci = min(max(wi, 0), H - 1)
                    cj = min(max(wj, 0), W - 1)
                    gx = ix[ci, cj]
                    gy = iy[ci, cj]
                    sIxx += gx * gx
                    sIyy += gy * gy
                    sIxy += gx * gy

            det_m = sIxx * sIyy - (sIxy * sIxy)
            trace_m = sIxx + sIyy
            naive_r[i, j] = det_m - k * (trace_m * trace_m)

    vec_r = harris_detector(img, window_size=(wh, ww), k=k)
    np.testing.assert_allclose(vec_r, naive_r, atol=1e-8)


def test_correlation_with_cv2_corner_harris(building_path: str) -> None:
    bgr = cv2.imread(building_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Custom implementation (window 5x5, k=0.04)
    r_custom = harris_detector(gray, window_size=(5, 5), k=0.04)

    # OpenCV cv2.cornerHarris (blockSize=5, ksize=3, k=0.04)
    r_cv2 = cv2.cornerHarris(
        gray.astype(np.float32),
        blockSize=5,
        ksize=3,
        k=0.04,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float64)

    # Pearson correlation coefficient r = cov(X, Y) / (std_X * std_Y)
    r_custom_flat = r_custom.ravel()
    r_cv2_flat = r_cv2.ravel()

    corr_matrix = np.corrcoef(r_custom_flat, r_cv2_flat)
    pearson_r = corr_matrix[0, 1]

    assert pearson_r > 0.90, f"Expected Pearson correlation > 0.90, got {pearson_r:.4f}"


def test_invalid_input_validation() -> None:
    # 3D image
    img_3d = np.zeros((10, 10, 3))
    with pytest.raises(ValueError, match="2-D grayscale"):
        corner_score(img_3d)
    with pytest.raises(ValueError, match="2-D grayscale"):
        harris_detector(img_3d)

    # Even window size
    img_2d = np.zeros((10, 10))
    with pytest.raises(ValueError, match="odd integers"):
        corner_score(img_2d, window_size=(4, 4))
    with pytest.raises(ValueError, match="odd integers"):
        harris_detector(img_2d, window_size=(4, 4))

    # Non-positive k
    with pytest.raises(ValueError, match="strictly positive"):
        harris_detector(img_2d, k=-0.04)
    with pytest.raises(ValueError, match="strictly positive"):
        harris_detector(img_2d, k=0.0)


def test_main_missing_file_raises_filenotfound() -> None:
    """main() must raise FileNotFoundError if file does not exist."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        main("non_existent_image_12345.jpg")


def test_shift_zero_pad() -> None:
    """Verify shift_zero_pad handles horizontal, vertical, zero, and boundary shifts."""
    grid = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    # u=0, v=0 should return identical array
    np.testing.assert_array_equal(shift_zero_pad(grid, 0, 0), grid)

    # u=1, v=0: out[y, x] = I[y, x+1], right column padded with 0
    s_right = shift_zero_pad(grid, u=1, v=0)
    expected_right = np.array(
        [
            [2.0, 3.0, 0.0],
            [5.0, 6.0, 0.0],
            [8.0, 9.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(s_right, expected_right)

    # u=-1, v=0: out[y, x] = I[y, x-1], left column padded with 0
    s_left = shift_zero_pad(grid, u=-1, v=0)
    expected_left = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 4.0, 5.0],
            [0.0, 7.0, 8.0],
        ]
    )
    np.testing.assert_array_equal(s_left, expected_left)

    # Large shift exceeding bounds produces all zeros
    s_out = shift_zero_pad(grid, u=5, v=5)
    np.testing.assert_array_equal(s_out, np.zeros_like(grid))


def test_detect_corners_on_synthetic_square(synthetic_square: np.ndarray) -> None:
    """Verify detect_corners isolates corners and clusters them near the 4 true corners."""
    R = harris_detector(synthetic_square, window_size=(5, 5), k=0.04)
    corners = detect_corners(R, threshold_ratio=0.01, nms_ksize=3)

    assert len(corners) > 0, "detect_corners should detect corners on synthetic square"

    # All detected corners should fall near outer corners: (16, 16), (16, 47), (47, 16), (47, 47)
    expected_corners = [(16, 16), (16, 47), (47, 16), (47, 47)]
    for r, c in corners:
        min_dist = min(np.hypot(r - er, c - ec) for er, ec in expected_corners)
        assert (
            min_dist <= 3.0
        ), f"Detected corner ({r}, {c}) is too far from true corners (min_dist={min_dist})"


def test_main_default_arguments_and_saved_files(
    tmp_path: Path, building_path: str
) -> None:
    """main() must run with defaults and create all 7 required artifacts in outputs/."""
    out_dir = tmp_path / "test_outputs"
    res = main(filename=building_path, output_dir=str(out_dir))

    # Verify return dictionary structure
    assert "original" in res
    assert "grayscale" in res
    assert "corner_score" in res
    assert "harris_response" in res
    assert "corners" in res
    assert res["corner_count"] > 0

    # Verify saved files
    expected_files = [
        out_dir / "grayscale.png",
        out_dir / "corner_score.png",
        out_dir / "corner_score.npy",
        out_dir / "harris_response.png",
        out_dir / "harris_response.npy",
        out_dir / "harris_corners.png",
        out_dir / "summary.png",
    ]
    for p in expected_files:
        assert p.is_file(), f"Expected artifact missing: {p}"

    # Verify npy data integrity
    cs_loaded = np.load(str(out_dir / "corner_score.npy"))
    hr_loaded = np.load(str(out_dir / "harris_response.npy"))
    assert cs_loaded.shape == res["grayscale"].shape
    assert hr_loaded.shape == res["grayscale"].shape
