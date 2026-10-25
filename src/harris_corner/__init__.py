from harris_corner.detector import (
    corner_score,
    corner_score_taylor,
    detect_corners,
    harris_detector,
)
from harris_corner.filters import (
    box_sum,
    compute_sobel_derivatives,
    shift_zero_pad,
)


def main(*args, **kwargs):
    """Run the main Harris detection pipeline."""
    from harris_corner.main import main as _main

    return _main(*args, **kwargs)


__all__ = [
    "box_sum",
    "compute_sobel_derivatives",
    "corner_score",
    "corner_score_taylor",
    "detect_corners",
    "harris_detector",
    "main",
    "shift_zero_pad",
]
