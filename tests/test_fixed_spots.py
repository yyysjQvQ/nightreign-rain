"""The ally panel is anchored to screen width, so its geometry can be hardcoded.

Measured on the three captures that actually show gauges: x/W = 0.0562 / 0.0564 /
0.0570 and row centres y/W = 0.1085 / 0.1083 / 0.1086, which agree to 0.3% across
1366x768, 1280x695 and 2276x1280. Read against screen *height* the same three
spread over 0.193-0.200, which is not the HUD moving, it is the aspect ratio.
"""
from pathlib import Path

import cv2
import pytest

from nrdoom.detector import MAX_ROWS, gauge_spots, read_ring

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot

# Where the gauges really are, measured by hand on each capture: (name, [(cx, cy, r), ...])
MEASURED = [
    ("golden_both_3bars.jpg", [(128, 247, 27.0), (128, 358, 27.0)]),
    ("real_dusk_3bars.jpg", [(77, 148, 14.5), (77, 215, 14.3)]),
    ("real_sunset_3bars.png", [(73, 139, 16.2), (73, 202, 16.2)]),
]


def load(name):
    return shot(name)


@pytest.mark.parametrize("name,rings", MEASURED)
def test_the_fixed_spots_land_on_the_real_gauges(name, rings):
    spots = gauge_spots(load(name).shape)
    assert len(spots) == MAX_ROWS
    for (gx, gy, _gr), (rx, ry, _rr) in zip(spots, rings):
        assert abs(gx - rx) <= 2, f"{name} x is off"
        assert abs(gy - ry) <= 2, f"{name} row centre is off"


@pytest.mark.parametrize("name,rings", MEASURED)
def test_the_fixed_spots_read_three_bars_with_no_search(name, rings):
    """The whole point: no calibration, no refinement pass, just read there."""
    img = load(name)
    for spot in gauge_spots(img.shape):
        got = read_ring(img, *spot)
        assert got is not None, f"{name} reads nothing at {spot}"
        assert got.bars == 3, f"{name} read {got.bars} bars at {spot}"
        assert got.confidence >= 0.8, f"{name} confidence {got.confidence} at {spot}"


def test_the_radius_absorbs_the_ui_size_setting():
    """The in-game UI size moves the radius by +-10% and the read survives it,
    which is why only the centre has to be exact."""
    img = load("golden_both_3bars.jpg")
    cx, cy, r = gauge_spots(img.shape)[0]
    for dr in (-0.10, 0.0, 0.10):
        got = read_ring(img, cx, cy, r * (1 + dr))
        assert got is not None and got.bars == 3, dr
        assert got.confidence >= 0.8, (dr, got.confidence)


def test_a_spot_is_off_by_too_much_to_read_if_you_nudge_it_three_pixels():
    """Pins *why* the anchor has to be right: 3 px costs a third of the
    confidence and 6 px loses the gauge entirely."""
    img = load("golden_both_3bars.jpg")
    cx, cy, r = gauge_spots(img.shape)[0]
    assert read_ring(img, cx, cy + 6, r) is None
    assert read_ring(img, cx, cy + 3, r).confidence < 0.8


def test_there_is_only_one_row_grid():
    """The portrait boxes and the gauges are the same rows, so the two must not
    drift apart into separate constants."""
    from nrdoom.detector import row_ys

    img = shot("golden_both_3bars.jpg")
    assert row_ys(img.shape) == [cy for _cx, cy, _r in gauge_spots(img.shape)]


def test_the_letterboxed_capture_needs_no_search():
    """That 16:10 capture's panel sits 80 px below where the frame puts it, because
    the display letterboxes the picture and the HUD rides with the picture. Measured
    against the picture the grid lands on it, so counting needs no help."""
    from nrdoom.detector import count_rows, hud_rect, row_ys

    img = shot("real_alive_panel.png")
    rect = hud_rect(img)
    assert min(row_ys(img.shape)) < 340, "anchoring to the frame is what was wrong"
    assert min(row_ys(img.shape, rect=rect)) > 340
    assert count_rows(img, rect) == 2


NAMES = ["golden_both_3bars.jpg", "real_dusk_3bars.jpg", "real_sunset_3bars.png",
         "real_dark_overlay_3bars.png", "real_alive_panel.png"]


@pytest.mark.parametrize("name", NAMES)
def test_the_steady_read_only_converts_the_two_gauge_boxes(name):
    """The band used to be 14% x 36% of the frame, and the colour conversion of
    it is 17.6 ms of the 18 ms a read costs. Nothing outside these boxes is
    looked at any more."""
    from nrdoom.detector import band_rect, detect, detect_spots, spots_rect

    img = load(name)
    spots = gauge_spots(img.shape)
    x0, y0, x1, y1 = spots_rect(img.shape, spots)
    bx0, by0, bx1, by1 = band_rect(img.shape)
    assert (x1 - x0) * (y1 - y0) <= (bx1 - bx0) * (by1 - by0) / 5
    for cx, cy, r in spots:
        assert x0 <= cx - r and cx + r <= x1, f"{name}: {cx} r={r} outside the box"
        assert y0 <= cy - r and cy + r <= y1, f"{name}: {cy} r={r} outside the box"
    assert [None if g is None else (g.cx, g.cy, g.bars, round(g.confidence, 6))
            for g in detect_spots(img, spots)] == \
           [None if g is None else (g.cx, g.cy, g.bars, round(g.confidence, 6))
            for g in detect(img, spots)], name



def test_the_drawn_zone_covers_everything_the_count_reads():
    """Black out everything outside `rows_rect` and the count must not change.

    The preview draws that box as "只检测这一块", so a row counted from the health-bar
    strip outside it would have no explanation on screen when the user screenshots
    the preview to ask why.
    """
    import numpy as np

    from nrdoom.detector import count_rows, hud_rect, rows_rect

    img = load("real_alive_panel.png")
    rect = hud_rect(img)
    spots = gauge_spots(img.shape, rect=rect)
    x0, y0, x1, y1 = rows_rect(img.shape, spots)
    kept = np.zeros_like(img)
    kept[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    assert count_rows(img, rect, spots) == 2
    assert count_rows(kept, rect, spots) == 2


# -- teaching the geometry to find itself ------------------------------------
PEAK_TRUTH = [("golden_both_3bars.jpg", [247, 358]), ("real_dusk_3bars.jpg", [148, 215]),
              ("real_sunset_3bars.png", [139, 202]), ("real_dark_overlay_3bars.png", [271, 410])]


@pytest.mark.parametrize("name,rings", PEAK_TRUTH)
def test_the_portrait_peak_is_within_a_known_bias_of_the_gauge_centre(name, rings):
    """The peak only has to be *bounded*, not unbiased: it is what tells the
    engine the panel has moved, and the gauge itself is located by reading it.
    Measured worst case is 28 px (the second row, whose portraits are busier at
    the top), so anything much above that means the panel really did move."""
    from nrdoom.detector import REALIGN_OFF_FRAC_W, row_peaks

    img = load(name)
    for (peak, score), cy in zip(row_peaks(img), rings):
        assert abs(peak - cy) <= 30, f"{name}: peak {peak} vs gauge {cy}"
        assert score >= 0.20
        assert abs(peak - cy) < img.shape[1] * REALIGN_OFF_FRAC_W,             "a correct panel must not look like a moved one"


def test_an_absent_row_has_no_peak_worth_aligning_to():
    from nrdoom.detector import row_peaks

    peaks = row_peaks(load("synth_rows_duo.png"))
    assert peaks[0][1] >= 0.20
    assert peaks[1][1] < 0.20, "the second row has no ally, so nothing to lock onto"
