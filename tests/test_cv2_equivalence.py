"""The numpy replacements must behave like the OpenCV calls they replace.

cv2 stays a test-only dependency for exactly this reason: it is the reference
implementation, but it is no longer shipped.
"""
import cv2
import numpy as np
import pytest

from pathlib import Path

from nrdoom import detector as D
from nrdoom.color import bgr_to_hsv
from nrdoom.label import label_blobs

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot

FIXTURES = ["golden_both_3bars.jpg", "real_alive_panel.png", "real_dusk_3bars.jpg",
            "real_sunset_3bars.png", "real_dark_overlay_3bars.png",
            "synth_both_1bar.png", "synth_both_2bars.png", "synth_mixed_3_1.png",
            "synth_no_rings.png", "synth_1bar_skyfill.png", "synth_2bars_skyfill.png"]

CHANNEL_VALUES = [0, 1, 2, 3, 32, 64, 85, 86, 127, 128, 129, 170, 200, 253, 254, 255]


def load(name):
    return shot(name)


def triples():
    v = np.asarray(CHANNEL_VALUES, dtype=np.uint8)
    cube = np.zeros((len(v), len(v), len(v), 3), np.uint8)
    cube[..., 0] = v[:, None, None]
    cube[..., 1] = v[None, :, None]
    cube[..., 2] = v[None, None, :]
    return cube.reshape(-1, 1, 3)


def _hue_gap(a, b):
    """Hue is circular: 0 and 179 are adjacent colours, so a linear difference
    would report the worst possible error for the closest possible pair."""
    d = np.abs(a.astype(int) - b.astype(int))
    return np.minimum(d, 180 - d)


def test_hsv_matches_cv2_on_every_channel_triple():
    """Exhaustive over a dense lattice of 4096 (B, G, R) triples."""
    cube = triples()
    got, ref = bgr_to_hsv(cube), cv2.cvtColor(cube, cv2.COLOR_BGR2HSV)
    assert np.array_equal(got[..., 2], ref[..., 2]), "V must be exact"
    assert int(_hue_gap(got[..., 0], ref[..., 0]).max()) <= 1
    assert int(np.abs(got[..., 1].astype(int) - ref[..., 1].astype(int)).max()) <= 1


def test_hsv_matches_cv2_on_random_noise():
    rng = np.random.default_rng(11)
    for _ in range(5):
        noise = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
        got, ref = bgr_to_hsv(noise), cv2.cvtColor(noise, cv2.COLOR_BGR2HSV)
        assert np.array_equal(got[..., 2], ref[..., 2]), "V must be exact"
        # Rounding ties land within one step; the mask test below is the real bar.
        assert int(_hue_gap(got[..., 0], ref[..., 0]).max()) <= 1
        assert int(np.abs(got[..., 1].astype(int) - ref[..., 1].astype(int)).max()) <= 1


def _lit(hsv):
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    flat = val.ravel()
    k = max(1, flat.size // 100)
    v_ref = int(np.partition(flat, -k)[-k])
    return ((hue >= D.HUE_LO) & (hue <= D.HUE_HI) & (sat >= D.SAT_MIN)
            & (val >= int(v_ref * D.LIT_V_FRAC)))


@pytest.mark.parametrize("name", FIXTURES)
def test_lit_mask_is_identical_to_the_opencv_path(name):
    """What actually matters: the classifier's per-pixel decision, not channel values."""
    img = load(name)
    band = img[:int(img.shape[0] * D.BAND_Y1_FRAC), :int(img.shape[1] * D.BAND_X_FRAC)]
    assert np.array_equal(_lit(bgr_to_hsv(band)),
                          _lit(cv2.cvtColor(band, cv2.COLOR_BGR2HSV)))


@pytest.mark.parametrize("name", FIXTURES)
def test_blobs_match_opencv_connected_components(name):
    img = load(name)
    band = img[:int(img.shape[0] * D.BAND_Y1_FRAC), :int(img.shape[1] * D.BAND_X_FRAC)]
    mask = _lit(bgr_to_hsv(band))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8) * 255, 8)
    ref = sorted([(int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_LEFT]),
                   int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]),
                   int(stats[i, cv2.CC_STAT_HEIGHT])) for i in range(1, n)], reverse=True)
    got = sorted([(b.area, b.x, b.y, b.w, b.h) for b in label_blobs(mask)], reverse=True)
    assert got == ref
    assert len(got) == len(ref)
    for b, _r in zip(label_blobs(mask), ref):
        assert 0 <= b.cx < band.shape[1] and 0 <= b.cy < band.shape[0]


@pytest.mark.parametrize("name", ["golden_both_3bars.jpg", "real_alive_panel.png",
                                  "real_dusk_3bars.jpg", "real_sunset_3bars.png",
                                  "real_dark_overlay_3bars.png"])
def test_row_strength_matches_opencv_sobel(name):
    """The portrait box is read with a hand-rolled Sobel; it must agree with cv2."""
    img = load(name)
    for cy in D.row_ys(img.shape):
        h, w = img.shape[:2]
        half = int(D.ROW_HALF_FRAC * h)
        box = img[int(cy) - half:int(cy) + half,
                  int(w * D.ROW_X_FRACS[0]):int(w * D.ROW_X_FRACS[1])]
        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        ref = float((np.abs(gx) + np.abs(gy) > D.ROW_EDGE_GRADIENT).mean())
        got = D.row_strength(img, cy)
        assert abs(got - ref) < 0.002, (name, cy, got, ref)


def test_blobs_on_empty_mask():
    assert label_blobs(np.zeros((20, 20), dtype=bool)) == []


def test_blobs_touching_diagonal_join():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0] = m[1, 1] = m[2, 2] = True
    assert [b.area for b in label_blobs(m)] == [3]


def test_blobs_keep_diagonally_separate_components_apart():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0] = True
    m[4, 4] = True
    assert sorted(b.area for b in label_blobs(m)) == [1, 1]
