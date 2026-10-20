"""The game letterboxes on a non-16:9 display, and the HUD rides with it.

Measured on the user's own 2560x1600 screenshots: 80 px of pure black above and
below a 2560x1440 (exactly 16:9) picture, and the ally gauge sits at y=358 -
which is 80 + 0.1085 x 2560. Anchoring the geometry to the captured frame gives
278 instead, 80 px out, where the gauge only tolerates about 5.
"""
from pathlib import Path

import cv2
import pytest

from nrdoom.detector import count_rows, gauge_spots, hud_rect, read_ring

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot

# name, expected rows, expected bars per row, what the user said it shows
SHOTS = [
    ("real_duo_1bar.jpg", 1, [1], "一格倒地，只有一个队友"),
    ("real_duo_2bar.jpg", 1, [2], "两格倒地，只有一个队友"),
    ("real_duo_3bar.jpg", 1, [3], "三格倒地，只有一个队友"),
    ("real_alive_panel.png", 2, [None, None], "两个队友都活着"),
]

FULL_FRAME = ["golden_both_3bars.jpg", "real_dusk_3bars.jpg", "real_sunset_3bars.png"]


def load(name):
    return shot(name)


@pytest.mark.parametrize("name", ["real_duo_1bar.jpg", "real_duo_2bar.jpg",
                                  "real_duo_3bar.jpg", "real_alive_panel.png"])
def test_the_black_bands_are_found(name):
    """2560x1600 with a 2560x1440 picture in it: 80 px of black above and below.

    Asserted within a pixel or two because one dithered pixel at the edge of a bar
    ends the run there, and a gauge read tolerates five.
    """
    img = load(name)
    x0, y0, w, h = hud_rect(img)
    assert (x0, w) == (0, 2560), "the picture is as wide as the frame"
    assert abs(y0 - 80) <= 2, y0
    assert abs(h - 1440) <= 4, h


@pytest.mark.parametrize("name", FULL_FRAME)
def test_a_frame_that_is_not_letterboxed_is_its_own_content(name):
    img = load(name)
    assert hud_rect(img) == (0, 0, img.shape[1], img.shape[0])


@pytest.mark.parametrize("name,rows,bars,note", SHOTS)
def test_the_rows_land_on_the_real_gauges(name, rows, bars, note):
    img = load(name)
    spots = gauge_spots(img.shape, rect=hud_rect(img))
    got = [read_ring(img, *spot) for spot in spots[:rows]]
    assert [None if g is None else g.bars for g in got] == bars, f"{name} ({note})"


@pytest.mark.parametrize("name,rows,bars,note", SHOTS)
def test_the_number_of_allies_is_right(name, rows, bars, note):
    img = load(name)
    assert count_rows(img, hud_rect(img)) == rows, note


@pytest.mark.parametrize("name", FULL_FRAME)
def test_the_full_frame_cases_keep_working(name):
    img = load(name)
    assert count_rows(img, hud_rect(img)) == 2
    assert [r.bars for r in (read_ring(img, *s) for s in gauge_spots(img.shape, rect=hud_rect(img)))] == [3, 3]


def test_a_dark_scene_is_not_mistaken_for_a_letterbox():
    """Only uniform black counts as a bar; a dark corner of the world is 13-18,
    a real bar is 0."""
    img = load("real_duo_3bar.jpg")
    left = img.copy()
    left[:, :60] = 16                       # a dark, but not black, band
    assert hud_rect(left)[0] == 0
    black = img.copy()
    black[:, :60] = 0
    assert hud_rect(black)[0] == 60


def test_a_capture_with_an_alpha_channel_reads_the_same():
    """The desktop capture hands over BGRA, whose alpha is 255. Folding channels
    has to stop before it or no line is ever black and the bars go unseen."""
    img = load("real_duo_3bar.jpg")
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    assert hud_rect(bgra) == hud_rect(img)
    assert count_rows(bgra, hud_rect(bgra)) == 1


def test_a_few_bright_pixels_in_the_bar_do_not_move_the_geometry():
    """The user's own desktop recording: rows 24-40 of the 80 px bar carry a handful
    of bright pixels - the ShadowPlay indicator drawn in the letterbox. Two of 320
    samples ended the run, the bar measured 23 px instead of 80, and the grid missed
    a real three-bar gauge on 43 of 43 sampled frames. A bar is judged by the share
    of its line that is black, not by all of it."""
    img = load("real_duo_3bar.jpg")
    specked = img.copy()
    specked[24:41, 180:186] = 240            # a small bright badge in the bar
    specked[28, 900:912] = 230
    assert hud_rect(specked) == hud_rect(img), "the bar is still 80 px of black"
    assert count_rows(specked, hud_rect(specked)) == 1
    got = read_ring(specked, *gauge_spots(specked.shape, rect=hud_rect(specked))[0])
    assert got is not None and got.bars == 3


def test_a_real_scene_row_is_not_a_bar():
    """The other side of the same share test: the darkest content row measured 13 of
    a possible 255, so a line has to be black over half its width to count."""
    img = load("real_duo_3bar.jpg")
    scene = img.copy()
    scene[:60, :] = 6                        # nearly black, but a *scene*: every column
    assert hud_rect(scene)[1] == 0 or hud_rect(scene)[1] >= 60
    scene2 = img.copy()
    scene2[:60, :] = 6
    scene2[:60, ::2] = 200                   # half lit: not a bar by any reading
    assert hud_rect(scene2)[1] == 0
