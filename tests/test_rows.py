"""How many ally rows are on screen right now?

Calibration freezes the party size, but players leave mid-run and the whole list
compacts upward, so a stale second row would make the trigger wait forever for an
ally who no longer exists. Every ally row carries a portrait whatever state the
ally is in, so the rows are counted from that rather than from a gauge, which only
exists while somebody is downed.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from nrdoom.detector import (MAX_ROWS, RING_ABSENT_MAX, V_REF_FLOOR, ally_bar,
                             count_rows, gauge_spots, hud_rect, ring_presence,
                             row_strength, row_ys)

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot

# Row centres measured on each real capture, in pixels of the captured frame.
REAL = {
    "golden_both_3bars.jpg": [247, 358],
    "real_dusk_3bars.jpg": [148, 215],
    "real_sunset_3bars.png": [139, 202],
    "real_dark_overlay_3bars.png": [271, 410],
    "real_alive_panel.png": [356, 480],
}


def load(name):
    return shot(name)


@pytest.mark.parametrize("name", sorted(REAL))
def test_a_portrait_reads_strong_and_bare_background_reads_weak(name):
    img = load(name)
    rect = hud_rect(img)
    ys = REAL[name]
    pitch = ys[1] - ys[0]
    for y in ys:
        assert row_strength(img, y, rect) >= 0.20, f"{name} has an ally at y={y}"
    # Only the row directly below the panel is asserted: two rows down is where
    # other HUD lives (the player's own panel), and this signal is a coarse anchor
    # for finding the ally list, not a way to tell an ally from a rock face.
    assert row_strength(img, ys[1] + pitch, rect) < 0.20, \
        f"{name} has nobody below the last ally"


@pytest.mark.parametrize("name", sorted(REAL))
def test_every_real_capture_is_a_trio_with_two_rows(name):
    img = load(name)
    if name == "real_dark_overlay_3bars.png":
        pytest.skip("that capture has the game's own black bands, see below")
    assert count_rows(img, hud_rect(img)) == 2, name


def test_a_panel_the_grid_misses_still_counts_at_its_own_rows():
    """The one capture whose HUD rides with the frame, not with the black bands
    inside it - the game drew those itself, and the in-game UI size is bigger
    too. Its rows are found by the aligner, not predicted, so this proves the row
    signal itself works there."""
    img = load("real_dark_overlay_3bars.png")
    rect = hud_rect(img)
    spots = [(144.3, 271.0, 31.4), (144.2, 410.3, 31.4)]
    assert count_rows(img, rect, spots) == 2


def test_row_positions_are_predicted_from_the_rendered_width():
    """The rows sit where the calibration found them, so the guess has to agree."""
    ys = row_ys(load("golden_both_3bars.jpg").shape)
    assert abs(ys[0] - 247) <= 12
    assert abs(ys[1] - 358) <= 12
    assert len(ys) == MAX_ROWS


def test_one_row_when_the_bottom_ally_left():
    assert count_rows(load("synth_rows_duo.png")) == 1


def test_no_rows_when_the_last_ally_left():
    assert count_rows(load("synth_rows_solo.png")) == 0


def test_a_blank_frame_has_no_rows():
    assert count_rows(np.zeros((1280, 2276, 3), dtype=np.uint8)) == 0


def test_a_gauge_in_a_lower_row_pulls_the_row_above_it_in():
    """The list compacts from the top, so an ally below proves one above.

    This is what counts a standing ally on nearly no health, whose own bar is too
    short to see. The only way to get the state below is to paste a real gauge under
    an empty row, which the game never does; if a false gauge read ever did, the
    answer withholds the music rather than playing it early, which is the direction
    that is safe. A bare row below the last ally still ends the count (see
    test_one_row_when_the_bottom_ally_left).
    """
    solo = load("synth_rows_solo.png")
    gold = load("golden_both_3bars.jpg")
    y1, y2 = row_ys(solo.shape)
    x0, x1 = int(gold.shape[1] * 0.03), int(gold.shape[1] * 0.08)
    solo[int(y2 - 30):int(y2 + 30), x0:x1] = gold[int(y1 - 30):int(y1 + 30), x0:x1]
    assert row_strength(solo, y2) > 0.20
    assert count_rows(solo) == 2


def test_the_health_bar_is_seen_for_a_standing_ally_and_not_for_a_downed_one():
    """What the stop decision rests on: "no gauge but a bar" has to mean standing.

    The plate behind a downed ally is maroon and a Nightreign sky goes pink, both at
    the top of the hue wheel where red wraps, so the test takes only the low end.
    Measured on the captures: the bar itself sits at hue 4-10 and sat 200+, the plate
    at hue 168-179 and sat 97-161.
    """
    from nrdoom.detector import ally_bar, gauge_spots

    standing = load("real_alive_panel.png")
    rect = hud_rect(standing)
    for cx, cy, _r in gauge_spots(standing.shape, rect=rect):
        assert ally_bar(standing, cy, rect) is True, "both allies are on their feet"

    downed = load("real_duo_3bar.jpg")
    rect = hud_rect(downed)
    assert ally_bar(downed, gauge_spots(downed.shape, rect=rect)[0][1], rect) is False
    assert ally_bar(load("synth_rows_solo.png"), row_ys(load("synth_rows_solo.png").shape)[0]) is False


def test_a_short_fill_still_counts_as_a_health_bar():
    """A revived ally comes back with little in him: Sung's fill was 0.025 W of a
    0.066 W bar. Length alone cannot be the test - the fill's left end is, and it
    sits at 0.082 W of the picture on both the 2560 and the 1920 captures."""
    img = load("real_low_hp_standing.png")
    rect = hud_rect(img)
    assert ally_bar(img, row_ys(img.shape, rect=rect)[0], rect) is True


def test_a_red_backdrop_does_not_make_a_downed_ally_look_standing():
    """The lava at 0.071 W of saturated red is wider than the bar itself, and it
    bleeds left of where a fill can start. A real bar starts at the trough's left
    end and has plate to the left of it."""
    img = load("real_red_backdrop_downed.png")
    rect = hud_rect(img)
    for cy in row_ys(img.shape, rect=rect):
        assert ally_bar(img, cy, rect) is False


def test_a_row_above_a_demonstrated_ally_is_an_ally_too():
    """The list is contiguous, so an ally below settles the row above it.

    Sung is standing at about 40% health: no gauge, and the remaining red fill is
    0.025 W, under the bar test's 0.030 W. Counting must not conclude from that
    that the party has one fewer member than the row below it proves.
    """
    img = load("real_low_hp_standing.png")
    rect = hud_rect(img)
    assert count_rows(img, rect) == 2


def test_rows_are_counted_at_the_calibrated_positions_too():
    """A HUD scaled differently from the default still counts its own rows."""
    img = load("golden_both_3bars.jpg")
    assert count_rows(img, spots=[(128, 247, 27), (128, 358, 27)]) == 2
    assert count_rows(img, spots=[(128, 247, 27)]) == 1


# -- "no gauge is being drawn here" as a positive reading --------------------
# The stop rule needs to tell a downed ally whose gauge it cannot parse from an
# ally who is on his feet. The share of the annulus that is lit does that where
# the full read cannot, because the full read also applies the spoke-darkness and
# centre-hole gates that a gauge over a purple sky fails.
CASES = [
    ("real_duo_3bar.jpg", 0, 0.80, 1.00, "three segments"),
    ("real_duo_2bar.jpg", 0, 0.40, 0.80, "two segments"),
    ("real_duo_1bar.jpg", 0, 0.15, 0.45, "one segment"),
    ("real_purple_sky_gauge.png", 0, 0.60, 1.00, "the gauge the spoke gate rejects"),
    ("real_alive_panel.png", 0, 0.00, RING_ABSENT_MAX, "standing"),
    ("real_alive_panel.png", 1, 0.00, RING_ABSENT_MAX, "the other one standing"),
    ("real_low_hp_standing.png", 0, 0.00, RING_ABSENT_MAX, "standing on low health"),
]


@pytest.mark.parametrize("name,slot,lo,hi,note", CASES)
def test_the_annulus_share_separates_downed_from_standing(name, slot, lo, hi, note):
    img = load(name)
    rect = hud_rect(img)
    cx, cy, r = gauge_spots(img.shape, rect=rect)[slot]
    share, v_ref = ring_presence(img, cx, cy, r)
    assert lo <= share <= hi, f"{name} slot {slot} ({note}): share {share}"
    assert v_ref >= V_REF_FLOOR, f"{name} slot {slot} is too dark to judge either way"


def test_a_panel_too_dark_to_judge_is_reported_as_such():
    """The empty second row of the duo capture has almost nothing on it; its
    brightness reference is what stops a share of zero meaning "standing"."""
    img = load("real_duo_3bar.jpg")
    rect = hud_rect(img)
    cx, cy, r = gauge_spots(img.shape, rect=rect)[1]
    share, v_ref = ring_presence(img, cx, cy, r)
    assert v_ref < V_REF_FLOOR, (share, v_ref)


def test_a_spot_off_the_edge_reports_nothing_rather_than_no_ring():
    img = load("real_duo_3bar.jpg")
    assert ring_presence(img, 5, 5, 30) is None

