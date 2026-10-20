import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from nrdoom.detector import (Ring, band_hsv, detect, detect_with_recovery,
                             propose_rings, read_ring, read_ring_recovering, refine_ring)

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot

# Measured on golden_both_3bars.jpg (2276x1280).
RING_A = (128, 247, 27)
RING_B = (128, 358, 27)

# Real captures from other sessions: a dusk run at 1366x768 and a sunset run at
# 1280x695. Both had every ally downed on all three segments. They differ from
# the golden fixture in colour cast, resolution and UI scale, which is the point.
REAL_SHOTS = [
    ("real_dusk_3bars.jpg", (77, 148, 14.5), (77, 215, 14.3)),
    ("real_sunset_3bars.png", (73, 139, 16.2), (73, 202, 16.2)),
]


def load(name):
    return shot(name)


def bars_at(name, ring):
    got = read_ring(load(name), *ring)
    return None if got is None else got.bars


@pytest.mark.parametrize(
    "name,expected",
    [
        ("golden_both_3bars.jpg", 3),
        ("synth_both_2bars.png", 2),
        ("synth_both_1bar.png", 1),
        ("synth_first_kept_1bar.png", 1),
    ],
)
def test_read_ring_counts_lit_arcs(name, expected):
    assert bars_at(name, RING_A) == expected
    assert bars_at(name, RING_B) == expected


def test_read_ring_handles_two_rings_with_different_counts():
    img = load("synth_mixed_3_1.png")
    assert read_ring(img, *RING_A).bars == 3
    assert read_ring(img, *RING_B).bars == 1


def test_read_ring_rejects_region_without_a_gauge():
    # No ally is downed here, so there is no donut to read at either slot.
    img = load("synth_no_rings.png")
    assert read_ring(img, *RING_A) is None
    assert read_ring(img, *RING_B) is None


def test_read_ring_rejects_the_purple_sky():
    """A patch of night sky that is as blue as a lit arc must not read as a gauge.

    Measured on the golden fixture: sky H 124 / S 148-171 versus lit arc
    H 121-126 / S 144-181. The search band now excludes this spot, so the test
    guards against someone widening the band back out and leaning on colour again.
    """
    img = load("golden_both_3bars.jpg")
    assert read_ring(img, 508, 187, 20) is None
    assert read_ring(img, 512, 192, 16) is None


def test_world_space_marker_over_a_downed_body_is_not_read():
    """Only the ally panel counts; the big gauge floating in the world does not."""
    img = load("golden_both_3bars.jpg")
    assert read_ring(img, 760, 735, 60) is None
    assert [round(r.cx) for r in propose_rings(img)] == [128, 128]


def test_band_covers_the_ally_panel_and_little_else():
    from nrdoom.detector import band_rect

    x0, y0, x1, y1 = band_rect((1280, 2276))
    assert (x0, y0) == (0, 128)
    assert x1 == 318 and y1 == 460
    area = (x1 - x0) * (y1 - y0)
    assert area / (1280 * 2276) < 0.06


@pytest.mark.parametrize("name", [n for n, _a, _b in REAL_SHOTS] + ["golden_both_3bars.jpg"])
def test_every_gauge_sits_inside_the_band(name):
    from nrdoom.detector import band_rect

    img = load(name)
    for ring in propose_rings(img):
        x0, y0, x1, y1 = band_rect(img.shape)
        assert ring.cx - ring.radius >= x0 and ring.cx + ring.radius <= x1
        assert ring.cy - ring.radius >= y0 and ring.cy + ring.radius <= y1


def test_read_ring_off_the_game_hud():
    img = load("golden_both_3bars.jpg")
    assert read_ring(img, 1100, 640, 30) is None
    assert read_ring(img, 40, 40, 27) is None


# -- duo mode: one ally -------------------------------------------------------
# synth_duo_one_ally.png keeps the second ally's portrait in place and only
# removes its gauge, which is the harder case than an empty row.
DUO_RING = (128, 247, 27)
DUO_EMPTY = (128, 358, 27)


def test_duo_finds_exactly_one_gauge():
    img = load("synth_duo_one_ally.png")
    got = propose_rings(img)
    assert [(round(r.cx), round(r.cy), r.bars) for r in got] == [(128, 247, 3)]


def test_duo_second_slot_reads_as_not_downed():
    """The detector must not invent a gauge over a portrait that has none."""
    img = load("synth_duo_one_ally.png")
    assert read_ring(img, *DUO_RING).bars == 3
    assert read_ring(img, *DUO_EMPTY) is None


def test_duo_survives_brightness_and_scale():
    img = load("synth_duo_one_ally.png")
    for gain in (0.75, 1.3):
        shifted = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        assert read_ring(shifted, *DUO_RING).bars == 3, gain
        assert read_ring(shifted, *DUO_EMPTY) is None, gain
    for scale in (0.7, 1.4):
        big = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        assert read_ring(big, *[v * scale for v in DUO_RING]).bars == 3, scale
        assert read_ring(big, *[v * scale for v in DUO_EMPTY]) is None, scale


@pytest.mark.parametrize("dx,dy,dr", [(0, 0, 0), (9, -7, 0), (-11, 4, 0), (0, 0, 7), (6, 6, -6)])
def test_refine_ring_recovers_a_bad_guess(dx, dy, dr):
    cx, cy, r = RING_A
    got = refine_ring(load("golden_both_3bars.jpg"), cx + dx, cy + dy, r + dr)
    assert got is not None
    assert math.hypot(got.cx - cx, got.cy - cy) <= 5
    assert abs(got.radius - r) <= 4
    assert got.bars == 3


@pytest.mark.parametrize("scale", [0.6, 0.84, 1.2, 1.7])
def test_geometry_scales_with_resolution(scale):
    img = load("golden_both_3bars.jpg")
    big = cv2.resize(img, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    for cx, cy, r in (RING_A, RING_B):
        got = read_ring(big, cx * scale, cy * scale, r * scale)
        assert got is not None and got.bars == 3, f"scale={scale} ring={(cx, cy, r)}"


@pytest.mark.parametrize("gain", [0.7, 0.85, 1.15, 1.4])
def test_survives_game_brightness_changes(gain):
    img = load("golden_both_3bars.jpg")
    shifted = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    for ring in (RING_A, RING_B):
        got = read_ring(shifted, *ring)
        assert got is not None and got.bars == 3, f"gain={gain} ring={ring}"


def test_propose_rings_finds_both_hud_gauges():
    got = propose_rings(load("golden_both_3bars.jpg"))
    for want in (RING_A, RING_B):
        hits = [r for r in got if math.hypot(r.cx - want[0], r.cy - want[1]) <= 10
                and abs(r.radius - want[2]) <= 6]
        assert hits, f"no proposal near {want}, got {got}"
        assert hits[0].bars == 3


def test_propose_rings_ignores_sky_and_offscreen_markers():
    got = propose_rings(load("golden_both_3bars.jpg"))
    assert len(got) == 2, f"expected exactly the two HUD gauges, got {got}"
    assert all(r.cx < 400 and r.cy < 500 for r in got)


def test_propose_rings_returns_nothing_when_nobody_is_downed():
    assert propose_rings(load("synth_no_rings.png")) == []


def test_ring_is_a_plain_value_object():
    r = Ring(1.0, 2.0, 3.0, 3, 0.9)
    assert r.bars == 3
    with pytest.raises(Exception):
        r.bars = 2


def test_detector_accepts_a_bgra_frame():
    """The capture path hands over BGRA; dropping alpha cost 32 ms a frame."""
    img = load("golden_both_3bars.jpg")
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    assert [r.bars for r in detect(bgra, [RING_A, RING_B])] == [3, 3]
    assert [r.bars for r in detect(img, [RING_A, RING_B])] == [3, 3]


# -- real captures from other sessions ----------------------------------------
def test_a_gauge_over_a_purple_sky_still_reads():
    """The spokes are judged by *not being gauge-coloured*, and a purple Nightreign
    sky is gauge-coloured: gap darkness fell to 0.639 against the 0.70 gate and the
    whole ring was thrown away, on a frame where all three segments are lit.

    It is now accepted on being a complete ring - three slots lit, annulus filled,
    hub empty - which is a statement about the sprite rather than about colour."""
    assert bars_at("real_purple_sky_gauge.png", (108.5, 208.3, 22.7)) == 3


def test_a_gauge_survives_motion_blur():
    """A fast camera turn smears the spokes into the arcs, the same failure by
    another route: 15 px of horizontal blur on the user's own 2560x1600 capture took
    gap darkness to 0.694 and lost the read."""
    img = load("real_duo_3bar.jpg")
    k = 15
    ker = np.zeros((k, k), np.float32)
    cv2.line(ker, (0, k // 2), (k - 1, k // 2), 1.0, 1)
    got = read_ring(cv2.filter2D(img, -1, ker / ker.sum()), 144.6, 357.8, 30.2)
    assert got is not None and got.bars == 3


@pytest.mark.parametrize("dy", [4, 5, 6, 8])
def test_the_relaxation_does_not_swallow_the_drift_signal(dy):
    """The gate was colour-based for a reason: "the first read failed" is how
    `detect_with_recovery` learns the stored position has drifted, and a ring 6 px
    off centre used to score the same as one over a purple sky. It no longer does,
    because a ring that is off centre cannot fill all three slots - which is also
    why only a *complete* ring is excused, and no partially lit sprite can borrow
    the relaxation."""
    img = load("golden_both_3bars.jpg")
    assert read_ring(img, 128, 247 + dy, 27) is None, \
        "a nudged gauge must still fail, or drift stops being detectable"


@pytest.mark.parametrize("name,a,b", REAL_SHOTS)
def test_reads_three_bars_on_real_captures(name, a, b):
    img = load(name)
    assert read_ring(img, *a).bars == 3, a
    assert read_ring(img, *b).bars == 3, b


def test_auto_locate_works_on_a_dark_scene_at_native_resolution():
    """A real 2560x1600 Roundhold capture, far darker than any other fixture.

    The lit gauges are the golden fixture's own panel pasted in, so this is a
    composite rather than a screenshot of a real doomed run, but the backdrop,
    resolution and brightness are genuine. Auto-locate must not need a bright
    scene to find the panel.
    """
    img = load("real_dark_overlay_3bars.png")
    assert float(img.mean()) < 60
    rings = [(144, 271, 31), (144, 410, 31)]
    got = propose_rings(img)
    assert [r.bars for r in got] == [3, 3], got
    for want in rings:
        assert any(math.hypot(r.cx - want[0], r.cy - want[1]) <= 4 for r in got), want
    for ring in rings:
        assert read_ring(img, *ring).bars == 3, ring


@pytest.mark.parametrize("name,a,b", REAL_SHOTS)
def test_propose_finds_exactly_the_two_gauges_on_real_captures(name, a, b):
    got = propose_rings(load(name))
    assert [r.bars for r in got] == [3, 3], got
    for want in (a, b):
        assert any(math.hypot(r.cx - want[0], r.cy - want[1]) <= 3 for r in got), want


@pytest.mark.parametrize("name,a,b", REAL_SHOTS)
def test_real_captures_survive_brightness_and_scale(name, a, b):
    img = load(name)
    for gain in (0.7, 1.35):
        shifted = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        for ring in (a, b):
            got = read_ring(shifted, *ring)
            assert got is not None and got.bars == 3, f"{name} gain={gain} {ring}"
    for scale in (0.7, 1.4):
        big = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        for cx, cy, r in (a, b):
            got = read_ring(big, cx * scale, cy * scale, r * scale)
            assert got is not None and got.bars == 3, f"{name} scale={scale} {(cx, cy, r)}"


# -- drift recovery ------------------------------------------------------------
# Where the ally portraits and the level badge sit in real_alive_panel.png
# (2560x1600, both allies standing). This is the only real capture of a
# not-downed state, and the stop-playing rule depends on it reading as None.
ALIVE_SLOTS = {"Saku portrait": (116, 350, 42), "mortis portrait": (116, 482, 42),
               "level badge": (137, 150, 42)}


def test_real_alive_panel_reads_as_nobody_downed():
    img = load("real_alive_panel.png")
    assert propose_rings(img) == []
    for where, ring in ALIVE_SLOTS.items():
        assert read_ring(img, *ring) is None, where


def test_real_alive_panel_stays_none_through_drift_recovery():
    """The armed loop uses the recovering read, so that must not invent a gauge.

    Requirement "stop the music when an ally is no longer downed" only works if
    a standing ally reliably reads as None rather than as a stray 1-segment gauge.
    """
    img = load("real_alive_panel.png")
    readings, drifted = detect_with_recovery(img, list(ALIVE_SLOTS.values()))
    assert readings == [None, None, None]
    assert drifted == []


def test_recovering_read_absorbs_a_translated_position():
    """A stored centre that is a few pixels off must still read.

    Translation is not rotation, so the phase search inside _read cannot fix it:
    at this gauge size the sampling annulus is about 3 px thick.
    """
    hsv = band_hsv(load("real_dusk_3bars.jpg"))
    got, drifted = read_ring_recovering(hsv, 77 + 3, 148 + 3, 14.5)
    assert got is not None and got.bars == 3
    assert drifted is True


def test_recovering_read_reports_no_drift_when_the_calibration_is_exact():
    hsv = band_hsv(load("real_dusk_3bars.jpg"))
    got, drifted = read_ring_recovering(hsv, 77, 148, 14.5)
    assert got.bars == 3
    assert drifted is False


def test_recovering_read_gives_up_quietly_on_empty_hud():
    hsv = band_hsv(load("synth_no_rings.png"))
    got, drifted = read_ring_recovering(hsv, 400, 400, 27)
    assert got is None
    assert drifted is False


def test_detect_with_recovery_reports_which_slots_moved():
    img = load("real_dusk_3bars.jpg")
    slots = [(77, 148, 14.5), (77 + 3, 215 + 3, 14.3)]
    readings, drifted = detect_with_recovery(img, slots)
    assert [r.bars for r in readings] == [3, 3]
    assert drifted == [1]


def test_nobody_downed_does_not_trigger_the_jitter_search(monkeypatch):
    """A guard, not a benchmark.

    An ally who is simply alive makes every read fail, and sweeping ~121 offsets
    for each of them every frame cost 88 ms and dropped the loop to 11 fps. The
    blue gate is what stops that, so pin the number of reads rather than the
    elapsed time.
    """
    from nrdoom import detector as mod

    calls = []
    real = mod._read
    monkeypatch.setattr(mod, "_read",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    readings, drifted = detect_with_recovery(load("golden_both_3bars.jpg"),
                                             [(400, 400, 27), (400, 600, 27)])
    assert readings == [None, None]
    assert drifted == []
    assert len(calls) == 2


def test_drifted_slot_does_search_and_finds_the_gauge(monkeypatch):
    from nrdoom import detector as mod

    calls = []
    real = mod._read
    monkeypatch.setattr(mod, "_read",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    readings, drifted = detect_with_recovery(load("golden_both_3bars.jpg"),
                                             [(131, 250, 27)])
    assert readings[0].bars == 3
    assert drifted == [0]
    assert len(calls) > 1


@pytest.mark.parametrize("name,cx,cy,radius,want", [
    ("golden_both_3bars.jpg", 128, 247, 27, True),
    ("real_dusk_3bars.jpg", 77, 148, 14.5, True),
    ("real_sunset_3bars.png", 73, 139, 16.2, True),
    ("synth_both_1bar.png", 128, 247, 27, True),
    ("synth_no_rings.png", 128, 247, 27, False),
    ("synth_no_rings.png", 400, 400, 27, False),
])
def test_blue_gate_separates_a_gauge_from_an_empty_slot(name, cx, cy, radius, want):
    from nrdoom.detector import _band_lit, _blue_around

    lit = _band_lit(band_hsv(load(name)))
    assert _blue_around(lit, cx, cy, radius) is want


def test_a_mask_too_dark_to_judge_gates_the_search_too():
    """"Cannot tell" has to mean "do not go looking", not "no gate". Getting this
    backwards costs 88 ms a frame the moment the scene is dark."""
    from nrdoom import detector as mod

    img = load("golden_both_3bars.jpg")
    slots = [(400, 400, 27), (400, 600, 27)]
    x0, y0, x1, y1 = mod.spots_rect(img.shape, slots)
    hsv = mod._hsv_of(img[y0:y1, x0:x1])
    assert mod._band_lit(hsv) is None, "this crop is too dark to judge either way"
    cx, cy, r = slots[1]
    gated, _ = mod.read_ring_recovering(hsv, cx - x0, cy - y0, r, lit=None)
    ungated, _ = mod.read_ring_recovering(hsv, cx - x0, cy - y0, r)
    assert gated is None
    assert ungated is not None, "without a mask it does search, which is the difference"


# -- a window that only grazes the ring is not a reading ----------------------
# Live capture from the user's own 2560x1600 run (diagnostics/20261018-150603),
# cropped to the HUD corner. The ally is downed on all three segments, so the
# truth where the panel actually is reads 3.
LIVE_SHOT = "live_downed_3bars_2560.png"
LIVE_RING = (145, 276, 30)


def _read_at(img, cy):
    from nrdoom.detector import detect_spots

    cx, _, r = LIVE_RING
    return detect_spots(img, [(cx, cy, r)])[0]


def test_the_live_ring_reads_three_where_the_panel_actually_is():
    ring = _read_at(load(LIVE_SHOT), LIVE_RING[1])
    assert ring is not None and ring.bars == 3
    assert ring.confidence > 0.9


def test_reading_13px_below_the_ring_does_not_report_two_bars():
    """The failure the user saw: three segments down, the tool said two, and the
    song stayed off for twenty seconds.

    A window pushed below the real ring still catches its bottom arc, counts two
    lit slots, and reports that as a fact. The confidence term is what knows
    better - 0.002 against 1.0 on the true centre - so a read that low has to be
    "nothing here", not a count.
    """
    from nrdoom.detector import RING_MIN_CONFIDENCE

    img = load(LIVE_SHOT)
    grazing = _read_at(img, LIVE_RING[1] + 13)
    assert grazing is None, (grazing.bars, grazing.confidence)
    # and nothing the detector chooses to report may sit under the floor
    for dy in range(-16, 17):
        got = _read_at(img, LIVE_RING[1] + dy)
        if got is not None:
            assert got.confidence >= RING_MIN_CONFIDENCE, (dy, got.confidence)
