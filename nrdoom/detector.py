"""Read the Nightreign downed-gauge rings out of a captured game frame.

Pure numpy/OpenCV, no Qt and no capture imports, so it can be driven from saved
frames in the test suite.

Why this is shape-based rather than colour-based: the night sky in Nightreign is
a saturated blue-violet whose hue and saturation overlap a lit gauge arc almost
completely (measured on the golden fixture: sky H 124 / S 148-171 versus lit arc
H 121-126 / S 144-181). Roughly half the frame passes a naive "is this the blue
of a lit arc" test, and the level badge in the top-left corner is a genuine blue
donut on top of that. Colour classifies pixels; the slot geometry decides
whether a location is a gauge.

The gauge is a static HUD sprite, so its three slots sit at fixed angles
(measured on the golden fixture: starts 100/220/341 degrees, each spanning 100,
separated by ~18 degree dark spokes). A candidate only counts if the spokes stay
dark and its centre stays empty, which is exactly what a blob of sky fails.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .color import bgr_to_hsv
from .label import label_blobs

# --- pixel classifier -------------------------------------------------------
HUE_LO, HUE_HI = 108, 142      # OpenCV hue, 0-180 scale
SAT_MIN = 100                  # the dark red ally panel sits well below this
V_REF_FLOOR = 70               # below this the ring is simply not lit at all
LIT_V_FRAC = 0.62              # lit when V >= 0.62 * that ring's own brightest band

# --- gauge sprite geometry --------------------------------------------------
RADIAL_FRACS = (0.66, 0.74, 0.82, 0.90)
LIT_RADIAL_MIN = 2             # of the 4 radial samples at one angle
HOLE_FRACS = (0.12, 0.26, 0.40)
HOLE_LIT_MAX = 0.35            # a blob has a blue centre, a gauge does not
SLOT_STARTS = (100, 220, 341)
SLOT_SPAN = 100
SLOT_MARGIN = 18               # sample slot interiors, away from the sprite's own edges
GAP_MARGIN = 4
LIT_SLOT_FRAC = 0.60
MAX_BARS = 3

# The separators between slots are the only structure that is dark no matter how
# many segments are lit, so they are used to recover the sprite's angular phase
# instead of assuming it. Without that, a centre that is off by one pixel rotates
# the sample grid enough to put bright arc edge where a separator should be, and a
# flawless three-segment read gets rejected.
#
# That last property is load-bearing in a second way: judging the spokes by
# brightness alone lets a ring nudged 6 px off centre read as a good one, and "the
# first read failed" is exactly how Engine.detect_with_recovery learns its stored
# position has drifted. The exemption below survives that because it asks for a
# complete ring, which an off-centre one cannot fake - measured radial fill 0.859 /
# 0.785 / 0.723 at +4 / +5 / +6 px against 1.000 for a true one. See
# tests/test_detector.py::test_the_relaxation_does_not_swallow_the_drift_signal.
PHASE_SEARCH_DEG = 12
GAP_DARK_MIN = 0.70
# A ring may fall as low as this on the spokes if it is otherwise complete: all
# three slots lit, the annulus filled, the hub empty.
GAP_DARK_RELAXED = 0.60
COMPLETE_FILL = 0.99

# Gate for the jitter search: a slot with nobody downed misses on the first read
# every frame, and searching for it unconditionally is far too expensive.
BLUE_GATE_FRAC = 0.06

# Below this share of the annulus is lit, no gauge is being drawn there at all.
# Measured: an ally on his feet is 0.000, one segment downed is 0.277-0.296, and
# the purple-sky gauge that the full read rejects still measures 0.844.
RING_ABSENT_MAX = 0.10

# --- HUD layout -------------------------------------------------------------
# Only the ally panel is ever analysed. Measured across three real captures the
# gauges sit at x 0.0564-0.0567 and rows at y 0.192-0.200 / 0.279-0.290, with a
# radius of 0.019-0.023 of screen height (the spread is the in-game UI size
# setting). The band below leaves roughly 1.5x margin on that. Keeping it tight
# both halves the per-frame cost and puts the purple sky, the world-space marker
# over a downed body, and the blue level badge outside the search area entirely.
BAND_X_FRAC, BAND_Y0_FRAC, BAND_Y1_FRAC = 0.14, 0.10, 0.36
MIN_BLOB_AREA_FRAC = 1e-5      # of the band; below this a blob is compression noise
PROPOSE_MIN_CONFIDENCE = 0.5

# What it takes to move the stored geometry onto an off-centre read. Deliberately
# far above what a real gauge needs: the jitter search exists to claw back a few
# pixels of drift, and a genuine gauge reads 1.0 on the first try and 0.72 three
# pixels out. Measured on the user's duo recording, where the second row is empty
# for all 277 s so every read there is a false positive by construction: nine
# ghosts, none above 0.55, against 841 frames of real three-bar reads at 1.0.
ADOPT_MIN_CONFIDENCE = 0.6

# Below this the "ring" is not a ring, whatever its slots look like. Confidence is
# `radial_fill * (hub empty) * (gaps dark)`, so it is the only term that speaks about
# the sprite rather than about the arcs, and the two ends are far apart: on the user's
# own 2560x1600 capture a true three-segment ring reads 1.00 at its centre and 0.69
# four pixels out, while the window thirteen pixels below - which still catches the
# bottom arc and counts two segments - reads 0.002. Every real read on the fixture
# set is 0.33 or better (0.33 is a genuine one-segment ring), so the floor sits in the
# empty band between the two, not at either edge.
RING_MIN_CONFIDENCE = 0.15

# --- where the ally panel is -------------------------------------------------
# Players leave mid-run and the list compacts upward, so the party size cannot
# stay frozen at whatever was calibrated. Two things mark a row: the portrait box,
# whose edge density is used to *find* the panel when the in-game UI size setting
# moves it (see row_peaks), and the two things the HUD only ever draws for an
# ally, which decide whether a row is one (see count_rows).
ROW_X_FRACS = (0.036, 0.075)
ROW_HALF_FRAC = 0.0234         # half the portrait box, ~60 px at 1280 high
ROW_SEARCH_FRAC_W = 0.024      # how far to look for the panel; under half a row pitch
ROW_MIN_STRENGTH = 0.20        # portrait edge density that means "something is here"
ROW_EDGE_GRADIENT = 90         # |gx|+|gy| that counts as sitting on an edge
MAX_ROWS = 2

# --- what makes a row an ally -------------------------------------------------
# Edge density alone cannot tell a portrait from a sunlit rock face: measured on
# the real duo captures, the portrait scored 0.38 and the rock below it 0.43. So a
# row counts when the HUD draws something only an ally gets - their health bar, or
# a gauge, which is up only while they are downed.
# The bar is looked for by where its fill *starts*, not by how much red happens to
# be in the row: the fill grows rightwards from the trough's left end, at 0.082 of
# the picture width on both the 2560 and the 1920 captures (0.0820, 0.0823, 0.0833).
# Against that, an ally revived on a third of his health shows 0.025 W of fill and
# passes, while lava of the same hue and saturation floods 0.071 W and reaches left
# of where a fill can start, so it fails on both halves of the test.
HP_LEFT_FRAC_W = 0.082         # where the fill begins
HP_START_TOL_FRAC_W = 0.0035   # ... within this (measured spread: 0.0820, 0.0823, 0.0833)
HP_BAND_TOP_FRAC_W = 0.0098    # the bar sits above the row centre, at cy-0.0098W..cy-0.0015W
HP_BAND_BOTTOM_FRAC_W = 0.0015
HP_MIN_FILL_FRAC_W = 0.006     # a sliver is enough, as long as it starts in place
HP_MIN_ROWS = 3                # measured band: 8-9 rows at 1920, 13 at 2560
HP_HUE_MAX = 12                # red-orange, and deliberately NOT the 168-179 end:
# the ally plate and a pink Nightreign sky both sit there (measured hue 168-178, sat
# 97-161 over a downed ally), while the health bar itself measures hue 4-10.
HP_SAT_MIN = 130               # the bar sits at 200+; the plate that fooled it, 150
HP_VAL_MIN = 60

# A portrait peak this far from a spot means the panel is not where the grid
# says, rather than that the peak is biased: the measured bias tops out at 28 px
# on a correctly placed panel, and the one capture with a larger in-game UI size
# sits 58 px out.
REALIGN_OFF_FRAC_W = 0.018

# How far the agreed position has to be from where we are looking before it is
# worth moving for. This is *not* the same number as the one above: that one is
# about trusting a portrait peak, this one is about the gauge read.
#
# Measured by sweeping the window off a real ring (`tools/offset_read.py`, and the
# user's own 2560x1600 frame in fixtures/live_downed_3bars_2560.png): a gauge reads
# at 4 px out and is gone by 5, and 8-13 px out it reports a *false* count - two
# segments of a three-segment ring, at confidence 0.002. So the threshold has to be
# under the point where a wrong count starts being believed, which says 5 px.
#
# It cannot go under the portrait peak's own bias, though. On
# fixtures/real_red_backdrop_downed.png the peaks sit 4.8 px off a panel that is
# already correct, and a threshold below that moves the window off it and reads
# [2, 2] where the truth is [1, 2] - `tools/wander_probe.py` checks exactly this.
# 0.0025 is the largest value that is both under the 5 px read cliff and over that
# bias: 6.4 px at 2560 wide, 4.8 px at 1920. The band left between the two is covered
# by the jitter search inside `detect_with_recovery`, which re-centres a ring it can
# still see; `tools/settle_band.py` sweeps the whole 0-16 px range to check that.
REALIGN_SETTLE_FRAC_W = 0.0025

# --- the letterbox ----------------------------------------------------------
# A display that is not 16:9 gets black bars. Measured on the user's 2560x1600
# screenshots: 80 px of pure black above and below a 2560x1440 picture, and the
# ally gauge sits at y=358, which is 80 + 0.1085 x 2560. Anchoring to the
# captured frame instead gives 278 - 80 px out, where a gauge read tolerates
# about 5. A bar is 0 in every channel; the darkest corner of a real scene
# measured 13 even after the channels were folded.
LETTER_BLACK_MAX = 8
LETTER_BLACK_SHARE = 0.5         # of a line's samples; a badge in the bar is a handful
LETTER_SCAN_FRAC = 0.25          # how far from each edge to look for a bar
LETTER_SAMPLE_STEP = 8           # a bar is black across every pixel, so sampling is safe
LETTER_MAX_SHARE = 0.25          # beyond this the "bar" is a dark scene, not letterboxing

# --- the panel is anchored to the rendered width ----------------------------
# Measured on the captures that show gauges (1366x768, 1280x695, 2276x1280 and
# four 2560x1600 shots): x = 0.0562-0.0570 of the rendered width and the first
# row centre 0.1083-0.1086 below its top. Only the radius varies with the in-game
# UI size setting (0.0106-0.0127 of width), and the read tolerates +-10% of it,
# so the centre is the part that has to be exact.
GAUGE_X_FRAC_W = 0.0565
ROW0_Y_FRAC_W = 0.1085
ROW_PITCH_FRAC_W = 0.0489
GAUGE_R_FRAC_W = 0.0118


def _run_of(flags):
    """How many entries at the start are True."""
    if not flags.any():
        return 0
    return int(len(flags) if flags.all() else np.argmax(~flags))


def _dark_lines(slice_bgr, axis: int):
    """Per line along one edge: is the bulk of it pure black?

    Judged by the share of the line, not by all of it. The user's own recording had
    the ShadowPlay indicator sitting inside the letterbox - two bright pixels in 320
    - and an all-or-nothing test cut an 80 px bar to 23, which moved the whole grid
    57 px and lost every gauge read. Measured: bar rows are 0.91-1.00 black, content
    rows 0.01-0.05.

    A bar is black in all three channels, so the channels fold first and no luma
    conversion is needed (0.65 ms against 4.1 for one). And a bar is black across its
    whole line, so a line is sampled *across* it, not along it.
    """
    folded = np.maximum.reduce((slice_bgr[..., 0], slice_bgr[..., 1],
                                slice_bgr[..., 2]))
    return (folded <= LETTER_BLACK_MAX).mean(axis=axis) >= LETTER_BLACK_SHARE


def hud_rect(frame):
    """(x, y, width, height) of the game's own picture inside the captured frame.

    The HUD is laid out against that picture, not against whatever the capture
    added around it, so every geometry number here is measured from this rect. A
    frame with nothing to trim gives the frame back.
    """
    h, w = frame.shape[:2]
    s = LETTER_SAMPLE_STEP
    band = max(s, int(min(h, w) * LETTER_SCAN_FRAC))
    cap = int(min(h, w) * LETTER_MAX_SHARE)
    top = min(_run_of(_dark_lines(frame[:band, s::s], 1)), cap)
    bottom = min(_run_of(_dark_lines(frame[h - band:h, s::s][::-1], 1)), cap)
    left = min(_run_of(_dark_lines(frame[::s, :band], 0)), cap)
    right = min(_run_of(_dark_lines(frame[::s, w - band:w][:, ::-1], 0)), cap)
    if w - left - right < w * 0.5 or h - top - bottom < h * 0.5:
        return (0, 0, w, h)
    return (left, top, w - left - right, h - top - bottom)


def _rect_of(shape, rect):
    """A rect is (x, y, w, h) of the game's own picture; None means the frame."""
    if rect is not None:
        return rect
    return (0, 0, shape[1], shape[0])


def gauge_spots(shape, n: int = MAX_ROWS, rect=None):
    """Where the ally gauges always are: (cx, cy, radius) per row, top to bottom.

    Absolute pixels in the captured frame, so pass the `rect` you measured from
    that same frame - on a 2560x1600 capture the panel rides 80 px down with the
    letterbox.
    """
    x0, y0, w, _h = _rect_of(shape, rect)
    return [(x0 + w * GAUGE_X_FRAC_W,
             y0 + w * (ROW0_Y_FRAC_W + ROW_PITCH_FRAC_W * i),
             w * GAUGE_R_FRAC_W) for i in range(n)]


def band_rect(shape):
    """Pixel bounds of the region that is searched and read: (x0, y0, x1, y1)."""
    h, w = shape[:2]
    return (0, int(h * BAND_Y0_FRAC), int(w * BAND_X_FRAC), int(h * BAND_Y1_FRAC))


@dataclass(frozen=True)
class Ring:
    cx: float
    cy: float
    radius: float
    bars: int
    confidence: float


def _slot_indices():
    slots, gaps = [], []
    for i, s in enumerate(SLOT_STARTS):
        slots.append((s + SLOT_MARGIN + np.arange(SLOT_SPAN - 2 * SLOT_MARGIN)) % 360)
        nxt = SLOT_STARTS[(i + 1) % len(SLOT_STARTS)]
        span = (nxt - s - SLOT_SPAN - 2 * GAP_MARGIN) % 360
        gaps.append((s + SLOT_SPAN + GAP_MARGIN + np.arange(span)) % 360)
    return slots, gaps


SLOT_IDX, GAP_IDX = _slot_indices()
GAP_ANGLES = np.concatenate(GAP_IDX)
_ANGLES = np.arange(360, dtype=np.float64)
_UNSET = object()          # "band mask not built yet", distinct from "too dark to build"


def _hsv_of(crop):
    """HSV of a small crop, accepting BGR or BGRA the way capture hands it over."""
    if crop.shape[2] == 4:
        crop = crop[:, :, :3]
    return bgr_to_hsv(crop)


def band_hsv(frame: np.ndarray) -> np.ndarray:
    """HSV of the HUD search band only; a 2276x1280 frame costs 46 ms otherwise.

    Only the one-off locator needs the whole band. The steady-state read uses
    :func:`detect_spots`, which converts a tenth of it.
    """
    # From the top of the frame, not from band_rect's y0: this is the region the
    # locator scans, and it deliberately sees the player's own HUD so it can
    # blank it afterwards. band_rect describes what gets read.
    return _hsv_of(frame[:int(frame.shape[0] * BAND_Y1_FRAC),
                        :int(frame.shape[1] * BAND_X_FRAC)])


def spots_rect(shape, spots):
    """Rectangle covering every spot's sampling circle with room to jitter."""
    h, w = shape[:2]
    pad = max(r for _cx, _cy, r in spots) * 1.5
    xs = [cx for cx, _cy, _r in spots]
    ys = [cy for _cx, cy, _r in spots]
    return (max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
            min(w, int(max(xs) + pad) + 1), min(h, int(max(ys) + pad) + 1))


def rows_rect(shape, spots):
    """Everything a row count looks at: the gauge spots plus the bar strip beside them.

    Wider than :func:`spots_rect`, and the preview has to say so - otherwise a
    bright red thing to the right of the panel that gets miscounted as an ally has
    no explanation on screen.
    """
    x0, y0, x1, y1 = spots_rect(shape, spots)
    if not spots:
        return x0, y0, x1, y1
    h, w = shape[:2]
    ys = [cy for _cx, cy, _r in spots]
    return (min(x0, int(w * (HP_LEFT_FRAC_W - HP_START_TOL_FRAC_W))),
            min(y0, int(min(ys) - w * HP_BAND_TOP_FRAC_W)),
            max(x1, int(w * 0.150)), max(y1, int(max(ys))))


def detect_spots(frame_bgr: np.ndarray, spots):
    """Read fixed spots, converting only the box around them.

    Same answers as :func:`detect` for a quarter of the pixel work, because the
    colour conversion of the whole band was 17.6 ms of the 18 ms a read cost.
    """
    spots = list(spots)
    if not spots:
        return []
    x0, y0, x1, y1 = spots_rect(frame_bgr.shape, spots)
    hsv = _hsv_of(frame_bgr[y0:y1, x0:x1])
    return [None if got is None else replace(got, cx=got.cx + x0, cy=got.cy + y0)
            for got in (_read(hsv, cx - x0, cy - y0, r) for cx, cy, r in spots)]


def _gather(hsv, cx, cy, radius, fracs):
    """Sample HSV at (len(fracs), 360) points around a circle. Out of frame -> None."""
    h, w = hsv.shape[:2]
    cos_a = np.cos(np.radians(_ANGLES))
    sin_a = np.sin(np.radians(_ANGLES))
    fr = np.asarray(fracs, dtype=np.float64)[:, None]
    xs = np.rint(cx + radius * fr * cos_a).astype(np.int64)
    ys = np.rint(cy - radius * fr * sin_a).astype(np.int64)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not inside.any():
        return None
    out = np.zeros(xs.shape + (3,), dtype=np.uint8)
    out[inside] = hsv[ys[inside], xs[inside]]
    return out


def _read(hsv, cx, cy, radius):
    if radius < 6:
        return None
    band = _gather(hsv, cx, cy, radius, RADIAL_FRACS)
    if band is None:
        return None
    hue, sat, val = band[..., 0].astype(int), band[..., 1].astype(int), band[..., 2].astype(int)
    v_ref = int(np.percentile(val, 95))
    if v_ref < V_REF_FLOOR:
        return None
    hue_ok = (hue >= HUE_LO) & (hue <= HUE_HI)
    litmat = hue_ok & (sat >= SAT_MIN) & (val >= int(v_ref * LIT_V_FRAC))
    flags = litmat.sum(axis=0) >= LIT_RADIAL_MIN

    dark_at = [1.0 - float(flags[(GAP_ANGLES + s) % 360].mean())
               for s in range(-PHASE_SEARCH_DEG, PHASE_SEARCH_DEG + 1)]
    best = int(np.argmax(dark_at))
    phase = best - PHASE_SEARCH_DEG
    gap_dark = dark_at[best]

    bars = sum(1 for s in SLOT_IDX
               if float(flags[(s + phase) % 360].mean()) >= LIT_SLOT_FRAC)
    if bars == 0:
        return None

    hole = 1.0
    centre = _gather(hsv, cx, cy, radius, HOLE_FRACS)
    if centre is not None:
        ch, cs, cv = (centre[..., i].astype(int) for i in range(3))
        hole_flags = ((ch >= HUE_LO) & (ch <= HUE_HI) & (cs >= SAT_MIN)
                      & (cv >= int(v_ref * LIT_V_FRAC))).sum(axis=0) >= 2
        hole = float(hole_flags.mean())
    if hole > HOLE_LIT_MAX:
        return None

    # How much of the sampled annulus is actually lit. Without this the radius is
    # unidentifiable: slots and spokes both stay correct over a wide annulus, so
    # a ring read at 34 px scores the same as one read at its true 27 px.
    slot_ang = np.concatenate([(s + phase) % 360 for s in SLOT_IDX])
    radial_fill = float(litmat[:, slot_ang].mean())

    # The spokes are the first thing this sprite loses: a purple Nightreign sky is
    # itself gauge-coloured (gap darkness 0.639 against the 0.70 gate) and 15 px of
    # motion blur smears them into the arcs (0.694), and both threw away a ring
    # whose three slots were fully lit and whose hub was entirely empty.
    #
    # Relaxing the gate on its own would also accept a ring nudged off centre, and
    # "the first read failed" is how drift is detected. It does not, because a ring
    # 4 px out cannot fill all three slots: measured fill 0.859, 0.785, 0.723 at
    # +4/+5/+6 px against 1.000 for a true one. So the exemption asks for a
    # complete ring, which is a statement about the sprite and not about colour.
    # Checked against the user's duo recording, where the second row is empty for
    # all 277 s: 0 false accepts in 5533 frames, and one real gauge recovered from
    # the Bilibili footage.
    complete = bars == MAX_BARS and radial_fill >= COMPLETE_FILL and hole == 0.0
    if gap_dark < GAP_DARK_MIN and not (complete and gap_dark >= GAP_DARK_RELAXED):
        return None

    confidence = radial_fill * (1.0 - hole / HOLE_LIT_MAX) * gap_dark
    if confidence < RING_MIN_CONFIDENCE:
        # The slots lit up, but nothing else about this is a ring. Counting it
        # anyway is how a window 13 px below a downed ally's ring reported "two
        # segments" and held that for twenty seconds.
        return None
    return Ring(cx=float(cx), cy=float(cy), radius=float(radius),
                bars=bars, confidence=round(float(confidence), 3))


def read_ring(frame_bgr: np.ndarray, cx: float, cy: float, radius: float):
    """Read one gauge by absolute position. None where there is no gauge."""
    return _read(band_hsv(frame_bgr), cx, cy, radius)


def detect(frame_bgr: np.ndarray, slots):
    """Read every calibrated slot in one frame. `slots` is ((cx, cy, radius), ...).

    Returns a list the same length as `slots`, with None where that ally is not
    downed. This is the steady-state hot path: one colour conversion of the band
    plus two ring samples, about 0.6 ms on a 2276x1280 frame.
    """
    slots = list(slots)
    hsv = band_hsv(frame_bgr)
    return [_read(hsv, cx, cy, radius) for cx, cy, radius in slots]


def detect_with_recovery(frame_bgr: np.ndarray, slots):
    """Read every slot, retrying off-centre where a stored position has drifted.

    Returns ``(readings, drifted_indices)``. Worth using only while armed: a slot
    with nobody downed misses on the first try every frame and so pays for the
    whole jitter search.

    Same narrow box as :func:`detect_spots`: the jitter search reaches 0.2 radii
    and the blue gate 1.3, both inside its 1.5 padding.
    """
    slots = list(slots)
    if not slots:
        return [], []
    x0, y0, x1, y1 = spots_rect(frame_bgr.shape, slots)
    hsv = _hsv_of(frame_bgr[y0:y1, x0:x1])
    # One mask for the frame, so a failed read can be gated without reading the
    # same spot twice. Building it costs about a tenth of one jitter sweep.
    lit = _band_lit(hsv)
    readings, drifted = [], []
    for i, (cx, cy, radius) in enumerate(slots):
        got, moved = read_ring_recovering(hsv, cx - x0, cy - y0, radius, lit)
        if moved:
            drifted.append(i)
        readings.append(None if got is None
                        else replace(got, cx=got.cx + x0, cy=got.cy + y0))
    return readings, drifted


def ring_presence(frame, cx: float, cy: float, radius: float):
    """How much of the gauge's annulus is lit, and how bright the ring is.

    A weaker, wider question than :func:`read_ring`: not "is this a readable
    three-segment gauge" but "is a gauge being drawn here at all". That matters
    because the full read applies gates (spoke darkness, centre hole) which a real
    gauge over a purple sky can fail - and then "no gauge" and "gauge I cannot
    parse" look identical to the caller. Measured on the fixtures, the lit share
    separates them cleanly: a downed ally reads 0.28 (one segment) to 0.84 (three,
    including the purple-sky case), while an ally on his feet reads 0.000.

    `v_ref` comes back too because it is the reason the share can be trusted: on a
    panel too dark to judge, every pixel passes the brightness test and the share
    saturates at 1.0 without meaning anything.

    Returns ``(share, v_ref)``, or ``None`` when the spot is off frame.
    """
    if radius < 6:
        return None
    h, w = frame.shape[:2]
    side = int(radius * 2.4) + 2
    x0, y0 = int(cx) - side // 2, int(cy) - side // 2
    if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
        return None
    hsv = _hsv_of(frame[y0:y0 + side, x0:x0 + side, :3])
    band = _gather(hsv, side / 2, side / 2, radius, RADIAL_FRACS)
    hue = band[..., 0].astype(int)
    sat = band[..., 1].astype(int)
    val = band[..., 2].astype(int)
    v_ref = int(np.percentile(val, 95))
    lit = ((hue >= HUE_LO) & (hue <= HUE_HI) & (sat >= SAT_MIN)
           & (val >= int(v_ref * LIT_V_FRAC)))
    return float(lit.mean()), v_ref


def _refine(hsv, cx, cy, radius, search, step):
    """Best donut near a guess, preferring positions and radii close to it.

    Confidence saturates at 1.0 across a small radius range, so without the
    distance term the search always collapses onto its smallest workable radius.
    """
    best = None
    winner = None
    offs = range(-search, search + 1, step)
    for dr in (-6, -3, 0, 3, 6):
        r = radius + dr
        if r < 8:
            continue
        for dy in offs:
            for dx in offs:
                got = _read(hsv, cx + dx, cy + dy, r)
                if got is None:
                    continue
                key = (got.bars, got.confidence, -(dx * dx + dy * dy + 4 * dr * dr))
                if best is None or key > best:
                    best, winner = key, got
    return winner


def refine_ring(frame_bgr: np.ndarray, cx: float, cy: float, radius: float,
                search: int = 12, step: int = 2):
    """Slide a rough guess onto the best-fitting nearby gauge.

    The calibration wizard starts from a mouse click, which is nowhere near
    pixel-exact.
    """
    return _refine(band_hsv(frame_bgr), cx, cy, radius, search, step)


def _band_lit(hsv: np.ndarray):
    """Mask of 'looks like a lit gauge arc' across the search band, or None if too dark.

    Everything stays in uint8: widening to int64 first costs 328k pixels x 3
    channels and dominated the frame budget at 12 ms on its own.
    """
    if hsv.size == 0:
        return None
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    flat = val.ravel()
    k = max(1, int(flat.size * 0.01))
    v_ref = int(np.partition(flat, -k)[-k])
    if v_ref < V_REF_FLOOR:
        return None
    return ((hue >= HUE_LO) & (hue <= HUE_HI) & (sat >= SAT_MIN)
            & (val >= int(v_ref * LIT_V_FRAC)))


def _jitter_offsets(radius):
    """Neighbourhood offsets ordered by distance, scaled to the gauge size.

    A translation is not a rotation, so the phase search inside _read cannot
    absorb it: at a 14 px gauge the sampling annulus is only ~3 px thick and a
    2 px shift moves it off the arcs entirely. Searching a few pixels outwards
    recovers that, and the caller adopts whatever it finds so the next frame
    reads on the first try.
    """
    reach = max(2, int(round(radius * 0.20)))
    offs = [(dx, dy) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1)]
    return sorted(offs, key=lambda o: o[0] ** 2 + o[1] ** 2)


def _jitter_search(hsv, cx, cy, radius):
    for dx, dy in _jitter_offsets(radius):
        got = _read(hsv, cx + dx, cy + dy, radius)
        if got is not None:
            return got
    return None


def read_ring_recovering(hsv, cx, cy, radius, lit=_UNSET):
    """Read a gauge, retrying nearby if the stored position has drifted.

    Returns ``(ring, drifted)``; `drifted` is True when the read only succeeded
    off-centre, which means the caller should update its position.

    Pass `lit` (the "looks like a lit arc" mask) to gate the retry: without it,
    a slot whose ally is merely alive misses on the first try every frame and so
    pays for the whole jitter search, which measured 88 ms a frame. A mask of
    None means the crop was too dark to judge either way, which also means
    "do not search".
    """
    got = _read(hsv, cx, cy, radius)
    if got is not None:
        return got, False
    if lit is not _UNSET and not (lit is not None and _blue_around(lit, cx, cy, radius)):
        return None, False
    got = _jitter_search(hsv, cx, cy, radius)
    return (got, True) if got is not None else (None, False)


def _blue_around(lit, cx, cy, radius) -> bool:
    """Is there gauge-blue anywhere near this slot?

    Without this gate the jitter search runs every frame for every ally who is
    simply alive, which costs 88 ms a frame at 2276x1280 and drops the loop to
    11 fps. Measured on the fixtures: a live gauge box reads 0.25-0.36 blue,
    one segment 0.11, nobody downed 0.03, empty HUD 0.00.
    """
    side = radius * 2.6
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    box = lit[y0:int(y0 + side), x0:int(x0 + side)]
    return box.size > 0 and float(box.mean()) >= BLUE_GATE_FRAC


def propose_rings(frame_bgr: np.ndarray):
    """Auto-locate the ally gauges in the search band, top to bottom.

    A convenience for first run: the result seeds the calibration wizard, which
    the user then confirms. Not the steady-state path, so it filters hard rather
    than risk presenting a patch of sky as a gauge.
    """
    hsv = band_hsv(frame_bgr)
    lit = _band_lit(hsv)
    if lit is None:
        return []
    lit = lit.copy()
    lit[:int(frame_bgr.shape[0] * BAND_Y0_FRAC), :] = False
    bh, bw = lit.shape
    min_area = max(40, int(bw * bh * MIN_BLOB_AREA_FRAC))
    blobs = [{"cx": b.cx, "cy": b.cy, "size": float(max(b.w, b.h)), "area": b.area}
             for b in label_blobs(lit) if b.area >= min_area]
    if not blobs:
        return []

    # One gauge arrives as up to three separate arcs; merge blobs that are close
    # relative to their own size, which keeps the two ally rows apart.
    clusters = []
    for b in sorted(blobs, key=lambda b: -b["area"]):
        for cl in clusters:
            if np.hypot(b["cx"] - cl[-1]["cx"], b["cy"] - cl[-1]["cy"]) <= 2.2 * np.mean(
                    [x["size"] for x in cl]):
                cl.append(b)
                break
        else:
            clusters.append([b])

    mid = float(np.mean(RADIAL_FRACS))
    out = []
    for cl in clusters[:8]:
        weights = np.array([b["area"] for b in cl], dtype=float)
        gx = float(np.average([b["cx"] for b in cl], weights=weights))
        gy = float(np.average([b["cy"] for b in cl], weights=weights))
        gr = np.mean([b["size"] for b in cl]) / mid / 2.0
        guess = _refine(hsv, gx, gy, gr, search=10, step=2)
        if guess is not None and guess.confidence >= PROPOSE_MIN_CONFIDENCE:
            out.append(guess)

    out.sort(key=lambda r: r.cy)
    deduped = []
    for r in out:
        if not any(np.hypot(r.cx - p.cx, r.cy - p.cy) < p.radius for p in deduped):
            deduped.append(r)
    return deduped


def _gray_l1_gradient(box):
    """|d/dx| + |d/dy| of the 3x3 Sobel over a BGR box, in OpenCV's integer form."""
    b = box[..., 0].astype(np.int32)
    g = box[..., 1].astype(np.int32)
    r = box[..., 2].astype(np.int32)
    lum = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14
    p = np.pad(lum, 1, mode="reflect")      # reflect-101, as OpenCV pads
    top, mid, bot = p[:-2], p[1:-1], p[2:]
    gx = (top[:, 2:] - top[:, :-2]) + 2 * (mid[:, 2:] - mid[:, :-2]) + (bot[:, 2:] - bot[:, :-2])
    gy = (bot[:, :-2] + 2 * bot[:, 1:-1] + bot[:, 2:]) \
        - (top[:, :-2] + 2 * top[:, 1:-1] + top[:, 2:])
    return np.abs(gx) + np.abs(gy)


def row_ys(shape, n: int = MAX_ROWS, rect=None):
    """Centres of the first `n` ally rows - the same grid the gauges sit on."""
    return [cy for _cx, cy, _r in gauge_spots(shape, n, rect)]


def _row_cols(shape, rect):
    """Where the portrait box is across: x0, x1, and its half height."""
    x0_r, _y0, w, h = _rect_of(shape, rect)
    return (x0_r + int(w * ROW_X_FRACS[0]), x0_r + int(w * ROW_X_FRACS[1]),
            int(ROW_HALF_FRAC * h))


def row_strength(frame, cy: float, rect=None) -> float:
    """Share of pixels on a hard edge inside that row's portrait box."""
    x0, x1, half = _row_cols(frame.shape, rect)
    y0, y1 = int(cy) - half, int(cy) + half
    if y0 < 0 or y1 > frame.shape[0] or x1 <= x0:
        return 0.0
    edges = _gray_l1_gradient(frame[y0:y1, x0:x1, :3])
    return float((edges > ROW_EDGE_GRADIENT).mean())


def _portrait_curve(frame, ys, step=2, rect=None):
    """(candidate y, portrait score) down the column the ally panel occupies.

    The gradient is computed once for the whole strip and summed with a running
    window, so scoring every candidate costs no more than sampling one point.
    Only the picture inside the letterbox is walked; a black bar is not a face.
    """
    _rx0, ry0, rw, rh = _rect_of(frame.shape, rect)
    x0, x1, half = _row_cols(frame.shape, rect)
    win = int(ROW_SEARCH_FRAC_W * rw)
    if not ys or x1 <= x0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    top, bot = max(ry0 + half, int(min(ys)) - win), min(ry0 + rh - half, int(max(ys)) + win)
    span = 2 * half
    if bot - top < span:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    per_row = (_gray_l1_gradient(frame[top - half:bot + half, x0:x1, :3])
               > ROW_EDGE_GRADIENT).sum(axis=1, dtype=np.int64)
    cum = np.concatenate(([0], np.cumsum(per_row)))
    cand = np.arange(top, bot + 1, step, dtype=np.int64)
    idx = np.clip(cand - top, 0, per_row.size - span)
    return cand, (cum[idx + span] - cum[idx]) / float(span * (x1 - x0))


def row_peaks(frame, ys=None, rect=None):
    """(y, score) of the busiest portrait box in each row's window.

    The grid is a starting point, not a certainty: the in-game UI size setting
    moves the whole panel - one capture has its row pitch 11% wider than the real
    screenshots' and its first row 87 px above where the picture's width puts it.
    The peak is biased by up to ~28 px because a face is busier than the shoulders
    below it, so it is a coarse anchor for finding the panel, not a substitute for
    reading a gauge.

    Each row may only look between the midpoints to its neighbours. Without that
    the window, which is nearly half a row pitch wide, reaches into the next
    ally's portrait and two rows collapse onto one.
    """
    ys = row_ys(frame.shape, rect=rect) if ys is None else list(ys)
    _rx0, _ry0, rw, _rh = _rect_of(frame.shape, rect)
    win = int(ROW_SEARCH_FRAC_W * rw)
    cand, scores = _portrait_curve(frame, ys, rect=rect)
    out = []
    for i, y in enumerate(ys):
        lo = y - win if i == 0 else max(y - win, (ys[i - 1] + y) / 2.0)
        hi = y + win if i == len(ys) - 1 else min(y + win, (y + ys[i + 1]) / 2.0)
        near = (cand >= lo) & (cand <= hi)
        if not near.any():
            out.append((float(y), 0.0))
            continue
        out.append((float(cand[near][int(np.argmax(scores[near]))]),
                    float(scores[near].max())))
    return out


def _hp_mask(crop):
    """Pixels the colour of an ally's health bar: lit, saturated red-orange.

    Only the low end of the hue wheel. Red wraps at 180 and the wrap is where the
    ally plate and a pink sky live, so taking it would call a downed ally standing.
    """
    hsv = _hsv_of(crop)
    hue = hsv[..., 0].astype(int)
    sat = hsv[..., 1].astype(int)
    val = hsv[..., 2].astype(int)
    return ((hue <= HP_HUE_MAX) & (sat >= HP_SAT_MIN) & (val >= HP_VAL_MIN))


def _rows_with_run(mask, length: int):
    """For each row of a boolean map: does it hold `length` set pixels in a row?"""
    if mask.shape[1] < length:
        return np.zeros(mask.shape[0], dtype=bool)
    gap = np.cumsum(~mask, axis=1, dtype=np.int32)
    gap = np.concatenate((np.zeros((mask.shape[0], 1), np.int32), gap), axis=1)
    return ((gap[:, length:] - gap[:, :-length]) == 0).any(axis=1)


def ally_bar(frame, cy: float, rect=None) -> bool:
    """Is an ally's health bar drawn across this row?

    Three facts about the sprite, none of which colour can supply: the bar sits in
    a band just above the row's centre, its fill begins at one fixed end, and it
    runs rightwards from there. Measured against that, lava of exactly the bar's
    hue and saturation fails because it floods the row *below* the centre and
    reaches left of where a fill can start, while an ally revived on a third of his
    health - 0.025 W of fill, under half the bar - still starts in the right place.
    """
    rx0, ry0, rw, rh = _rect_of(frame.shape, rect)
    left = rx0 + int(rw * HP_LEFT_FRAC_W)
    x1 = rx0 + int(rw * 0.150)          # past the longest bar (0.066 W)
    tol = int(rw * HP_START_TOL_FRAC_W)
    need = max(2, int(rw * HP_MIN_FILL_FRAC_W))
    y0 = int(cy) - int(rw * HP_BAND_TOP_FRAC_W)
    y1 = int(cy) - int(rw * HP_BAND_BOTTOM_FRAC_W)
    if y0 < ry0 or y1 > ry0 + rh or x1 - need <= left - tol:
        return False
    mask = _hp_mask(frame[y0:y1, left - tol:x1, :3])
    ok = np.zeros(mask.shape[0], dtype=bool)
    for i, line in enumerate(mask):
        first = int(np.argmax(line)) if line.any() else -1
        ok[i] = first >= tol and first <= tol + tol and \
            bool(line[first:first + need].all())
    return bool(_rows_with_run(ok[None, :], HP_MIN_ROWS).any())


def count_rows(frame, rect=None, spots=None) -> int:
    """How many ally rows are on screen, counted from the top down.

    A row is an ally when the HUD has drawn something only an ally gets: a gauge,
    which is up while they are downed, or their health bar, which is up while they
    are not. And because the list is contiguous, an ally in a lower row settles the
    row above it - which matters when the one above is standing on nearly no
    health, with too little of its bar left to see. Counting stops at the first row
    with neither. Over-counting only withholds the music, which is the safe
    direction; under-counting would play it while an ally is still on their feet.
    """
    spots = gauge_spots(frame.shape, rect=rect) if spots is None else list(spots)
    if not spots:
        return 0
    gauges = [got is not None for got in detect_spots(frame, spots)]
    rows = 0
    for i, ((cx, cy, _r), has_gauge) in enumerate(zip(spots, gauges)):
        if has_gauge or any(gauges[i + 1:]) or ally_bar(frame, cy, rect):
            rows += 1
        else:
            break
    return rows
