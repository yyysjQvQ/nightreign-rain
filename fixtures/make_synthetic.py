"""Generate synthetic 1-bar / 2-bar / no-ring fixtures from the real screenshot.

Re-runnable: `python fixtures/make_synthetic.py`. The painted counts exercise the
arc-counting logic; how the game really draws an empty segment is now settled by
the genuine 1- and 2-bar captures in fixtures/real_duo_*.jpg, which show an opaque
dark track.
"""
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_both_3bars.jpg"
SELFTEST = HERE.parent / "assets" / "selftest_panel.npz"

# Measured on golden_both_3bars.jpg (2276x1280).
RINGS = [(128, 247, 27), (128, 358, 27)]
TRACK_RGB = (38, 37, 65)

LIT_MIN_B = 135
LIT_MIN_B_MINUS_R = 55
LIT_MIN_B_MINUS_G = 70


def is_lit(px_bgr):
    b, g, r = (int(v) for v in px_bgr)
    return b > LIT_MIN_B and b - r > LIT_MIN_B_MINUS_R and b - g > LIT_MIN_B_MINUS_G


def arc_runs(img, cx, cy, radius):
    """Return [(start_deg, end_deg)] of contiguous lit arcs, clockwise from 3 o'clock."""
    h, w = img.shape[:2]
    flags = []
    for deg in range(360):
        a = math.radians(deg)
        hit = 0
        for frac in (0.70, 0.80, 0.90):
            x = int(round(cx + radius * frac * math.cos(a)))
            y = int(round(cy - radius * frac * math.sin(a)))
            if 0 <= x < w and 0 <= y < h and is_lit(img[y, x]):
                hit += 1
        flags.append(hit >= 2)

    if not any(flags):
        return []
    shift = flags.index(True)
    rot = flags[shift:] + flags[:shift]
    runs = []
    start = 0
    for i in range(1, 361):
        if i == 360 or not rot[i]:
            if i - start >= 25:
                runs.append(((shift + start) % 360, (shift + i) % 360))
            start = i
    return runs


def paint_arc_down(img, cx, cy, radius, start_deg, span, colour=TRACK_RGB):
    """Fill one arc band with the given colour, leaving the ring frame intact."""
    h, w = img.shape[:2]
    step = 0.25
    for k in np.arange(-2.0, span + 2.0, step):
        a = math.radians(start_deg + k)
        cos_a, sin_a = math.cos(a), math.sin(a)
        for frac in np.arange(0.60, 0.97, 0.005):
            x = int(round(cx + radius * frac * cos_a))
            y = int(round(cy - radius * frac * sin_a))
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = colour
    return img


def variant(keep_bars, drop_from_end=True, rings=RINGS, fill=None):
    """Copy the golden frame, keeping only `keep_bars` lit arcs on the given rings."""
    img = cv2.imread(str(GOLDEN))
    track = TRACK_RGB if fill is None else fill
    for cx, cy, radius in rings:
        runs = arc_runs(img, cx, cy, radius)
        surplus = len(runs) - keep_bars
        if surplus <= 0:
            continue
        # Drop from the end of the run list so the surviving arcs stay put.
        for start, end in (runs[-surplus:] if drop_from_end else runs[:surplus]):
            paint_arc_down(img, cx, cy, radius, start, (end - start) % 360, track)
    return img


def sky_patch_colour():
    """Brightest night-sky sample, for the harshest unlit-segment fill.

    The sky and a lit arc have near-identical hue and saturation, so brightness
    is the only thing telling them apart. Using the mean would be an easy test;
    this takes the top-V pixel so an unlit segment is as close to a lit one as
    the sky ever gets.
    """
    img = cv2.imread(str(GOLDEN))
    patch = img[175:205, 485:535].reshape(-1, 3)
    hsv = cv2.cvtColor(img[175:205, 485:535], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return tuple(int(v) for v in patch[int(np.argmax(hsv[:, 2]))])


def shift_panel_up(img, drops=1):
    """Move the ally list up by one row, so the bottom row shows bare background.

    That is what leaving looks like: the remaining allies keep their order and
    the last row simply stops existing. Repeated for more than one departure.
    """
    h, _w = img.shape[:2]
    pitch = int(0.0885 * h)
    top = int((0.200 - 0.055) * h)          # just above the first row's portrait
    for _ in range(drops):
        img[top:h - pitch] = img[top + pitch:h]
    return img


def make_selftest_panel():
    """The picture the packaged exe checks itself against.

    Letterboxed on purpose, like the user's 2560x1600 screen: a 16:9 picture
    inside black bars, so the shipped self-check only passes if the detector finds
    the bars and measures the HUD against the picture. Half scale, whole frame -
    a crop would hand the geometry the wrong ruler.
    """
    img = cv2.imread(str(GOLDEN))
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h = small.shape[0]
    pad = (h * 10 // 9 - h) // 2 * 2          # a 16:10 frame around a 16:9 picture
    framed = np.zeros((h + pad, small.shape[1], 3), dtype=np.uint8)
    framed[pad // 2:pad // 2 + h] = small
    np.savez_compressed(SELFTEST, frame=framed)
    print("wrote %s %s, %d px of black above and below" % (
        SELFTEST.name, framed.shape, pad // 2))


def main():
    sky = sky_patch_colour()
    print("night-sky fill colour (BGR):", sky)
    outputs = {
        "synth_both_2bars.png": variant(2),
        "synth_both_1bar.png": variant(1),
        "synth_first_kept_1bar.png": variant(1, drop_from_end=False),
        "synth_mixed_3_1.png": variant(1, rings=[RINGS[1]]),
        # A what-if: an unlit segment that lets the purple night sky show through
        # instead of leaving a dark track. The real 1- and 2-bar captures answer
        # this - unlit segments are an opaque dark track - so the detector reads
        # them as 1 and 2, and these two are only kept as gradient inputs for
        # tests/test_cv2_equivalence.
        "synth_2bars_skyfill.png": variant(2, fill=sky),
        "synth_1bar_skyfill.png": variant(1, fill=sky),
    }
    empty = cv2.imread(str(GOLDEN))
    for cx, cy, radius in RINGS:
        paint_arc_down(empty, cx, cy, radius, 0, 360)
    outputs["synth_no_rings.png"] = empty

    # Duo mode: only one ally, so only the first row carries a gauge. The second
    # row's portrait is left in place, which is the harder case: the detector
    # must not invent a gauge there.
    duo = cv2.imread(str(GOLDEN))
    paint_arc_down(duo, *RINGS[1], 0, 360)
    outputs["synth_duo_one_ally.png"] = duo

    # Row counting: nobody has left / one ally left / solo. These shift the real
    # panel up, so the vacated rows are genuine background rather than a blur.
    for n, name in ((1, "synth_rows_duo.png"), (2, "synth_rows_solo.png")):
        img = cv2.imread(str(GOLDEN))
        outputs[name] = shift_panel_up(img, n)

    for name, img in outputs.items():
        cv2.imwrite(str(HERE / name), img)
        print("wrote", name, img.shape)

    for name in outputs:
        img = cv2.imread(str(HERE / name))
        print("%-28s colour-only arcs per ring: %s"
              % (name, [len(arc_runs(img, *r)) for r in RINGS]))

    make_selftest_panel()


if __name__ == "__main__":
    main()
