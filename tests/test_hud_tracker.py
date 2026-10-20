"""One frame must not be allowed to move the picture box.

`hud_rect` measures a single frame, and the HUD geometry hangs off its top edge.
These are the cases the user's own material produced or could have produced: a
fade that measures 312 px where the bar is 80, an overlay sitting in the bottom
bar so that one reads 0, and a genuine window resize that has to be followed.
"""
import numpy as np
import pytest

from nrdoom.hud_tracker import (ACQUIRING, LOCKED, LOST, OCCLUDED, HudTracker)


def shot(w=320, h=200, top=20, bottom=20, left=0, right=0, badge=None):
    """A letterboxed frame: black bars around a mid-grey picture.

    `badge` paints a bright block inside the top bar, which is what a recording
    indicator does to the measurement.
    """
    img = np.zeros((h, w, 3), np.uint8)
    img[top:h - bottom, left:w - right] = 90
    if badge:
        x, y, bw, bh = badge
        img[y:y + bh, x:x + bw] = 240
    return img


def tops(tracker, frames):
    return [tracker.measure(f).top for f in frames]


def test_a_single_bad_frame_does_not_move_the_box():
    """One frame measuring 0 - a fade, a half-composited frame - must not drag the
    geometry 20 px, which is four times what a gauge read tolerates."""
    tracker = HudTracker(window=20)
    good = shot()
    for _ in range(10):
        tracker.measure(good)
    assert tracker.measure(shot(top=0, bottom=0)).top == 20
    assert tracker.measure(shot(top=0, bottom=0)).top == 20
    assert tracker.state == OCCLUDED, "disagreeing, but still held"


def test_a_badge_in_the_bar_is_not_a_new_bar():
    """The ShadowPlay case: bright pixels in the middle of the top bar make the
    measurement start below them. The box we already confirmed stays."""
    tracker = HudTracker(window=20)
    for _ in range(12):
        tracker.measure(shot())
    specked = shot(badge=(100, 5, 24, 8))
    assert tracker.measure(specked).top == 20


def test_a_real_move_is_followed_within_the_window():
    """The other side: the box did genuinely change, and the tracker has to catch
    up rather than defend a stale one forever."""
    tracker = HudTracker(window=20, share=0.6)
    for _ in range(10):
        tracker.measure(shot())
    moved = shot(top=40, bottom=0)
    seen = [tracker.measure(moved).top for _ in range(20)]
    assert seen[-1] == 40
    assert tracker.state == LOCKED


def test_a_resize_starts_over():
    """Nothing measured on the old window describes the new one."""
    tracker = HudTracker(window=20)
    for _ in range(10):
        tracker.measure(shot())
    pose = tracker.measure(shot(w=640, h=400, top=40, bottom=40))
    assert pose.state == ACQUIRING
    assert pose.rect[2:] == (640, 320)


def test_the_top_edge_survives_a_bottom_bar_being_covered():
    """(80, 80) and (80, 0) are the same answer where the geometry lives; the vote
    is on the origin, so a covered bottom bar does not unsettle it."""
    tracker = HudTracker(window=20)
    wide = dict(w=640, h=400)
    got = tops(tracker, [shot(**wide, top=80, bottom=80) if i % 2
                         else shot(**wide, top=80, bottom=0) for i in range(10)])
    assert set(got) == {80}


def test_the_first_frame_is_used_rather_than_waited_out():
    """Arming should not go blind for a second while the vote collects quorum -
    but it must not claim to be locked on one frame's word either."""
    tracker = HudTracker(window=20)
    first = tracker.measure(shot(top=40, bottom=40))
    assert first.top == 40 and first.state == ACQUIRING
    assert tracker.measure(shot(top=40, bottom=40)).state == LOCKED


def test_the_pose_reports_how_much_to_trust_it():
    tracker = HudTracker(window=10)
    steady = tracker.measure(shot())
    for _ in range(9):
        steady = tracker.measure(shot())
    assert steady.confidence == 1.0
    mixed = tracker.measure(shot(top=0, bottom=0))
    assert 0.0 < mixed.confidence < 1.0


# -- LOST: the plan's §5.3 asks for the state to be named, not silently held ----
def test_a_long_disagreement_stops_being_called_occluded():
    """A frame or two out of place is occlusion. Half a second of not finding the
    anchor is something else, and the difference is what the user is told."""
    tracker = HudTracker(window=20, lost_after=5)
    for _ in range(10):
        tracker.measure(shot())
    assert tracker.state == LOCKED
    odd = shot(top=0, bottom=0)
    for _ in range(4):
        assert tracker.measure(odd).state == OCCLUDED
    assert tracker.measure(odd).state == LOST
    assert tracker.measure(odd).state == LOST
    assert tracker.measure(odd).top == 20, "held, even while admitting it is lost"


def test_the_anchor_coming_back_clears_lost():
    tracker = HudTracker(window=20, lost_after=5)
    good = shot()
    for _ in range(10):
        tracker.measure(good)
    for _ in range(6):
        tracker.measure(shot(top=0, bottom=0))
    assert tracker.state == LOST
    assert tracker.measure(good).state == LOCKED


def test_a_half_second_overlay_is_not_a_new_letterbox():
    """§5.3: a cinematic bar crossed the user's own recording in 13 frames and moved
    the whole geometry with it, which threw away the accumulated positions and reset
    the roster. Same capture size plus short duration is not a change of box."""
    good, bar = shot(h=400), shot(h=400, top=60, bottom=20)
    tracker = HudTracker(window=20, relocate_frames=30)
    for _ in range(12):
        tracker.measure(good)
    assert tracker.state == LOCKED
    for _ in range(13):
        pose = tracker.measure(bar)
    assert pose.top == 20, "a 0.6 s overlay moved the picture box"
    assert tracker.state == LOST, "and it was not even reported as disputed"
    for _ in range(30):                      # a change that stays is adopted
        pose = tracker.measure(bar)
    assert pose.top == 60
    assert tracker.state == LOCKED


def test_a_real_move_passes_through_lost_on_the_way():
    """Following a box that genuinely moved is allowed to take a second, and during
    it the tracker must not pretend the old anchor is still being seen."""
    tracker = HudTracker(window=20, share=0.6, lost_after=5)
    for _ in range(10):
        tracker.measure(shot())
    states = [tracker.measure(shot(top=40, bottom=0)).state for _ in range(20)]
    assert LOST in states, "the gap where neither anchor is being confirmed"
    assert states[-1] == LOCKED
    assert tracker.measure(shot(top=40, bottom=0)).top == 40
