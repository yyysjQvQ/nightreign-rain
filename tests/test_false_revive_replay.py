"""The stop-the-song rule must not be fooled by time it cannot see.

Phase 0 of the V2 plan: these are written against the *shipped* behaviour and are
expected to fail there - that failure is the baseline the plan asks for. Each one
is driven through `nrdoom.replay`, i.e. the real `Engine` on a clock we control,
so the assertions say what the user would hear rather than what a counter holds.

The shared defect: `ReviveWatcher` counts *observations of a frame*, and nothing
ties those observations to the time they were made or to whether the frame was
actually new. So a frozen feed, a burst that ended, and a revive spread over a
minute all accumulate into the same event.
"""
from pathlib import Path

import cv2
import pytest

from nrdoom.engine import Engine
from nrdoom.replay import EventRecorder, ManualClock, ReplaySource
from nrdoom.settings import Settings, Slot

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot
GOLDEN_SLOTS = [Slot(128, 247, 27), Slot(128, 358, 27)]
FPS = 20.0


def frame(name="golden_both_3bars.jpg"):
    return shot(name)


def erase_ring(img, top):
    """The gauge is gone and nothing was drawn in its place: proves nothing."""
    img[top:top + 54, 100:158] = (38, 37, 65)
    return img


def stand_up(img, cy):
    """A real revive: the gauge is gone *and* the health bar is back."""
    erase_ring(img, cy - 26)
    img[int(cy - 0.0080 * img.shape[1]):int(cy - 0.0030 * img.shape[1]),
        int(img.shape[1] * 0.082):int(img.shape[1] * 0.140)] = (30, 45, 185)
    return img


def rig(qapp, first, **kw):
    """Engine + replay source + recorder, all on one manual clock.

    Only `hold_frames` is cut down, so a song starts inside the first 0.2 s of
    every test. Everything else keeps the shipped defaults, because the whole
    point is what the shipped combination does with time.
    """
    st = Settings(slots=list(GOLDEN_SLOTS), hold_frames=1, **kw)
    clock = ManualClock()
    src = ReplaySource(fps=FPS, clock=clock)
    player = EventRecorder(clock)
    eng = Engine(st, source=src, player=player, clock=clock)
    src.feed(first)
    eng.set_listening(True)
    eng._tick()
    return eng, src, player


def show(eng, src, img, seconds):
    for _ in range(int(round(seconds * FPS))):
        src.feed(img)
        eng._tick()


def holds(eng, src, seconds):
    """Time passes on the same buffer - what a stalled capture looks like."""
    for _ in range(int(round(seconds * FPS))):
        src.hold()
        eng._tick()


def actions(player):
    return [e["action"] for e in player.events]


def fire(eng, src, player):
    show(eng, src, frame(), 0.2)
    assert actions(player) == ["play"], "the song should have started"
    player.events.clear()


# -- the cases ---------------------------------------------------------------
# Each wrong-stop case runs under both policies: "v1" documents what the shipped
# frame counter does (that is the Phase 0 baseline, recorded as an assertion
# rather than a comment), "v2" is what the plan requires instead.
OLD_STOPS = [("v1", ["stop"]), ("v2", [])]


@pytest.mark.parametrize("policy,expected", OLD_STOPS)
def test_a_frozen_feed_is_not_an_hour_of_evidence(qapp, policy, expected):
    """One new frame saying "standing", then the capture stops delivering. The
    same buffer read sixty times is one observation, not sixty, and a screen that
    has not moved in three seconds proves nothing about the game."""
    eng, src, player = rig(qapp, frame(), stop_policy=policy)
    fire(eng, src, player)
    show(eng, src, stand_up(frame(), 358), 0.05)      # one genuine frame
    holds(eng, src, 3.0)                              # then silence
    assert actions(player) == expected


@pytest.mark.parametrize("policy,expected", OLD_STOPS)
def test_scattered_barely_alive_frames_do_not_add_up(qapp, policy, expected):
    """Ten bursts of four standing-looking frames, two seconds of "cannot tell"
    between them. Nothing here is a revive in progress; the plan's rule is that
    error scattered over minutes must not accumulate into an event."""
    eng, src, player = rig(qapp, frame(), stop_policy=policy)
    fire(eng, src, player)
    standing, blank = stand_up(frame(), 358), erase_ring(frame(), 358 - 26)
    for _ in range(10):
        show(eng, src, standing, 0.2)
        show(eng, src, blank, 2.0)
    assert actions(player) == expected


@pytest.mark.parametrize("policy,expected", OLD_STOPS)
def test_a_candidate_does_not_resume_after_a_gap(qapp, policy, expected):
    """Three half-second stretches of standing evidence, a second apart. Each on
    its own is too short to be a revive, so the only way to stop here is to add
    the three together across the gaps - which is the accumulation being removed."""
    eng, src, player = rig(qapp, frame(), stop_policy=policy)
    fire(eng, src, player)
    standing, blank = stand_up(frame(), 358), erase_ring(frame(), 358 - 26)
    for _ in range(3):
        show(eng, src, standing, 0.5)
        show(eng, src, blank, 1.0)
    assert actions(player) == expected


def test_a_real_revive_still_stops_the_song(qapp):
    """The other side of the same rule, and the reason "never stop" is not a fix:
    one ally gets up, stays up, and the music ends within two seconds."""
    eng, src, player = rig(qapp, frame())
    fire(eng, src, player)
    show(eng, src, stand_up(frame(), 358), 1.0)
    assert actions(player) == ["stop"], "a confirmed standing ally must end it"


def test_a_real_revive_is_not_stopped_early(qapp):
    """... and not a half-second of flicker either: the stop belongs to the moment
    the ally is demonstrably up, so the song keeps playing through a doubt."""
    eng, src, player = rig(qapp, frame())
    fire(eng, src, player)
    show(eng, src, stand_up(frame(), 358), 0.5)
    assert actions(player) == []


def test_a_revive_in_progress_does_not_stop_the_song(qapp):
    """Three segments to one is the gauge draining while someone is still on the
    floor. Guard: unchanged behaviour, and the case a length-based rule breaks."""
    eng, src, player = rig(qapp, frame())
    fire(eng, src, player)
    show(eng, src, frame("synth_mixed_3_1.png"), 3.0)
    assert actions(player) == []


def test_one_ally_up_while_the_other_is_down_stops_once(qapp):
    """Per ally: slot 1 standing for three seconds while slot 0 is still downed
    is one revive, and it must reach the user's ears exactly once."""
    eng, src, player = rig(qapp, frame())
    fire(eng, src, player)
    show(eng, src, stand_up(frame(), 358), 3.0)
    assert actions(player) == ["stop"]
    show(eng, src, stand_up(frame(), 358), 3.0)
    assert actions(player) == ["stop"], "the same revive must not fire twice"
