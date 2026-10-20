"""The stop decision, one ally at a time, on a clock instead of a counter."""
import pytest

from nrdoom.event_arbiter import EventArbiter
from nrdoom.slot_observation import STRONG_ALIVE, STRONG_DOWN, UNKNOWN, SlotObservation
from nrdoom.temporal_filter import ReviveTracker


def obs(t, frame_id=None, bars=None, standing=False, fresh=True, pos=True,
        slot=0, ring_share=None, panel_lit=0):
    return SlotObservation(slot_id=slot, t=t,
                           frame_id=frame_id if frame_id is not None else int(t * 20),
                           bars=bars, standing=standing, fresh=fresh,
                           position_ok=pos, ring_share=ring_share,
                           panel_lit=panel_lit)


# -- what a frame is allowed to say ------------------------------------------
def test_a_gauge_read_is_downed_evidence():
    assert obs(0.0, bars=3).evidence() == STRONG_DOWN


def test_a_health_bar_where_the_gauge_was_is_standing_evidence():
    assert obs(0.0, standing=True).evidence() == STRONG_ALIVE


def test_reading_nothing_proves_nothing():
    assert obs(0.0).evidence() == UNKNOWN


def test_no_gauge_on_a_stale_frame_is_not_a_revive():
    """The colour rules failing on a picture that has not arrived is the exact
    shape of the false stop the user reported, so it must not be standing
    evidence even when the health bar test agrees."""
    assert obs(0.0, standing=True, fresh=False).evidence() == UNKNOWN


def test_no_gauge_at_a_position_that_just_moved_is_not_a_revive():
    assert obs(0.0, standing=True, pos=False).evidence() == UNKNOWN


# -- an empty ring is evidence, an unreadable one is not ---------------------
# Measured on the user's recording: an ally stood for seven seconds with no
# visible health-bar fill at all, so "wait for the red" holds the song over a
# rescue. The annulus being empty is the other positive reading - but only where
# the panel is bright enough for "empty" to mean anything.
def test_an_empty_annulus_on_a_lit_panel_is_standing():
    assert obs(0.0, ring_share=0.0, panel_lit=150).evidence() == STRONG_ALIVE


def test_a_gauge_the_spoke_gate_rejected_is_still_a_gauge():
    """The purple-sky case: the full read refuses it, but 84% of the annulus is
    lit, and that is a downed ally, not a standing one."""
    assert obs(0.0, ring_share=0.844, panel_lit=178).evidence() == UNKNOWN


def test_one_segment_down_is_not_an_empty_annulus():
    assert obs(0.0, ring_share=0.277, panel_lit=185).evidence() == UNKNOWN


def test_a_dark_panel_cannot_report_an_empty_ring():
    """Below the brightness floor every pixel passes the lit test, so the share
    saturates near 1.0 on an empty row and near 0.0 on a real one - either way it
    has stopped meaning anything."""
    assert obs(0.0, ring_share=0.0, panel_lit=42).evidence() == UNKNOWN


def test_no_annulus_measurement_is_no_evidence():
    assert obs(0.0, ring_share=None, panel_lit=0).evidence() == UNKNOWN


def test_the_empty_ring_rule_can_be_switched_off():
    """The rollback switch the A/B is measured with: off, the stop waits for the
    red fill exactly as it did before the annulus was consulted."""
    frame = obs(0.0, ring_share=0.0, panel_lit=150)
    assert frame.evidence() == STRONG_ALIVE
    assert SlotObservation(slot_id=0, t=0.0, frame_id=0, ring_share=0.0,
                           panel_lit=150,
                           accept_empty_ring=False).evidence() == UNKNOWN


# -- what a candidate has to survive -----------------------------------------
def stream(kind, start, seconds, fps=20.0, **kw):
    """`kind` frames of the given evidence, at a realistic frame rate."""
    for i in range(int(seconds * fps)):
        yield obs(start + i / fps,
                  bars=3 if kind == STRONG_DOWN else kw.get("bars"),
                  standing=kind == STRONG_ALIVE)


def feed(tracker, frames):
    """The times at which the tracker said 'confirmed', which is what the
    assertions are about."""
    return [obs.t for obs in frames if tracker.update(obs)]


def test_a_second_of_standing_evidence_ends_the_song():
    tracker = ReviveTracker()
    assert feed(tracker, stream(STRONG_ALIVE, 0.0, 1.0)) == [pytest.approx(0.8)]


def test_half_a_second_of_it_does_not():
    tracker = ReviveTracker()
    assert feed(tracker, stream(STRONG_ALIVE, 0.0, 0.5)) == []


def test_the_same_frame_twice_is_one_observation():
    """A frozen capture feeds the tracker the same buffer forever; the frame id is
    the only thing that says "this is not new news"."""
    tracker = ReviveTracker()
    again = [obs(0.0, standing=True) for _ in range(200)]
    assert feed(tracker, again) == []


def test_a_clock_that_goes_backwards_is_not_evidence():
    tracker = ReviveTracker()
    assert feed(tracker, [obs(5.0, standing=True), obs(4.0, standing=True)]) == []


def test_a_gap_in_the_evidence_lapses_the_candidate():
    """0.4 s of standing, a second of nothing readable, 0.6 s more. Neither stretch
    is long enough on its own and the first one expired, so the song plays on."""
    tracker = ReviveTracker()
    frames = (list(stream(STRONG_ALIVE, 0.0, 0.4))
              + list(stream(UNKNOWN, 0.5, 1.0))
              + list(stream(STRONG_ALIVE, 1.5, 0.6)))
    assert feed(tracker, frames) == []


def test_a_downed_reading_clears_a_candidate_in_progress():
    tracker = ReviveTracker()
    frames = (list(stream(STRONG_ALIVE, 0.0, 0.5))
              + list(stream(STRONG_DOWN, 0.5, 0.1))
              + list(stream(STRONG_ALIVE, 0.6, 1.0)))
    assert feed(tracker, frames) == [pytest.approx(1.45, abs=0.05)], \
        "it has to start over from after he was seen on the floor again"


def test_a_conflict_inside_the_recent_window_vetoes_the_stop():
    """Standing evidence with a downed reading 0.3 s ago is a flicker, not a
    revive: the share can be perfect and it still must not stop."""
    tracker = ReviveTracker()
    frames = list(stream(STRONG_ALIVE, 0.0, 1.0))
    frames.insert(14, obs(0.7, bars=2))        # seen on the floor at 0.7 s
    assert feed(tracker, frames) == [],         "the window is long enough at 0.8 s, but he was downed 0.1 s ago"


def test_mostly_standing_is_enough_but_not_mostly_downed():
    tracker = ReviveTracker()
    frames = []
    for i in range(30):
        frames.append(obs(i / 20.0, standing=True, bars=None if i % 5 else 3))
    assert feed(tracker, frames) == [], "80% alive with a downed reading every 5th"


def test_a_confirm_is_reported_once():
    tracker = ReviveTracker()
    assert len(feed(tracker, stream(STRONG_ALIVE, 0.0, 3.0))) == 1


def test_reset_clears_everything():
    tracker = ReviveTracker()
    feed(tracker, stream(STRONG_ALIVE, 0.0, 0.6))
    tracker.reset()
    assert tracker.candidate_age == 0.0
    assert feed(tracker, stream(STRONG_ALIVE, 5.0, 0.5)) == []


# -- who is being watched ----------------------------------------------------
def test_only_the_allies_that_triggered_are_watched():
    """The slots that took part are locked when the song starts; a row that shows
    up later is somebody else and cannot end this song."""
    arb = EventArbiter()
    arb.start_song([0])
    assert arb.observe(obs(0.0, standing=True, slot=1)) is False
    assert arb.observe(obs(0.0, standing=True, slot=0)) is False   # too short yet


def test_a_new_song_starts_every_slate_clean():
    arb = EventArbiter()
    arb.start_song([0, 1])
    feed(arb.tracker(0), stream(STRONG_ALIVE, 0.0, 0.6))
    arb.start_song([0, 1])
    assert arb.tracker(0).candidate_age == 0.0
    assert len(arb.tracker(0)._samples) == 0


def test_the_locked_set_survives_a_row_that_stops_being_read():
    """A row going unreadable is not a row leaving the run: identity must not
    drift onto whoever happens to be readable next."""
    arb = EventArbiter()
    arb.start_song([0, 1])
    assert arb.locked == (0, 1)
    for t in stream(STRONG_ALIVE, 0.0, 1.0):
        arb.observe(t)
    assert arb.locked == (0, 1)
