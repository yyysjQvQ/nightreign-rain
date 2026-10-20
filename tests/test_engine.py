from pathlib import Path

import cv2
import pytest

from fakes import FakePlayer, FakeSource
from nrdoom.engine import Engine
from nrdoom.settings import Settings, Slot

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot
GOLDEN_SLOTS = [Slot(128, 247, 27), Slot(128, 358, 27)]


def frame(name="golden_both_3bars.jpg"):
    return shot(name)


def build(img=None, slots=GOLDEN_SLOTS, hold=3, **kw):
    st = Settings(slots=list(slots), hold_frames=hold, **kw)
    src = FakeSource(img)
    player = FakePlayer()
    eng = Engine(st, source=src, player=player)
    return eng, src, player, st


def test_cannot_arm_without_a_live_feed(qapp):
    eng, src, player, st = build()
    errors = []
    eng.error.connect(errors.append)
    assert eng.set_listening(True) is False
    assert "窗口" in errors[0]


def test_arming_succeeds_with_both(qapp):
    eng, src, player, st = build(frame())
    assert eng.set_listening(True) is True
    assert eng.listening is True


def test_fires_once_the_hold_completes(qapp):
    eng, src, player, st = build(frame(), hold=3)
    eng.set_listening(True)
    for _ in range(2):
        eng._tick()
    assert player.calls == []
    eng._tick()
    assert player.calls == ["doom.mp3"]


def paint_health_bar(img, cy):
    """Draw what the game draws for an ally who is on their feet: a saturated red
    fill starting at the trough's left end (0.082 W) and running right, in the band
    just above the row's centre. Measured on the user's own capture."""
    w = img.shape[1]
    img[int(cy - 0.0080 * w):int(cy - 0.0030 * w),
        int(w * 0.082):int(w * 0.140)] = (30, 45, 185)


def test_does_not_fire_when_only_one_ally_is_at_three(qapp):
    eng, src, player, st = build(frame(), hold=1)
    # The second ally is standing: no gauge, and the health bar says so.
    img = frame()
    img[332:386, 100:158] = (38, 37, 65)
    paint_health_bar(img, 358)
    src.set_frame(img)
    eng.set_listening(True)
    for _ in range(20):
        eng._tick()
    assert eng.party.count == 2
    assert player.calls == []


def test_does_not_refire_while_the_condition_holds(qapp):
    eng, src, player, st = build(frame(), hold=1)
    eng.set_listening(True)
    for _ in range(20):
        eng._tick()
    assert player.calls == ["doom.mp3"]


def test_rearms_after_the_condition_clears(qapp):
    eng, src, player, st = build(frame(), hold=1, revive_hold_frames=1,
                                 stop_policy="v1")
    eng.set_listening(True)
    eng._tick()
    assert len(player.calls) == 1

    src.set_frame(stand_up(frame(), 358))
    eng._tick()
    assert player.stops == 1                      # seen standing, so the song ended

    src.set_frame(frame())
    eng._tick()
    assert len(player.calls) == 2


def test_readings_are_published_for_the_ui(qapp):
    eng, src, player, st = build(frame(), hold=1)
    seen = []
    eng.readings_changed.connect(seen.append)
    eng.set_listening(True)
    eng._tick()
    assert [None if r is None else r.bars for r in seen[-1]] == [3, 3]


def test_disarming_clears_the_readings(qapp):
    eng, src, player, st = build(frame(), hold=1)
    eng.set_listening(True)
    eng._tick()
    eng.set_listening(False)
    assert eng.readings == []


def test_missing_music_reports_instead_of_crashing(qapp):
    eng, src, player, st = build(frame(), hold=1)
    player.name = None
    errors = []
    eng.error.connect(errors.append)
    eng.set_listening(True)
    eng._tick()
    assert errors and "音乐" in errors[0]


def test_reconfigure_rebuilds_the_trigger(qapp):
    eng, src, player, st = build(frame(), hold=3)
    st.hold_frames = 1
    eng.reconfigure()
    assert eng.trigger.hold_frames == 1


def test_allies_follows_the_slot_count(qapp):
    eng, src, player, st = build(frame(), slots=GOLDEN_SLOTS[:1])
    assert eng.trigger.allies == 1


# -- stop-on-revive: the song survives a revive in progress, ends when someone stands
def erase_ring(img, top):
    """Paint a HUD gauge out so that slot reads as 'not downed'."""
    img[top:top + 54, 100:158] = (38, 37, 65)
    return img


def stand_up(img, cy):
    """What a revive actually looks like: the gauge is gone *and* the health bar
    is back. Only the first of those also happens when a read fails, so the stop
    has to want both."""
    erase_ring(img, cy - 26)
    paint_health_bar(img, cy)
    return img


def arm_and_fire(eng, player):
    eng.set_listening(True)
    eng._tick()
    assert player.calls == ["doom.mp3"]


def test_the_engine_keeps_what_each_slot_said_while_playing(qapp):
    """§10's explainability, for the interface: while a track is running the engine
    keeps the per-slot evidence it just acted on, and while it is not, it keeps
    nothing (building them costs a crop per unread slot)."""
    eng, src, player, st = build(frame(), hold=1, party_hold_frames=1)
    eng.set_listening(True)
    assert eng.observations == [], "nothing is playing yet"
    eng._tick()
    assert player.calls == ["doom.mp3"]
    src.set_frame(frame())
    eng._tick()
    assert [o.evidence() for o in eng.observations] == ["down", "down"]


def test_a_cutscene_bar_does_not_move_the_geometry_or_the_roster(qapp):
    """§5.3, measured at 100.25 s of the user's own `bili-arrives`: thirteen frames
    of cinematic black crossed the top of the frame, the vote followed them, and
    that threw away the positions it had found *and* reset the party to zero."""
    from nrdoom.hud_tracker import LOST
    real = frame("real_duo_3bar.jpg")
    eng, src, player, st = build(real, slots=[], hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(4):
        src.set_frame(real)
        eng._tick()
    assert [round(s.cy) for s in st.slots] == [358, 483]
    assert eng.party.count == 1

    bar = real.copy()
    bar[:270] = 0                                # above the panel, below the cap
    for _ in range(13):
        src.set_frame(bar)
        eng._tick()
    assert eng._pose_state == LOST, "the dispute was never noticed"
    assert [round(s.cy) for s in st.slots] == [358, 483], "a 0.6 s overlay moved it"
    assert eng.party.count == 1, "and reset the roster"

    for _ in range(40):                          # a change that stays is adopted
        src.set_frame(bar)
        eng._tick()
    assert [round(s.cy) for s in st.slots] != [358, 483]


def test_losing_the_picture_box_does_not_end_the_song(qapp):
    """Plan §5.3: `LOST` is about us, not about the game.

    A black overlay moves the letterbox measurement, and between the frame where the
    confirmed anchor stopped being seen and the frame where the vote adopts the new
    box, the geometry is not trustworthy. An ally whose gauge cannot be found from
    there is not an ally who stood up.

    The engine is driven off an explicit clock here: the revive rule is written in
    seconds, and on `time.monotonic()` the outcome would depend on how fast this
    machine happens to spin the loop."""
    from nrdoom.hud_tracker import LOST
    from nrdoom.replay import ManualClock
    clock = ManualClock()
    st = Settings(slots=list(GOLDEN_SLOTS), hold_frames=1, revive_min_samples=6,
                  revive_min_evidence_s=0.1, revive_lapse_s=10.0,
                  revive_window_s=10.0, revive_min_alive_share=0.3)
    src, player = FakeSource(frame()), FakePlayer()
    eng = Engine(st, source=src, player=player, clock=clock)
    eng.set_listening(True)
    eng._tick()
    assert player.calls == ["doom.mp3"]
    for _ in range(20):
        clock.advance(0.05)
        src.set_frame(frame())                 # fill the vote with the real box
        eng._tick()
    assert eng._pose_state == "LOCKED"
    lost = stand_up(stand_up(frame(), 247), 358)
    lost[:110] = 0                             # a band the measurement cannot ignore
    seen = set()
    for _ in range(12):
        clock.advance(0.05)
        src.set_frame(lost)
        eng._tick()
        seen.add(eng._pose_state)
    assert LOST in seen, "the tracker never admitted it had lost the anchor"
    assert player.stops == 0, "standing evidence taken while lost proved nothing"
    assert eng.watcher.watching is True


def test_dropping_to_fewer_segments_does_not_stop_the_track(qapp):
    eng, src, player, st = build(frame(), hold=1, revive_hold_frames=2)
    arm_and_fire(eng, player)
    assert eng.watcher.watching is True
    mixed = frame("synth_mixed_3_1.png")
    src.set_frame(mixed)
    for _ in range(6):
        eng._tick()
    assert player.stops == 0


def test_an_ally_standing_up_stops_the_track(qapp):
    eng, src, player, st = build(frame(), hold=1, revive_hold_frames=3,
                                 stop_policy="v1")
    arm_and_fire(eng, player)
    src.set_frame(stand_up(frame(), 358))
    for _ in range(2):
        eng._tick()
    assert player.stops == 0            # still inside the debounce
    eng._tick()
    assert player.stops == 1
    assert eng.watcher.watching is False


def test_a_panel_that_only_stopped_reading_does_not_stop_the_track(qapp):
    """The panel over a purple sky can fail the spoke test and read as nothing at
    all. A missing reading is not an ally standing up, so neither the revive watch
    nor the party count may cut the track off - that is the false stop the user hit
    every time the camera moved."""
    eng, src, player, st = build(frame(), hold=1, party_hold_frames=1)
    arm_and_fire(eng, player)
    blank = erase_ring(erase_ring(frame(), 222), 332)     # both gauges gone, no bars
    src.set_frame(blank)
    for _ in range(20):
        eng._tick()
    assert eng.party.count == 2, "reading nothing is not the allies leaving"
    assert player.stops == 0
    assert eng.watcher.watching is True


def test_stopping_the_watch_disarms_repeats(qapp):
    eng, src, player, st = build(frame(), hold=1, revive_hold_frames=1,
                                 stop_policy="v1")
    arm_and_fire(eng, player)
    src.set_frame(stand_up(frame(), 358))
    eng._tick()
    assert player.stops == 1
    src.set_frame(frame())
    eng._tick()
    assert player.stops == 1            # watcher already disarmed


def test_pause_stops_the_track_and_blocks_auto_play(qapp):
    eng, src, player, st = build(frame(), hold=1)
    arm_and_fire(eng, player)
    assert eng.set_paused(True) is True
    assert player.stops == 1
    player.calls.clear()
    for _ in range(5):
        eng._tick()
    assert player.calls == []


def test_resuming_from_pause_allows_firing_again(qapp):
    eng, src, player, st = build(frame(), hold=1)
    eng.set_listening(True)
    eng.set_paused(True)
    eng.set_paused(False)
    assert eng.paused is False
    eng._tick()
    assert player.calls == ["doom.mp3"]


def test_arming_again_clears_a_stale_pause(qapp):
    eng, src, player, st = build(frame(), hold=1)
    eng.set_listening(True)
    eng.set_paused(True)
    eng.set_listening(False)
    eng.set_listening(True)
    assert eng.paused is False


def test_disarming_stops_a_running_track(qapp):
    eng, src, player, st = build(frame(), hold=1)
    arm_and_fire(eng, player)
    eng.set_listening(False)
    assert player.stops == 1
    assert eng.watcher.watching is False


def test_stopped_signal_reaches_the_ui(qapp):
    eng, src, player, st = build(frame(), hold=1, revive_hold_frames=1,
                                 stop_policy="v1")
    seen = []
    eng.stopped.connect(lambda: seen.append(1))
    arm_and_fire(eng, player)
    src.set_frame(stand_up(frame(), 358))
    eng._tick()
    assert seen == [1]


# -- duo mode: a single ally --------------------------------------------------
DUO_SLOTS = [Slot(128, 247, 27)]


def duo_engine(**kw):
    """synth_rows_duo.png is a real duo: the second row has no portrait at all."""
    return build(frame("synth_rows_duo.png"), slots=list(DUO_SLOTS), **kw)


def test_duo_arms_with_a_single_slot(qapp):
    eng, src, player, st = duo_engine(hold=1)
    errors = []
    eng.error.connect(errors.append)
    assert eng.set_listening(True) is True
    assert errors == []
    assert eng.trigger.allies == 1


def test_duo_fires_when_the_only_ally_hits_three(qapp):
    eng, src, player, st = duo_engine(hold=1)
    eng.set_listening(True)
    eng._tick()
    assert player.calls == ["doom.mp3"]


def test_duo_does_not_fire_at_two_bars(qapp):
    eng, src, player, st = build(frame("synth_both_2bars.png"),
                                 slots=list(DUO_SLOTS), hold=1)
    eng.set_listening(True)
    for _ in range(5):
        eng._tick()
    assert player.calls == []


def test_duo_stops_when_that_ally_is_revived(qapp):
    eng, src, player, st = duo_engine(hold=1, revive_hold_frames=2, stop_policy="v1")
    eng.set_listening(True)
    eng._tick()
    assert player.calls == ["doom.mp3"]
    src.set_frame(stand_up(frame("synth_rows_duo.png"), 247))   # the ally stood up
    for _ in range(1):
        eng._tick()
    assert player.stops == 0
    eng._tick()
    assert player.stops == 1


def test_solo_never_fires_and_never_crashes(qapp):
    """Arming needs nothing from the user now, so solo has to be caught by the
    row count instead of by a missing calibration."""
    eng, src, player, st = build(frame("synth_rows_solo.png"), slots=[], hold=1,
                                 party_hold_frames=1)
    assert eng.set_listening(True) is True
    for _ in range(5):
        eng._tick()
    assert eng.party.count == 0
    assert player.calls == []


def test_armed_read_adopts_a_drifted_calibration(qapp):
    """If the game window shifts, the stored positions should follow, not go silent."""
    drifted = [Slot(80, 151, 14.5), Slot(80, 218, 14.3)]
    eng, src, player, st = build(frame("real_dusk_3bars.jpg"), slots=drifted, hold=1)
    changed = []
    eng.calibrationChanged.connect(lambda: changed.append(1))
    eng.set_listening(True)
    eng._tick()
    assert changed == [1]
    assert abs(st.slots[0].cx - 77) <= 2 and abs(st.slots[0].cy - 148) <= 2
    assert abs(st.slots[1].cx - 77) <= 2 and abs(st.slots[1].cy - 215) <= 2
    assert [r.bars for r in eng.readings] == [3, 3]


def test_a_weak_off_centre_read_does_not_move_the_geometry(qapp, monkeypatch):
    """A ghost gauge on the empty second row moved the stored position 8 px and it
    never came back, which cost 3303 frames of the user's own recording. The jitter
    search exists to claw back a few pixels from a real gauge - 1.0 on the first
    try, 0.72 three pixels out - so adopting a move asks for that much confidence.

    Measured on the same recording, where row 1 is empty for all 277 s and so every
    read there is a false positive by construction: nine ghosts, none above 0.55.
    """
    from nrdoom.detector import Ring

    weak = Ring(cx=128.0, cy=364.0, radius=27.0, bars=1, confidence=0.28)
    monkeypatch.setattr("nrdoom.engine.detect_with_recovery",
                        lambda frame, slots: ([None, weak], [1]))
    eng, src, player, st = build(frame(), slots=GOLDEN_SLOTS, hold=1)
    eng.set_listening(True)
    eng._tick()
    assert st.slots[1].cy == 358, "the ghost must not pull the row down onto it"


def test_a_strong_off_centre_read_still_moves_it(qapp, monkeypatch):
    """The other half: a window that really did shift has to be recoverable."""
    from nrdoom.detector import Ring

    real = Ring(cx=128.0, cy=361.0, radius=27.0, bars=3, confidence=0.72)
    monkeypatch.setattr("nrdoom.engine.detect_with_recovery",
                        lambda frame, slots: ([None, real], [1]))
    eng, src, player, st = build(frame(), slots=GOLDEN_SLOTS, hold=1)
    eng.set_listening(True)
    eng._tick()
    assert st.slots[1].cy == 361.0


def test_exact_calibration_does_not_churn(qapp):
    eng, src, player, st = build(frame("real_dusk_3bars.jpg"),
                                 slots=[Slot(77, 148, 14.5), Slot(77, 215, 14.3)], hold=1)
    changed = []
    eng.calibrationChanged.connect(lambda: changed.append(1))
    eng.set_listening(True)
    for _ in range(3):
        eng._tick()
    assert changed == []


def test_a_position_sixteen_pixels_off_finds_its_way_back(qapp):
    """The dead zone the v1 report left behind: a gauge read is lost by ~16 px of
    error, while the aligner used to call anything within 46 px "already right" and
    never move. In that band the tool saw nothing, counted no allies, and stayed
    there - deaf for the rest of the run.

    Measured with `tools/shift_probe.py`; the split that fixes it is
    `REALIGN_SETTLE_FRAC_W`, kept separate from the portrait-bias threshold."""
    from nrdoom.detector import detect_spots
    drifted = [Slot(s.cx, s.cy + 16, s.radius) for s in GOLDEN_SLOTS]
    img = frame()
    assert [r for r in detect_spots(img, [(s.cx, s.cy, s.radius) for s in drifted])] \
        == [None, None], "16 px off has to be blind, or this tests nothing"

    eng, src, player, st = build(img, slots=drifted, hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(10):
        eng._tick()
    assert [r.bars if r else None for r in eng.readings] == [3, 3], \
        "still blind 10 frames in: the dead zone is open"
    assert all(abs(s.cy - g.cy) <= 6 for s, g in zip(st.slots, GOLDEN_SLOTS))


def test_a_position_five_pixels_off_finds_its_way_back(qapp):
    """The same dead zone one size down, measured on the user's own 2560x1600 frame.

    A gauge is lost by 5 px of error (`tools/offset_read.py` sweeps it: 1.00 at the
    centre, 0.69 at 4 px, nothing at 5), so a threshold that calls 9 px "already
    right" leaves a band where the tool reads nothing and refuses to look elsewhere.
    """
    from nrdoom.detector import detect_spots
    drifted = [Slot(s.cx, s.cy + 5, s.radius) for s in GOLDEN_SLOTS]
    img = frame()
    assert detect_spots(img, [(s.cx, s.cy, s.radius) for s in drifted]) \
        == [None, None], "5 px off has to be blind, or this tests nothing"

    eng, src, player, st = build(img, slots=drifted, hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(10):
        eng._tick()
    assert [r.bars if r else None for r in eng.readings] == [3, 3], \
        "still blind 10 frames in: the band is still open"


def test_a_correct_panel_survives_a_red_backdrop(qapp):
    """The floor under `REALIGN_SETTLE_FRAC_W`, so it cannot be lowered again.

    On this capture the portrait peaks sit 4.8 px off a window that is already on
    the ring. Moving for that disagreement turns a one-segment ally into a two,
    which is a wrong answer instead of a late one - so the threshold has to stay
    above the peak's own bias. Measured by `tools/wander_probe.py`.
    """
    from nrdoom.detector import detect_spots, gauge_spots, hud_rect

    img = frame("real_red_backdrop_downed.png")
    true = gauge_spots(img.shape, rect=hud_rect(img))
    want = [None if r is None else r.bars for r in detect_spots(img, true)]
    assert want == [1, 2], "the truth has to be a mixed count, or this tests nothing"

    eng, src, player, st = build(img, slots=[Slot(cx, cy, r) for cx, cy, r in true],
                                 hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(60):
        eng._tick()
    assert [None if r is None else r.bars for r in eng.readings] == want
    drift = max(abs(s.cy - t[1]) for s, t in zip(st.slots, true))
    assert drift < 1.0, "the aligner walked off a position that was already right"


def test_disarmed_read_does_not_rewrite_the_calibration(qapp):
    """Calibration must only move when the user has committed to it by arming."""
    drifted = [Slot(80, 151, 14.5), Slot(80, 218, 14.3)]
    eng, src, player, st = build(frame("real_dusk_3bars.jpg"), slots=drifted)
    eng._tick()
    assert st.slots == drifted


# -- party size follows the rows on screen, not the calibration ---------------
def test_a_row_count_change_is_published(qapp):
    eng, src, player, st = build(frame(), hold=1)
    seen = []
    eng.party_changed.connect(seen.append)
    eng.set_listening(True)
    eng._tick()
    assert seen == [2]


def test_a_one_frame_ghost_row_does_not_hold_the_music_back(qapp, monkeypatch):
    """The user's own recording, 103.4 s: the empty second row read a 1-segment
    gauge for a single frame, the party went to 2, and it stayed there for the 60
    frames it takes to fall back - through the 24 frames the real ally spent at
    three bars. The music should have started and did not.

    So a bigger party has to be seen more than once, and while it is only half
    proved the count is not settled enough to either fire on or withhold from."""
    import nrdoom.engine as eng_mod
    eng, src, player, st = build(frame("synth_rows_duo.png"), hold=5,
                                 party_hold_frames=60)
    real = eng_mod.count_rows
    tick = []

    def rows(frame_, rect, spots):
        tick.append(1)
        return 2 if len(tick) == 3 else real(frame_, rect, spots)

    monkeypatch.setattr(eng_mod, "count_rows", rows)
    eng.set_listening(True)
    for _ in range(3):
        eng._tick()
    assert eng.party.count == 1, "one fluke frame is not a second ally"
    assert player.calls == [], "and the hold must not be spent arguing about it"
    for _ in range(5):
        eng._tick()
    assert player.calls == ["doom.mp3"]


def test_fire_requires_the_second_ally_until_they_leave(qapp):
    """Calibrated as a trio, then one hunter quits: the last ally going downed has
    to be enough on its own, because nobody is coming."""
    eng, src, player, st = build(frame(), hold=1, party_hold_frames=3,
                                 revive_hold_frames=1, stop_policy="v1")
    eng.set_listening(True)
    for _ in range(3):
        eng._tick()
    assert player.calls == ["doom.mp3"]

    # Only a real revive may hand the trigger back: a second play_next over the top
    # of a song that is still going is the restart-from-zero the user reported.
    src.set_frame(stand_up(frame(), 358))
    eng._tick()
    assert player.stops == 1
    assert player.calls == ["doom.mp3"], "ending the song must not start another"
    player.calls.clear()

    src.set_frame(frame("synth_rows_duo.png"))
    for _ in range(2):
        eng._tick()
    assert eng.party.count == 2, "one frame of a shorter list must not lower the bar"
    assert player.calls == []

    for _ in range(3):
        eng._tick()
    assert eng.party.count == 1
    assert player.calls == ["doom.mp3"]


def test_the_song_survives_the_panel_going_blank_entirely(qapp):
    """A row count of zero is what a failed read looks like, and it is also what a
    hidden HUD looks like, so it proves nothing about the roster. Stopping the track
    there was the false cut-off the user hit; the pause button and a real revive are
    how the song ends."""
    eng, src, player, st = build(frame(), hold=1, party_hold_frames=1)
    arm_and_fire(eng, player)
    stopped = []
    eng.stopped.connect(lambda: stopped.append(1))
    src.set_frame(frame("synth_rows_solo.png"))
    for _ in range(5):
        eng._tick()
    assert eng.party.count == 2, "seeing nothing is not the allies leaving"
    assert stopped == []
    assert player.stops == 0


def test_solo_never_starts_the_music(qapp):
    eng, src, player, st = build(frame("synth_rows_solo.png"), hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(10):
        eng._tick()
    assert player.calls == []
    assert eng.party.count == 0


def test_rearming_recounts_the_party(qapp):
    eng, src, player, st = build(frame("synth_rows_duo.png"), hold=1, party_hold_frames=1)
    eng.set_listening(True)
    eng._tick()
    assert eng.party.count == 1
    eng.set_listening(False)
    assert eng.party.count == 0


def test_still_fires_once_the_extra_ally_is_calibrated(qapp):
    eng, src, player, st = build(frame(), slots=GOLDEN_SLOTS[:1], hold=1,
                                 party_hold_frames=1)
    eng.settings.slots = list(GOLDEN_SLOTS)
    eng.reconfigure()
    eng.set_listening(True)
    eng._tick()
    assert player.calls == ["doom.mp3"]


# -- adopting a gauge the calibration missed, without asking the user ---------
def test_a_duo_frame_does_not_make_the_second_spot_read_something(qapp):
    """Two spots are always watched, so on a duo the empty one must stay empty
    rather than invent a gauge out of the backdrop."""
    eng, src, player, st = build(frame("synth_rows_duo.png"), slots=[],
                                 hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(10):
        eng._tick()
    assert len(st.slots) == 2
    assert [None if r is None else r.bars for r in eng.readings] == [3]
    assert player.calls == ["doom.mp3"]


def test_the_app_arms_and_plays_with_no_calibration_at_all(qapp):
    """Nothing to click: the panel is where the panel is."""
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    errors = []
    eng.error.connect(errors.append)
    assert eng.set_listening(True) is True, errors
    eng._tick()
    assert errors == []
    assert player.calls == ["doom.mp3"]


def test_changing_resolution_seeds_the_new_geometry_without_asking(qapp):
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    eng.set_listening(True)
    eng._tick()
    assert [round(s.cy) for s in st.slots] == [247, 358]
    player.calls.clear()

    src.set_frame(frame("real_dusk_3bars.jpg"))
    eng._last_frame_id = -1
    eng._tick()
    assert [round(s.cy) for s in st.slots] == [148, 215]


def test_each_resolution_keeps_its_own_geometry(qapp):
    """The in-game UI size moves the panel, so a geometry that reads well is worth
    keeping - but only for the resolution it was found at, and a position that
    already reads well must not be churned every frame."""
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    # 2 px off. Small errors are the jitter search's job, not the aligner's, so the
    # expectation is that it converges on the ring and then leaves it alone.
    st.slots = [Slot(128, 245, 27), Slot(128, 356, 27)]
    eng.set_listening(True)
    for _ in range(3):
        eng._tick()
    assert [(round(s.cx), round(s.cy)) for s in st.slots] == [(128, 247), (128, 358)]
    assert player.calls == ["doom.mp3"]
    eng._tick()
    assert [round(s.cy) for s in st.slots] == [247, 358], "moved again: that is churn"

    src.set_frame(frame("real_dusk_3bars.jpg"))
    eng._last_frame_id = -1
    eng._tick()
    assert [round(s.cy) for s in st.slots] == [148, 215]

    src.set_frame(frame())
    eng._last_frame_id = -1
    eng._tick()
    got = [round(s.cy) for s in st.slots]
    assert got == [247, 358], "the 1366-wide geometry came back to a 2276 frame: %s" % got


def test_a_two_pixel_error_is_left_alone_but_a_five_pixel_one_is_not(qapp):
    """Where the settle threshold actually sits, on the user's own measurements.

    A gauge reads at 4 px out and is gone by 5 (`tools/offset_read.py`), so 5 px has
    to be worth moving for; the portrait peaks themselves sit ~4.8 px off a panel
    that is already right, so 4 px must not be.
    """
    from nrdoom.detector import detect_spots

    img = frame()
    for dy, should_move in ((4, False), (5, True)):
        drifted = [Slot(s.cx, s.cy + dy, s.radius) for s in GOLDEN_SLOTS]
        blind = detect_spots(img, [(s.cx, s.cy, s.radius) for s in drifted]) \
            == [None, None]
        assert blind is should_move, "%d px off: blind=%s" % (dy, blind)


# -- teaching the geometry to find the panel on its own -----------------------
def test_the_real_duo_capture_counts_one_ally_and_fires(qapp):
    """The user's own 2560x1600 screenshot: a 16:9 picture inside a 16:10 frame,
    one ally, and that ally downed on their last segment. This is the exact frame
    the tool is supposed to answer, end to end."""
    eng, src, player, st = build(frame("real_duo_3bar.jpg"), slots=[], hold=1,
                                 party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(3):
        eng._tick()
    assert eng.party.count == 1
    assert [r.bars for r in eng.readings] == [3]
    assert player.calls == ["doom.mp3"]


def test_the_real_trio_capture_stays_quiet_while_both_stand(qapp):
    """Same machine, two allies on their feet: the count has to see both of them
    even though neither has a gauge to read."""
    eng, src, player, st = build(frame("real_alive_panel.png"), slots=[], hold=1,
                                 party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(10):
        eng._tick()
    assert eng.party.count == 2
    assert eng.readings == [None, None]
    assert player.calls == []


def test_a_panel_that_is_already_right_does_not_move(qapp):
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(6):
        eng._tick()
    assert [round(s.cy) for s in st.slots] == [247, 358]
    assert player.calls == ["doom.mp3"]


def test_a_moved_panel_then_snaps_onto_the_real_gauges(qapp):
    """Peak finding is coarse (up to 28 px biased); the gauges themselves are what
    settles the last bit. Start 58 px out and end up on the sprite."""
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    # 45 px out: close enough that the grid still believes it, far enough that
    # the portraits say otherwise.
    st.slots = [Slot(128, 247 + 45, 27), Slot(128, 358 + 45, 27)]
    eng.set_listening(True)
    for _ in range(4):
        eng._tick()
    assert [round(s.cy) for s in st.slots] == [247, 358]
    assert [None if r is None else r.bars for r in eng.readings] == [3, 3]
    assert player.calls == ["doom.mp3"]


def test_a_bad_first_picture_box_does_not_pin_the_geometry(qapp):
    """A recording that opens on a fade measures 312 px of black where the game
    later shows 80, and the seed lands 232 px below the panel. The next real frame
    has to pull the geometry back or the whole session reads nothing."""
    real = frame("real_duo_3bar.jpg")
    fade = real.copy()
    fade[:232] = 0
    eng, src, player, st = build(fade, slots=[], hold=1, party_hold_frames=1)
    eng.set_listening(True)
    eng._tick()
    assert st.slots[0].cy > 500, "the fade really does seed the wrong place"

    for _ in range(3):               # a live feed keeps delivering, same picture
        src.set_frame(real)
        eng._tick()
    assert [round(s.cy) for s in st.slots] == [358, 483]
    assert [None if r is None else r.bars for r in eng.readings] == [3]
    assert player.calls == ["doom.mp3"]


def test_a_move_that_reads_nothing_is_not_taken(qapp, monkeypatch):
    """On every frame of the user's own recording the portraits claimed the panel
    was elsewhere, and the spots went: 130 moves, and 230 frames of a downed ally
    the grid could read while the spots could not. A position with no gauge in it
    is not evidence that the panel is there."""
    monkeypatch.setattr("nrdoom.engine.row_peaks",
                        lambda frame, ys, rect=None: [(y + 200, 0.5) for y in ys])
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(6):
        eng._tick()
    assert [round(s.cy) for s in st.slots] == [247, 358], "the grid stays put"
    assert player.calls == ["doom.mp3"]


def test_the_realignment_is_not_recomputed_every_frame(qapp, monkeypatch):
    """Looking at the gauges to nail the last few pixels costs about a second, so
    one move pays for it at most REALIGN_TRIES times."""
    calls = []
    import nrdoom.detector as real
    monkeypatch.setattr("nrdoom.engine.propose_rings",
                        lambda frame: calls.append(1) or real.propose_rings(frame))
    # A panel 60 px below the grid with both rows agreeing on it, which is what the
    # settle pass is for. How the rows get found is tests/test_rows' business.
    monkeypatch.setattr("nrdoom.engine.row_peaks",
                        lambda frame, ys, rect=None: [(y + 60, 0.5) for y in ys])
    eng, src, player, st = build(frame(), slots=[], hold=1, party_hold_frames=1)
    eng.set_listening(True)
    for _ in range(40):
        eng._tick()
    assert 0 < len(calls) <= 3, len(calls)


def test_it_does_not_start_a_second_song_over_the_one_already_playing(qapp):
    """The false restart the user heard: the geometry jumped, the readings went
    blank, the trigger re-armed, and the next qualifying frame called play_next()
    again while the first song was still playing - so it began from zero. One song
    at a time: while the revive watch is running, there is nothing to trigger."""
    eng, src, player, st = build(frame(), hold=1, party_hold_frames=1)
    arm_and_fire(eng, player)
    blank = erase_ring(erase_ring(frame(), 222), 332)
    src.set_frame(blank)
    for _ in range(6):
        eng._tick()
    assert player.calls == ["doom.mp3"]          # nothing stopped, nothing started
    src.set_frame(frame())
    for _ in range(6):
        eng._tick()
    assert player.calls == ["doom.mp3"], player.calls
    assert player.stops == 0
