"""The party size must follow the game, and must not be guessed from one frame."""
import pytest

from nrdoom.party import PartyCounter


def test_first_frame_sets_the_count_at_once():
    pc = PartyCounter(hold_frames=30)
    assert pc.update(2) == 2


def test_more_allies_take_effect_immediately():
    """With no corroboration asked for, the count follows every frame."""
    pc = PartyCounter(hold_frames=30)
    pc.update(1)
    assert pc.update(2) == 2


def test_losing_an_ally_needs_the_shorter_count_to_persist():
    pc = PartyCounter(hold_frames=3)
    pc.update(2)
    assert pc.update(1) == 2
    assert pc.update(1) == 2
    assert pc.update(1) == 1


def test_one_frame_of_dropout_does_not_lower_the_count():
    """Under-counting is the dangerous direction: it would let the music start
    while the missing ally is still standing."""
    pc = PartyCounter(hold_frames=2)
    pc.update(2)
    for _ in range(10):
        pc.update(1)
        pc.update(2)
    assert pc.count == 2


def test_reading_nothing_is_not_evidence_that_the_allies_left():
    """A panel that yields neither gauge nor health bar says the picture was
    unhelpful, not that the roster shrank.

    Measured on the user's own footage: after every revive the panel reads blank
    for up to three seconds, and all seven party drops across four recordings came
    from that and none from a positively shorter list. The plan is explicit that a
    failed read must not turn 2 allies into 1."""
    pc = PartyCounter(hold_frames=2)
    pc.update(2)
    for _ in range(20):
        assert pc.update(0) == 2
    assert pc.count == 2


def test_a_shorter_list_that_is_really_there_still_lowers_the_count():
    """Someone who left has to be believed, or the tool waits for nobody."""
    pc = PartyCounter(hold_frames=2)
    pc.update(2)
    assert pc.update(1) == 2
    assert pc.update(1) == 1


def test_a_blank_panel_breaks_the_run_toward_a_shorter_count():
    """The shorter count needs its own `hold_frames` of evidence, not a tally
    collected on either side of a stretch where nothing could be read."""
    pc = PartyCounter(hold_frames=3)
    pc.update(2)
    pc.update(1)
    pc.update(1)
    pc.update(0)
    assert pc.count == 2
    pc.update(1)
    assert pc.count == 2, "the two earlier frames do not carry over"
    pc.update(1)
    pc.update(1)
    assert pc.count == 1


def test_a_solo_run_never_invents_allies():
    pc = PartyCounter(hold_frames=2)
    for _ in range(5):
        assert pc.update(0) == 0


def test_reset_forgets_the_party():
    pc = PartyCounter(hold_frames=2)
    pc.update(2)
    pc.reset()
    assert pc.count == 0
    assert pc.update(1) == 1


@pytest.mark.parametrize("hold", [1, 5])
def test_a_steady_count_settles_to_that_count(hold):
    pc = PartyCounter(hold_frames=hold)
    pc.update(3)
    for _ in range(hold + 2):
        assert pc.update(1) in (1, 3)
    assert pc.count == 1


# -- a rise above an established count is the ghost's way in ------------------
# On the user's own 276 s recording one frame out of 5533 read a 1-segment gauge on
# the empty second row (confidence 0.21). That single frame held the count at 2 for
# a full 60 frames, and the ally on row 0 was at 3 bars for 24 of them: the music
# should have started and did not. So a count of "one more ally than we knew about"
# has to be seen more than once before it is believed.
GATED = dict(hold_frames=60, rise_samples=2, rise_frames=8)


def test_one_frame_of_an_extra_ally_does_not_raise_the_count():
    pc = PartyCounter(**GATED)
    pc.update(1)
    for rows in [2] + [1] * 20:
        assert pc.update(rows) == 1
    assert pc.count == 1


def test_a_second_ally_is_adopted_once_the_row_shows_up_twice():
    pc = PartyCounter(**GATED)
    pc.update(1)
    assert pc.update(2) == 1
    assert pc.pending is True, "the engine has to know the count is not settled yet"
    assert pc.update(2) == 2
    assert pc.pending is False


def test_the_rise_needs_the_sightings_close_together():
    """A row that turns up again only after the window has closed is not being
    followed; it is the same one-off flake wearing a different hat."""
    pc = PartyCounter(hold_frames=60, rise_samples=2, rise_frames=4)
    pc.update(1)
    for rows in [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2]:
        pc.update(rows)
    assert pc.count == 1


def test_a_row_that_flickers_in_and_out_still_gets_believed():
    """The second hunter can be behind a rock. Refusing to ever adopt them would
    leave the tool waiting for nobody and playing on their own."""
    pc = PartyCounter(**GATED)
    pc.update(1)
    for _ in range(6):
        pc.update(1)
        pc.update(2)
    assert pc.count == 2


def test_acquiring_the_party_from_nothing_is_never_made_to_wait():
    """Arming mid-run has to see the allies on the frame they appear; the gate is
    only for growing a count that is already standing."""
    pc = PartyCounter(**GATED)
    assert pc.update(1) == 1
    pc.reset()
    assert pc.update(2) == 2
    assert pc.pending is False
