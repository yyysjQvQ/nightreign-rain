import pytest

from nrdoom.revive import ReviveWatcher


def feed(w, seq):
    """Push a sequence of per-slot downed flags; return indices where it said stop."""
    return [i for i, flags in enumerate(seq) if w.update(flags)]


def make(hold=3):
    w = ReviveWatcher(hold_frames=hold)
    w.start()
    return w


def test_losing_segments_during_a_revive_does_not_stop():
    """3 -> 2 -> 1 is the revive in progress; the song must ride through it."""
    w = make()
    assert feed(w, [[True, True], [True, False], [False, True], [True, True]]) == []
    # The second entry above is "one ally has one segment left", still downed.


def test_stops_once_an_ally_has_no_gauge_at_all():
    w = make(hold=3)
    assert feed(w, [[True, True]] * 2 + [[True, False]] * 3) == [4]


def test_a_single_frame_of_an_ally_standing_is_not_enough():
    w = make(hold=3)
    assert feed(w, [[True, False], [True, True], [True, False], [True, False]]) == []


def test_the_hold_resets_when_everybody_goes_back_down():
    w = make(hold=3)
    assert feed(w, [[True, False], [True, False], [True, True],
                    [True, False], [True, False], [True, False]]) == [5]


def test_disarms_after_stopping_once():
    w = make(hold=1)
    assert w.update([True, False]) is True
    assert w.watching is False
    assert w.update([True, False]) is False


def test_ignores_updates_before_start():
    assert ReviveWatcher(hold_frames=1).update([True, False]) is False


def test_restarting_arms_it_again():
    w = make(hold=1)
    assert w.update([True, False]) is True
    w.start()
    assert w.update([True, True]) is False
    assert w.update([False, True]) is True


def test_no_slots_never_stops_the_music():
    """An empty reading list means we have nothing to judge, not that everyone is up."""
    w = make(hold=1)
    assert w.update([]) is False


def test_either_ally_standing_is_enough_to_stop():
    w = make(hold=1)
    assert w.update([False, True]) is True
    w.start()
    assert w.update([True, False]) is True


def test_rejects_a_nonsense_hold():
    with pytest.raises(ValueError):
        ReviveWatcher(hold_frames=0)


def test_a_frame_that_proves_nothing_holds_the_count_instead_of_resetting_it():
    """While the camera moves, a revived ally's health bar flickers frame to frame.

    Those frames say neither "downed" nor "standing", and treating them as downed
    put the stop 6.5 seconds late on real footage. None means "cannot tell": the
    count waits.
    """
    w = make(hold=4)
    assert feed(w, [[False], [None], [False], [None], [False], [None], [False]]) == [6]


def test_a_gauge_that_comes_back_still_clears_the_count():
    """The difference between a flicker and a revive: a read gauge is proof they are
    down again, and the count starts over."""
    w = make(hold=4)
    assert feed(w, [[False], [False], [True], [False], [False], [False]]) == []
