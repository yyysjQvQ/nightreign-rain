import pytest

from nrdoom.trigger import DoomTrigger

BOTH_3 = [3, 3]
ONE_3 = [3, 2]
NOT_BOTH = [3, None]


def make(**kw):
    return DoomTrigger(hold_frames=3, **kw)


def feed(trig, seq, start_time=0.0, step=0.05):
    """Push a sequence of readings; return the indices at which it fired."""
    fired = []
    t = start_time
    for i, reading in enumerate(seq):
        if trig.update(reading, t):
            fired.append(i)
        t += step
    return fired


def test_fires_once_the_hold_is_completed():
    trig = make()
    assert feed(trig, [BOTH_3, BOTH_3, BOTH_3]) == [2]


def test_does_not_fire_on_a_single_frame():
    trig = make()
    assert feed(trig, [BOTH_3, [3, 2], BOTH_3]) == []


def test_a_blip_resets_the_hold():
    """One bad frame in the middle must not count toward the hold.

    Without this the trigger would fire on a signal that flickers in and out,
    which is exactly what a partially revived ally looks like.
    """
    trig = make()
    assert feed(trig, [BOTH_3, BOTH_3, [3, 2], BOTH_3, BOTH_3]) == []


def test_needs_every_ally_at_the_threshold():
    trig = make()
    assert feed(trig, [NOT_BOTH] * 10) == []
    assert feed(trig, [[2, 2]] * 10) == []
    assert feed(trig, [[3, 2]] * 10) == []


def test_fires_again_only_after_the_condition_clears():
    trig = make()
    assert feed(trig, [BOTH_3] * 3 + [ONE_3, NOT_BOTH] + [BOTH_3] * 3) == [2, 7]


def test_does_not_re_fire_while_the_condition_holds():
    trig = make()
    assert feed(trig, [BOTH_3] * 40) == [2]


def test_cooldown_blocks_a_second_fire():
    trig = make(cooldown_s=5.0)
    fired = []
    t = 0.0
    for reading in [BOTH_3] * 3 + [ONE_3] + [BOTH_3] * 5:
        if trig.update(reading, t):
            fired.append(t)
        t += 0.05
    assert fired == [0.10]


def test_cooldown_expiry_allows_the_next_fire():
    trig = make(cooldown_s=0.2)
    fired = []
    t = 0.0
    for reading in [BOTH_3] * 3 + [ONE_3] + [BOTH_3] * 8:
        if trig.update(reading, t):
            fired.append(round(t, 2))
        t += 0.05
    assert len(fired) == 2
    assert fired[1] - fired[0] >= 0.2


def test_reset_arms_the_trigger_again():
    trig = make()
    feed(trig, [BOTH_3] * 3)
    trig.reset()
    assert feed(trig, [BOTH_3, BOTH_3, BOTH_3]) == [2]


def test_allies_count_is_configurable():
    trig = DoomTrigger(required_bars=3, hold_frames=1, allies=1)
    assert trig.update([3, 1], 0.0) is True


def test_threshold_is_configurable():
    trig = DoomTrigger(required_bars=2, hold_frames=1)
    assert trig.update([2, 2], 0.0) is True
    trig = DoomTrigger(required_bars=2, hold_frames=1)
    assert trig.update([1, 2], 0.0) is False


def test_empty_readings_never_fire():
    trig = make()
    assert feed(trig, [[], [], []]) == []


def test_progress_reports_the_run_length():
    trig = make()
    trig.update(BOTH_3, 0.0)
    trig.update(BOTH_3, 0.05)
    assert trig.held == 2
    assert trig.armed is True
    trig.update(BOTH_3, 0.1)
    assert trig.armed is False
    assert trig.fires == 1


@pytest.mark.parametrize("hold", [1, 2, 5, 12])
def test_hold_length_is_honoured(hold):
    trig = DoomTrigger(required_bars=3, hold_frames=hold, allies=2)
    assert feed(trig, [BOTH_3] * (hold - 1)) == []
    trig = DoomTrigger(required_bars=3, hold_frames=hold, allies=2)
    assert feed(trig, [BOTH_3] * hold) == [hold - 1]
