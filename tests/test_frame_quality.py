"""The stalled-capture counter, which is a report and not a gate."""
import numpy as np
import pytest

from nrdoom.frame_quality import FrameValidator, SAMPLE_STEP


def frame(v):
    img = np.zeros((200, 320, 3), np.uint8)
    img[::7, ::11] = v                      # some texture so the sample is not flat
    return img


def test_repeats_are_counted_and_a_change_clears_them():
    v = FrameValidator(hold=3)
    assert v.check(frame(10)) is True
    for _ in range(3):
        assert v.check(frame(10)) is False
    assert v.frozen is True
    assert v.check(frame(11)) is True
    assert v.stale == 0 and v.frozen is False


def test_below_the_hold_it_is_not_called_frozen():
    """Two identical frames in a row is an ordinary still moment, not a stall.
    The first frame only sets the baseline, so four checks are three repeats."""
    v = FrameValidator(hold=5)
    for _ in range(4):
        v.check(frame(10))
    assert v.stale == 3 and v.frozen is False


def test_a_different_frame_size_is_not_a_repeat():
    v = FrameValidator(hold=1)
    v.check(frame(10))
    big = np.zeros((400, 640, 3), np.uint8)
    assert v.check(big) is True and v.stale == 0


def test_reset_forgets_the_picture():
    v = FrameValidator(hold=1)
    v.check(frame(10))
    v.check(frame(10))
    assert v.frozen
    v.reset()
    assert v.stale == 0 and v.check(frame(10)) is True


def test_it_rejects_an_unusable_hold():
    with pytest.raises(ValueError):
        FrameValidator(hold=0)


def test_only_the_sampled_grid_has_to_match():
    """The check is a sixteenth-pixel sample, so a difference that falls between the
    samples is not seen - which is the point, and worth knowing before anyone reads
    `frozen` as "the frames were bit-identical"."""
    v = FrameValidator(hold=1)
    a = frame(10)
    b = a.copy()
    b[3, 3] = 250                            # row 3, column 3: not on the grid
    v.check(a)
    assert v.check(b) is False, "the sample missed the only changed pixel"
