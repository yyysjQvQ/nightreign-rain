"""The evidence a position change needs before it is kept (报告 P0-3 / G07-G10).

`GeometryManager` is only about proof, so these run without a frame in sight: what
matters is which captures voted, how far apart they were, and whether each vote
actually read better than where the tool already was.
"""
import pytest

from nrdoom.geometry import GeometryManager

VERIFIED = [(128.0, 247.0, 27.0), (128.0, 358.0, 27.0)]
MOVED = [(128.0, 251.0, 27.0), (128.0, 362.0, 27.0)]     # 4 px down, rigid
WONKY = [(128.0, 251.0, 27.0), (128.0, 380.0, 27.0)]     # only the second row moved


def fresh(spots=MOVED, source="recovery"):
    m = GeometryManager()
    m.settle(VERIFIED, "cache", 0.0)
    return m


def vote(m, spots, frame_id, t, score=2.0, base=0.0):
    return m.propose(spots, "recovery", frame_id, t, score, base)


def test_a_proposal_is_used_at_once_but_believed_by_nobody():
    """G07: one frame's confident read must become where we look, and must not
    become what we keep. Recovery speed and cache safety are different questions."""
    m = fresh()
    used = vote(m, MOVED, frame_id=1, t=0.0)
    assert used == MOVED, "the tool stopped looking at the place it just found"
    assert m.verified == VERIFIED
    assert not m.earned(t=0.0)


def test_separate_captures_are_what_earn_it():
    """G08: three different capture frames, spread over at least the minimum age."""
    m = fresh()
    for i in range(3):
        vote(m, MOVED, frame_id=i, t=i * 0.05)     # 3 votes, 0.1 s of watching
    assert m.earned(t=0.1) is False, "below min_age_s the votes are not enough yet"
    vote(m, MOVED, frame_id=3, t=0.25)
    assert m.earned(t=0.25) is True
    assert m.accept(t=0.25) == MOVED
    assert m.verified == MOVED
    assert m.candidate is None


def test_a_frozen_picture_casts_no_extra_votes():
    """G09: the frames have to be different captures. Ticking a stalled feed forty
    times is one observation repeated, and it used to be enough to write a
    calibration."""
    m = fresh()
    for _ in range(30):
        vote(m, MOVED, frame_id=7, t=0.0)
    assert m.candidate.votes == 1
    assert m.earned(t=9.0) is False


def test_a_vote_that_did_not_read_better_is_not_a_vote():
    """A candidate is only ahead if the captures that saw it say so; one frame where
    the old position read just as well is a frame that did not confirm the move."""
    m = fresh()
    for i in range(4):
        vote(m, MOVED, frame_id=i, t=i * 0.1, score=1.0, base=1.0)
    assert m.earned(t=0.3) is False


def test_the_frames_after_the_move_are_what_confirm_it():
    """The hole this closes: once the spots are where the panel is, nothing proposes
    any more - so a move that is *right* would never reach three votes unless the
    later frames could vote too."""
    m = fresh()
    vote(m, MOVED, frame_id=1, t=0.0)
    assert m.earned(t=0.5) is False, "one capture's opinion is not agreement"
    for i in (2, 3):
        m.confirm(frame_id=i, t=i * 0.15, score=2.0, base_score=0.0)
    assert m.earned(t=0.4) is True
    assert m.accept(t=0.4) == MOVED


def test_a_confirmation_that_did_not_read_better_does_not_count():
    m = fresh()
    vote(m, MOVED, frame_id=1, t=0.0)
    m.confirm(frame_id=2, t=0.15, score=0.0, base_score=1.0)
    m.confirm(frame_id=3, t=0.3, score=0.0, base_score=1.0)
    assert m.candidate.votes == 3
    assert m.earned(t=0.4) is False


def test_a_move_that_only_carries_one_row_is_refused():
    """The ally list is one rigid sprite: a proposal that shifts a single row is two
    unrelated errors that happened to agree with nothing."""
    m = fresh()
    for i in range(4):
        vote(m, WONKY, frame_id=i, t=i * 0.1)
    assert m.earned(t=0.3) is False, "the row pitch bent by %s px" % m.candidate.pitch_off


def test_a_candidate_that_moves_is_a_new_candidate():
    """Collecting evidence across two different places would be evidence for neither."""
    m = fresh()
    vote(m, MOVED, frame_id=1, t=0.0)
    elsewhere = [(128.0, 270.0, 27.0), (128.0, 381.0, 27.0)]
    vote(m, elsewhere, frame_id=2, t=0.1)
    assert m.candidate.spots == tuple(tuple(s) for s in elsewhere)
    assert m.candidate.votes == 1


def test_losing_the_read_reverts_to_the_last_kept_layout():
    """G10: after the move is accepted, evidence disappearing goes back to the
    position that had it - the kept layout is not a promise to defend it blind."""
    m = fresh()
    for i in range(4):
        vote(m, MOVED, frame_id=i, t=i * 0.1)
    m.accept(t=0.3)
    assert m.verified == MOVED
    for i in range(m.rollback_frames - 1):
        assert m.report_read(t=0.4 + i * 0.05, score=0.0) is None
    assert m.report_read(t=0.55, score=0.0) == VERIFIED
    assert m.verified == VERIFIED, "the fallback never came back"
    assert m.active == VERIFIED
    assert m.source == "rollback"

    # A blind stretch interrupted by a read is not a blind stretch.
    m2 = fresh()
    for i in range(4):
        vote(m2, MOVED, frame_id=i, t=i * 0.1)
    m2.accept(t=0.3)
    m2.report_read(0.4, 0.0)
    m2.report_read(0.45, 0.0)
    m2.report_read(0.5, 1.0)
    m2.report_read(0.55, 0.0)
    m2.report_read(0.6, 0.0)
    assert m2.verified == MOVED


def test_refusing_a_candidate_leaves_the_kept_layout_reading():
    m = fresh()
    vote(m, MOVED, frame_id=1, t=0.0)
    m.refuse(t=0.05)
    assert m.candidate is None
    assert m.active == VERIFIED


def test_settling_a_new_box_clears_the_evidence():
    """A different picture box is a different layout: a candidate collected against
    the old one must not be inherited by the new one."""
    m = fresh()
    vote(m, MOVED, frame_id=1, t=0.0)
    m.settle([(160.0, 300.0, 30.0)], "seed", 0.05)
    assert m.candidate is None
    assert m.verified == [(160.0, 300.0, 30.0)]
    assert m.fallback is None


def test_the_kinds_of_evidence_are_named_for_the_log():
    m = fresh()
    vote(m, MOVED, frame_id=1, t=0.0)
    for i in (2, 3, 4):
        vote(m, MOVED, frame_id=i, t=i * 0.1)
    m.accept(t=0.4)
    assert [k for k, _t, _s in m.events] == ["settled", "proposed", "verified"]
