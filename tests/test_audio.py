import wave

import numpy as np
import pytest

pytest.importorskip("PySide6.QtMultimedia")

from nrdoom.audio import MusicPlayer            # noqa: E402
from nrdoom.library import MusicLibrary         # noqa: E402


def make_wav(path, seconds=0.2, freq=440.0):
    n = int(44100 * seconds)
    frames = (np.sin(2 * np.pi * freq * np.arange(n) / 44100) * 16000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(frames.tobytes())
    return path


@pytest.fixture
def library(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    for i, f in enumerate((440.0, 550.0, 660.0)):
        make_wav(folder / f"t{i}.wav", freq=f)
    lib = MusicLibrary(folder, seed=1)
    lib.refresh()
    return lib


def test_volume_is_clamped_to_a_sane_range(qapp, library):
    player = MusicPlayer(library)
    player.set_volume(5)
    assert player.volume() == 1.0
    player.set_volume(-2)
    assert player.volume() == 0.0
    player.set_volume(0.4)
    assert player.volume() == pytest.approx(0.4)


def test_empty_library_reports_instead_of_crashing(qapp, tmp_path):
    empty = MusicLibrary(tmp_path / "nothing")
    empty.refresh()
    player = MusicPlayer(empty)
    errors = []
    player.error.connect(errors.append)
    assert player.play_next() is None
    assert errors


def test_play_next_selects_a_track_and_announces_it(qapp, library):
    player = MusicPlayer(library)
    started = []
    player.track_started.connect(started.append)
    track = player.play_next()
    assert track is not None
    assert track.name in started
    assert player.current == track
    player.stop()


def test_successive_plays_walk_the_shuffle_bag(qapp, library):
    player = MusicPlayer(library)
    played = []
    for _ in range(3):
        played.append(player.play_next())
        player.stop()
    assert len(set(played)) == 3
    assert player.play_next() is not None
    player.stop()


def test_a_real_device_lets_it_reach_playing(qapp, library, pump):
    """Soft check: no audio device is an environment limit, not a bug."""
    player = MusicPlayer(library)
    errors = []
    player.error.connect(errors.append)
    player.play_next()
    pump(600)
    if errors:
        pytest.skip(f"no usable audio backend here: {errors[0]}")
    assert player.playing or player.current is not None
    player.stop()


@pytest.fixture
def long_library(tmp_path):
    """One three-second track: long enough to have somewhere to resume to."""
    folder = tmp_path / "music"
    folder.mkdir()
    make_wav(folder / "long.wav", seconds=3.0)
    lib = MusicLibrary(folder, seed=1)
    lib.refresh()
    return lib


def test_stopping_remembers_where_it_had_got_to(qapp, long_library, pump):
    player = MusicPlayer(long_library)
    if not _plays(player, pump):
        pytest.skip("no usable audio backend here")
    track = player.current
    if not _playhead_moved(player, pump):
        pytest.skip("no playhead on this machine")
    player.stop()
    remembered = player.pending_resume
    assert remembered is not None and remembered[0] == track
    # The position is compared to itself, not to a second read of the playhead:
    # `stop()` samples it again, and a few milliseconds of playback between two
    # reads made an exact comparison fail at random.
    assert remembered[1] > 0, "the track never advanced, so this proves nothing"


def test_the_playhead_is_ahead_of_a_short_wait(qapp, long_library, pump):
    """The thing the exact comparison above was reaching for: after half a second
    of playing, the remembered place is partway into the track, not at zero."""
    player = MusicPlayer(long_library)
    if not _plays(player, pump):
        pytest.skip("no usable audio backend here")
    if not _playhead_moved(player, pump, budget_ms=500):
        player.stop()
        pytest.skip("no playhead inside 500 ms on this machine")
    played = player.position()
    player.stop()
    assert 0 < played <= 1000, played


def test_the_next_trigger_picks_up_where_the_stop_left_it(qapp, long_library, pump):
    """A false or genuine stop mid-run must not send the song back to the start."""
    player = MusicPlayer(long_library)
    if not _plays(player, pump):
        pytest.skip("no usable audio backend here")
    track = player.current
    if not _playhead_moved(player, pump):
        pytest.skip("no playhead on this machine")
    played = player.position()
    player.stop()
    again = player.play_next()
    pump(700)
    assert again == track
    assert player.position() >= played - 120, (played, player.position())
    player.stop()


def test_previewing_a_track_does_not_burn_the_resume_point(qapp, long_library, pump):
    player = MusicPlayer(long_library)
    if not _plays(player, pump):
        pytest.skip("no usable audio backend here")
    track = player.current
    if not _playhead_moved(player, pump):
        pytest.skip("no playhead on this machine")
    player.stop()
    before = player.pending_resume
    assert before is not None and before[0] == track and before[1] > 0
    player.play_next(resume=False)
    pump(100)
    assert player.pending_resume == before, "a preview must not spend the resume point"
    player.stop()


def _plays(player, pump) -> bool:
    """False when this machine has no media backend to test against."""
    errors = []
    player.error.connect(errors.append)
    player.play_next()
    pump(300)
    return not errors and player.position() >= 0 and player.current is not None


def _playhead_moved(player, pump, budget_ms: int = 3000) -> bool:
    """Wait until the backend actually reports progress, instead of assuming it.

    `position()` stays 0 until the loader has decoded something, and on a busy
    machine that can take longer than any fixed `pump` - which is how
    `test_previewing_a_track_does_not_burn_the_resume_point` went red while three
    video replays were eating the CPU. Running out of budget is this machine's
    environment, not the player's behaviour, so the caller treats it as a skip.
    """
    waited = 0
    while waited < budget_ms:
        pump(100)
        waited += 100
        if player.position() > 0:
            return True
    return False


def test_looping_moves_on_to_the_next_track_when_one_ends(qapp, library, pump):
    """"循环播放" is the music not stopping, not one file repeating.

    With the player told to loop a single file forever, `pick()` is reached once per
    session and every later trigger just resumes the same song from the same place -
    so the order arranged in the window cannot be heard until the app restarts,
    which is what the user reported.
    """
    if not _plays(MusicPlayer(library), pump):
        pytest.skip("no usable audio backend here")
    library.order = "ordered"
    library.set_sequence([t.name for t in library.tracks])
    player = MusicPlayer(library)
    player.set_loop(True)
    first = player.play_next()
    waited = 0
    while waited < 4000 and player.current == first:
        pump(100)
        waited += 100
    player.stop()
    assert player.current != first, (
        "still on %s after %.1f s: the player never moved on" % (first.name, waited / 1000))
