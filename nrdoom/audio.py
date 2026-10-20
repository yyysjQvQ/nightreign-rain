"""Playback.

Built on Qt's own media stack rather than a separate audio library: the app
already ships PySide6 for the GUI, and QMediaPlayer covers mp3/ogg/flac/wav/m4a
through the OS backend, which is a wider format net than anything I would add a
dependency for.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class MusicPlayer(QObject):
    track_started = Signal(str)
    playing_changed = Signal(bool)
    error = Signal(str)

    def __init__(self, library, parent=None):
        super().__init__(parent)
        self._library = library
        self._output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._output)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_status)
        self._player.errorOccurred.connect(self._on_error)
        self.current: Path | None = None
        self._loop = True
        self._resume_to: tuple[Path, int] | None = None
        self._seek_to = 0
        self.set_volume(0.8)

    def set_loop(self, on: bool) -> None:
        """Keep the music going until something else stops it.

        "Going on" means the next track from the library, not one file repeated:
        with `Loops.Infinite` the player never reached an end, so `pick()` ran once
        per session and every later trigger resumed the same song from the same
        place - the order arranged in the window could not be heard without a
        restart. A one-track library still repeats, because the next track is it.
        """
        self._loop = bool(on)
        self._player.setLoops(QMediaPlayer.Loops.Once)

    def looping(self) -> bool:
        return self._loop

    def set_volume(self, volume: float) -> None:
        volume = max(0.0, min(1.0, float(volume)))
        self._volume = volume
        self._output.setVolume(volume)

    def volume(self) -> float:
        return self._volume

    def play_next(self, resume: bool = True) -> Path | None:
        """Start the next shuffled track - or pick the interrupted one back up.

        A stop in the middle of a run, false or real, must not send the song back to
        the beginning, so where it got to is kept until something plays it. The
        preview button passes resume=False so it cannot spend that on itself.
        """
        if resume and self._resume_to is not None:
            track, position = self._resume_to
            self._resume_to = None
            if track.exists():
                self._start(track, position)
                return track
        track = self._library.pick()
        if track is None:
            self.error.emit("音乐文件夹里没有可播放的文件")
            return None
        self._start(track, 0)
        return track

    def _start(self, track: Path, position_ms: int) -> None:
        self.current = track
        self._seek_to = position_ms
        self._player.setSource(QUrl.fromLocalFile(str(track)))
        # setSource resets the loop count, so re-apply it before starting.
        self.set_loop(self._loop)
        self._player.play()
        self.track_started.emit(track.name)
        self._on_status(self._player.mediaStatus())

    def stop(self) -> None:
        """Stop, keeping the place so the next trigger can carry on from here."""
        if self.playing and self.current is not None:
            position = int(self._player.position())
            if position > 0:
                self._resume_to = (self.current, position)
        self._player.stop()

    def pause(self) -> None:
        self._player.pause()

    def resume(self) -> None:
        if self.current is not None:
            self._player.play()

    def position(self) -> int:
        """Milliseconds into the current track."""
        return int(self._player.position())

    @property
    def pending_resume(self):
        """Where the next trigger will pick up, if there is one to pick up."""
        return self._resume_to

    @property
    def playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def _on_status(self, status) -> None:
        if (status == QMediaPlayer.MediaStatus.EndOfMedia and self._loop
                and self.current is not None):
            # The end of a track is the queue's chance to move: `resume=False`
            # because a track that finished is not an interrupted one.
            self.play_next(resume=False)
            return
        # Seeking before the source has loaded is dropped by the backend, so the
        # resume position waits here until the media is actually ready.
        if self._seek_to and status in (QMediaPlayer.MediaStatus.LoadedMedia,
                                        QMediaPlayer.MediaStatus.BufferedMedia):
            position, self._seek_to = self._seek_to, 0
            self._player.setPosition(position)

    def _on_state(self, state) -> None:
        self.playing_changed.emit(state == QMediaPlayer.PlaybackState.PlayingState)

    def _on_error(self, error, message) -> None:
        if error != QMediaPlayer.Error.NoError:
            self.error.emit(message or "播放器出错")
            self.playing_changed.emit(False)
