"""Stand-ins shared by the test modules."""
import pathlib

import numpy as np


class FakeSource:
    """A FrameSource stand-in that serves a fixed image instead of a screen."""

    backend = "fake"

    def __init__(self, img=None):
        self.img = img
        self.frames = 0
        self.running = img is not None
        self.error = None
        self.target_fps = 20
        self._min_interval = 0.05
        self.stopped = 0

    def latest(self):
        return self.img

    def set_frame(self, img):
        self.img = img
        self.frames += 1

    def stop(self):
        self.stopped += 1
        self.running = False


class FakePlayer:
    def __init__(self, name="doom.mp3"):
        self.calls = []
        self.name = name
        self.stops = 0

    def play_next(self):
        self.calls.append(self.name)
        return None if self.name is None else pathlib.Path(self.name)

    def stop(self):
        self.stops += 1

    def set_volume(self, value):
        self.volume = value
