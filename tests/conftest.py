import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pytest
from PySide6.QtWidgets import QApplication

from fakes import FakePlayer, FakeSource


@pytest.fixture
def fake_source():
    return FakeSource


@pytest.fixture
def fake_player():
    return FakePlayer


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def pump(qapp):
    """Let Qt deliver queued signals for a moment."""

    def run(ms=200):
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.005)
    return run
