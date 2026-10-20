"""Read a test capture, and say why it is not there.

The real game screenshots under `fixtures/` are not in this repository: they are
43 MB of FromSoftware artwork, and one of them is a friend's character name in a HUD
crop. The synthetic cases are generated from the one capture that does ship, so
running `python fixtures/make_synthetic.py` brings most of the suite back. What is
left - the dusk/sunset/purple-sky/red-backdrop frames the thresholds were measured
against - can only be supplied by someone who has their own captures, so a test
that wants one reports skipped rather than failing on a `cv2.imread` that returned
None, which would read like a broken detector.
"""
from pathlib import Path

import cv2
import pytest

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def shot(name: str):
    path = FIX / name
    if not path.is_file():
        pytest.skip("%s is not in this checkout (see fixtures/README.md)" % name)
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip("%s could not be decoded by OpenCV on this machine" % name)
    return img
