"""宁姆韦德的雨 — plays your own track when every ally is downed on their last segment."""
import faulthandler
import sys
import time
import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from nrdoom import __version__
from nrdoom.assets import APP_NAME, APP_NAME_ASCII, app_dir, ensure_default_music
from nrdoom.gui import MainWindow
from nrdoom.gui.theme import apply as apply_theme
from nrdoom.selfcheck import run as run_selfcheck
from nrdoom.settings import Settings

CRASH_LOG = "崩溃.log"
_log_handle = None


def _open_crash_log():
    """One append-mode file next to the exe that everything unusual is written to.

    A windowed build has no console, so before this an exception inside a Qt slot
    was printed to a stderr nobody could see while the app carried on half-broken -
    which is exactly what "好像崩了一下" looks like from inside the game, and left
    nothing behind to debug.
    """
    global _log_handle
    if _log_handle is None:
        try:
            _log_handle = open(app_dir() / CRASH_LOG, "a", buffering=1, encoding="utf-8")
        except OSError:
            return None
    return _log_handle


def _log_crash(exc: BaseException, where: str = "启动") -> None:
    handle = _open_crash_log()
    if handle is None:
        return
    try:
        handle.write("\n==== %s %s ====\n" % (where, time.strftime("%Y-%m-%d %H:%M:%S")))
        handle.write("".join(traceback.format_exception(exc)))
    except OSError:
        pass


def _excepthook(kind, value, tb) -> None:
    """What PySide6 calls when a slot raises; the default only prints to stderr."""
    _log_crash(value or kind, "运行期")
    sys.__excepthook__(kind, value, tb)


def _crash_probe() -> None:
    """Raise from a timer slot on purpose, so `崩溃.log` can be proven to work.

    A diagnostic nobody has seen fire is not a diagnostic: this goes through the
    same path a real slot failure does (PySide hands it to `sys.excepthook`) while
    the app stays open, which is exactly the case the windowed build used to lose.
    """
    def boom():
        raise RuntimeError("crash probe: 这条是故意抛的，用来确认崩溃.log 写得下来")

    QTimer.singleShot(1500, boom)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME_ASCII)
    app.setOrganizationName(APP_NAME_ASCII)
    app.setApplicationDisplayName(APP_NAME)
    apply_theme(app)
    sys.excepthook = _excepthook
    handle = _open_crash_log()
    if handle is not None:
        # A segfault never reaches the hook above; this writes the C-level
        # traceback to the same file on the way down.
        faulthandler.enable(file=handle)

    window = MainWindow(Settings.load())
    # Destroy the widget on the GUI thread when it closes. If the Python wrapper
    # outlives the app and reaches the garbage collector first, Qt segfaults.
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.show()
    if "--crash-probe" in sys.argv:
        _crash_probe()
    return app.exec()


if __name__ == "__main__":
    if "--version" in sys.argv:
        # A bug report that cannot say which build it came from is worth nothing.
        print(f"{APP_NAME} {__version__}")
        sys.exit(0)
    if "--selftest" in sys.argv:
        sys.exit(run_selfcheck())
    try:
        ensure_default_music()
        rc = main()
    except BaseException as exc:                       # noqa: BLE001 - last resort
        _log_crash(exc)
        raise
    sys.exit(rc)
