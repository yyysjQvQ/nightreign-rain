"""The small always-on-top readout you keep visible while playing."""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import QHBoxLayout, QMenu, QLabel, QVBoxLayout, QWidget

from .preview import GaugeBadge


class PetWidget(QWidget):
    """Two mini gauges plus a state line, frameless and above the game.

    It is a debugging aid as much as a gadget: if the app misreads a gauge you
    can see it disagree with the game in the same instant.
    """

    quitRequested = Signal()
    hideRequested = Signal()

    def __init__(self, slots: int = 2, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # This window gets closed and rebuilt whenever the calibration changes.
        # Letting its Python wrapper fall through to the garbage collector segfaults
        # the process, so hand the C++ object back to Qt on close instead.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._drag: QPoint | None = None
        self.setStyleSheet("background: transparent;")

        self.badges: list[GaugeBadge] = []
        self.labels: list[QLabel] = []
        row = QHBoxLayout()
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)
        for i in range(max(1, slots)):
            badge = GaugeBadge(0, side=26)
            label = QLabel("-")
            label.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
            label.setStyleSheet("color: #e8e8f0;")
            row.addWidget(badge)
            row.addWidget(label)
            self.badges.append(badge)
            self.labels.append(label)
        self.state = QLabel("")
        self.state.setStyleSheet("color: #ffd166; font-weight: bold;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(row)
        outer.addWidget(self.state)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def set_readings(self, readings):
        for i, badge in enumerate(self.badges):
            got = readings[i] if i < len(readings) else None
            bars = None if got is None else got.bars
            badge.set_state(bars, 0.0 if got is None else got.confidence)
            self.labels[i].setText("-" if bars is None else str(bars))

    def flash(self, text: str):
        self.state.setText(text)

    def clear_flash(self):
        self.state.setText("")

    # -- window behaviour -----------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def _menu(self, _pos):
        menu = QMenu(self)
        hide = QAction("隐藏悬浮窗", self)
        # Ask the main window rather than hiding ourselves: a plain `hide()` leaves
        # the widget alive, and the next calibration change rebuilds the pet and
        # quietly puts it back on screen.
        hide.triggered.connect(self.hideRequested)
        quit_ = QAction("退出程序", self)
        quit_.triggered.connect(self.quitRequested)
        menu.addAction(hide)
        menu.addAction(quit_)
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
