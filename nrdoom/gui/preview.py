"""Widgets that draw the gauge state, so what the app reads is visible at a glance."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..detector import rows_rect

LIT = QColor(112, 92, 226)
DIM = QColor(58, 58, 70)
OK_EDGE = QColor(90, 220, 130)
GUESS = QColor(255, 190, 70)
ZONE = QColor(110, 130, 150)


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Wrap a captured BGR or BGRA frame as a QImage.

    The QImage constructor from a raw pointer does not take ownership, and
    detach() does not fix that, so the buffer has to be copied out or it is
    freed under Qt as soon as the caller's array goes away.

    BGRA maps straight onto Format_RGB32: Qt defines that format as the byte
    order B, G, R, pad on a little-endian machine, which is exactly the capture
    buffer, so no channel shuffle is needed.
    """
    buf = np.ascontiguousarray(frame)
    h, w = buf.shape[:2]
    fmt = (QImage.Format.Format_RGB32 if buf.shape[2] == 4
           else QImage.Format.Format_BGR888)
    return QImage(buf.data, w, h, buf.strides[0], fmt).copy()


def _as_tuple(slot) -> tuple[float, float, float]:
    """Accept either a settings.Slot or a bare (cx, cy, radius) triple."""
    if isinstance(slot, tuple):
        return (float(slot[0]), float(slot[1]), float(slot[2]))
    return (float(slot.cx), float(slot.cy), float(slot.radius))


class GaugeBadge(QWidget):
    """A miniature of the in-game three-segment gauge, showing what we read."""

    def __init__(self, bars: int = 0, confidence: float = 0.0, side: int = 34, parent=None):
        super().__init__(parent)
        self.setFixedSize(side, side)
        self._bars = bars
        self._conf = confidence

    def set_state(self, bars: int | None, confidence: float = 0.0):
        self._bars = 0 if bars is None else bars
        self._conf = confidence
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(3, 3, -3, -3)
        pen = QPen(DIM, 4)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Three 100 degree segments with 20 degree gaps, matching the sprite.
        for start in (100, 220, 340):
            p.drawArc(r, (360 - start - 100) * 16, 100 * 16)
        pen = QPen(LIT, 4)
        p.setPen(pen)
        for i in range(self._bars):
            start = (100, 220, 340)[i % 3]
            p.drawArc(r, (360 - start - 100) * 16, 100 * 16)
        p.end()


class PreviewWidget(QWidget):
    """The captured frame with the ally panel drawn on top: the spots we read,
    the rows we counted, and what each gauge reported. Read-only, because the
    positions are no longer anybody's to mark.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._scale = 1.0
        self._origin = QPointF(0, 0)
        self.slots: list[tuple[float, float, float]] = []
        self.readings: list = []
        # Where these positions came from and how much the tracker trusts the box
        # they hang off, so the overlay can say so instead of looking confident.
        self.hud_state = "NONE"
        self.geometry_state = "verified"
        self.geometry_source = "seed"
        self.setMinimumSize(480, 270)

    # -- content ------------------------------------------------------------
    def set_frame(self, frame: np.ndarray):
        self._image = bgr_to_qimage(frame)
        self.update()

    def set_snapshot(self, snap):
        """The picture, the positions and the counts of one frame, in one repaint.

        Deliberately one method rather than three setters the engine would have to
        keep in step: apart, a new frame could be shown under circles from whenever
        the calibration last moved, which is the bug this replaced.
        """
        self._image = bgr_to_qimage(snap.frame)
        self.slots = [_as_tuple(s) for s in snap.spots]
        self.readings = list(snap.readings)
        self.hud_state = snap.hud_state
        self.geometry_state = snap.geometry_state
        self.geometry_source = snap.geometry_source
        self.update()

    # -- geometry ------------------------------------------------------------
    def _target_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        iw, ih = self._image.width(), self._image.height()
        aw, ah = self.width(), self.height()
        self._scale = min(aw / iw, ah / ih)
        w, h = iw * self._scale, ih * self._scale
        self._origin = QPointF((aw - w) / 2.0, (ah - h) / 2.0)
        return QRectF(self._origin, QPointF(w, h))

    def to_frame(self, pos: QPointF) -> tuple[float, float]:
        return ((pos.x() - self._origin.x()) / self._scale,
                (pos.y() - self._origin.y()) / self._scale)

    def to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(self._origin.x() + x * self._scale, self._origin.y() + y * self._scale)

    # -- painting ---
    # -- painting -------------------------------------------------------------
    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 24))
        rect = self._target_rect()
        if self._image is None:
            p.setPen(QColor(150, 150, 160))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                       "还没有画面\n选一个游戏窗口点「连接」")
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(rect, self._image)
            self._paint_overlays(p)
        p.end()

    def _paint_zone(self, p: QPainter):
        """Outline the only region the detector reads any more."""
        if self._image is None or not self.slots:
            return
        w, h = self._image.width(), self._image.height()
        x0, y0, x1, y1 = rows_rect((h, w), self.slots)
        rect = QRectF(self.to_widget(x0, y0), self.to_widget(x1, y1))
        p.save()
        p.setPen(QPen(ZONE, 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)
        p.setPen(QPen(ZONE))
        p.drawText(rect.adjusted(5, 2, -2, 0), Qt.AlignmentFlag.AlignTop
                   | Qt.AlignmentFlag.AlignLeft, "只检测这一块")
        p.restore()

    def _trust_pen(self):
        """How far the circles being drawn may be believed, as a pen and a word.

        The colour answers "are these positions the ones the tool is sure about",
        which is a different question from "did a gauge read here" - and the two are
        only the same when the geometry has actually been verified. Text goes with
        the colour because a colour alone is not something to hand a user.
        """
        if self.hud_state != "LOCKED":
            return QPen(ZONE, 2, Qt.PenStyle.DashLine), "画面框不确定"
        if self.geometry_state == "pending":
            return QPen(GUESS, 2, Qt.PenStyle.DashLine), "正在重新定位"
        if self.geometry_state == "fallback":
            return QPen(GUESS, 2, Qt.PenStyle.DashLine), "退回上一个可靠位置"
        return QPen(OK_EDGE, 2), ""

    def _paint_overlays(self, p: QPainter):
        self._paint_zone(p)
        pen, note = self._trust_pen()
        for i, (cx, cy, radius) in enumerate(self.slots):
            reading = self.readings[i] if i < len(self.readings) else None
            bars = None if reading is None else reading.bars
            r = radius * self._scale
            centre = self.to_widget(cx, cy)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(centre, r, r)
            p.drawEllipse(centre, r * 0.45, r * 0.45)
            p.setPen(QPen(pen.color()))
            p.drawText(QRectF(centre.x() + r + 4, centre.y() - 10, 60, 20),
                       Qt.AlignmentFlag.AlignLeft, f"#{i + 1}  {'-' if bars is None else bars}")
        if note and self.slots:
            centre = self.to_widget(self.slots[0][0], self.slots[0][1])
            p.setPen(QPen(pen.color()))
            p.drawText(QRectF(centre.x() + 40, centre.y() - 34, 260, 20),
                       Qt.AlignmentFlag.AlignLeft, note)
