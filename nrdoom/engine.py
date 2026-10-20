"""The loop that ties capture, detection, triggering and playback together.

Runs on the Qt GUI thread off a QTimer. The capture backend already has its own
thread, so this only ever grabs the newest frame and reads a couple of fixed
spots in it; nothing here blocks.

Preview and listening are separate states on purpose: you need to see the game
to calibrate the gauges, and you should not be triggering music while doing it.
"""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .capture import FrameSource
from .detector import (ADOPT_MIN_CONFIDENCE, REALIGN_OFF_FRAC_W,
                       REALIGN_SETTLE_FRAC_W, ROW_MIN_STRENGTH, Ring, ally_bar,
                       count_rows, detect_spots,
                       detect_with_recovery, gauge_spots, hud_rect, propose_rings,
                       ring_presence, row_peaks, row_ys)
from .event_arbiter import EventArbiter
from .frame_quality import FrameValidator
from .hud_tracker import LOST, HudTracker
from .party import PartyCounter
from .revive import ReviveWatcher
from .settings import Slot
from .slot_observation import SlotObservation
from .trigger import DoomTrigger

# Detection runs every frame; the preview only needs to look smooth. Converting a
# 2560x1600 frame to a QImage costs about 6 ms, and doing it 20 times a second to
# move a picture nobody can read is the single largest thing in the loop.
PREVIEW_FPS_CAP = 12.0

# How far the measured picture box has to move before the geometry is re-seeded
# from it. The letterbox is steady to a pixel or two on real frames; a fade or a
# menu is off by hundreds.
RECT_RESEED_PX = 8


class Engine(QObject):
    # Following the portraits is free; looking at the gauges to nail the last few
    # pixels costs about a second, so it is tried a few times per move.
    REALIGN_TRIES = 3
    REALIGN_GAP_FRAMES = 100

    frame_ready = Signal(object)          # newest captured frame (BGR ndarray)
    readings_changed = Signal(list)       # [Ring | None, ...] one per slot
    party_changed = Signal(int)           # allies still in the run, from the row count
    calibrationChanged = Signal()         # positions were re-derived after drift
    fired = Signal(str)                   # track name that just started
    stopped = Signal(int)                 # an ally stood up: which slot ended it
    error = Signal(str)

    def __init__(self, settings, source: FrameSource | None = None, player=None,
                 clock=None, diag=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.source = source or FrameSource(target_fps=settings.target_fps)
        self.player = player
        # Replays drive this from the frame's own capture stamp, so a duration
        # written in seconds means the same thing offline as it does live.
        self._clock = clock or time.monotonic
        self.diag = diag
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._last_frame_id = -1
        self._last_preview = 0.0
        self._frame: np.ndarray | None = None
        self._rect = None
        # One second of frames is the vote on where the picture box is; a fade or
        # an overlay in a bar cannot move the geometry on its own.
        self.hud = HudTracker(window=max(5, int(settings.target_fps)),
                              relocate_frames=self._relocate_frames())
        # Diagnostic only: a stalled capture is reported, never used to
        # accept or refuse evidence. See nrdoom/frame_quality.py.
        self.validator = FrameValidator()
        self._rect_top = -1 << 30
        self._readings: list[Ring | None] = []
        self._evidence: list[SlotObservation] = []
        self._party = 0
        self.listening = False
        self.paused = False
        self.trigger = self._build_trigger()
        self.watcher = ReviveWatcher(settings.revive_hold_frames)
        # The v2 stop path: one tracker per locked ally, running on the capture
        # clock. `watcher` stays the single "a song is going" latch for both
        # policies, so switching only changes who decides the stop.
        self.arbiter = EventArbiter(self._tracker_args())
        self._frame_fresh = True
        self._reseeded = False
        self._pose_state = "ACQUIRING"
        # Whether the positions we are holding have ever actually found the panel.
        # A seed that has not must not be defended by the vote - see `_tick`.
        self._geometry_proven = False
        self.party = PartyCounter(settings.party_hold_frames, *self._rise_args())
        self._spot_key = None
        self._tune_tries = 0
        self._tune_wait = 0

    # -- wiring -------------------------------------------------------------
    def _tracker_args(self) -> dict:
        s = self.settings
        return {"window_s": s.revive_window_s,
                "min_candidate_s": s.revive_min_evidence_s,
                "min_samples": s.revive_min_samples,
                "min_alive_share": s.revive_min_alive_share,
                "lapse_s": s.revive_lapse_s,
                "conflict_s": s.revive_conflict_s}

    def _relocate_frames(self) -> int:
        """How long a new picture box has to hold before we give up ours."""
        return max(0, int(round(max(0.0, self.settings.box_hold_s)
                                * self.settings.target_fps)))

    def _rise_args(self) -> tuple[int, int]:
        """How much proof a bigger party needs, in samples and in frames."""
        s = self.settings
        if s.party_rise_s <= 0:
            return 1, 1           # zero seconds is the A/B switch: no gate at all
        samples = max(1, int(s.party_rise_samples))
        frames = max(samples, int(round(s.party_rise_s * s.target_fps)))
        return samples, frames

    def _build_trigger(self) -> DoomTrigger:
        s = self.settings
        return DoomTrigger(required_bars=s.required_bars, hold_frames=s.hold_frames,
                           allies=max(1, len(s.slots) or 2), cooldown_s=s.cooldown_s)

    def reconfigure(self):
        """Call after mutating settings."""
        self.hud.window = max(5, int(self.settings.target_fps))
        self.hud.relocate_frames = self._relocate_frames()
        self.source.target_fps = self.settings.target_fps
        self.source._min_interval = 1.0 / self.source.target_fps
        self.trigger = self._build_trigger()
        self.watcher.hold_frames = max(1, self.settings.revive_hold_frames)
        self.arbiter.tracker_args = self._tracker_args()
        self.party.hold_frames = max(1, self.settings.party_hold_frames)
        self.party.rise_samples, self.party.rise_frames = self._rise_args()
        if self._timer.isActive():
            self._timer.setInterval(max(10, int(1000 / self.settings.target_fps)))

    def connect_window(self, hwnd: int | None) -> str:
        self._last_frame_id = -1
        if hwnd:
            backend = self.source.start_window(hwnd)
        else:
            backend = self.source.start_screen()
        if not self.source.running:
            self.error.emit(self.source.error or "抓屏启动失败")
            return backend
        self._timer.start(max(10, int(1000 / self.settings.target_fps)))
        return backend

    def disconnect(self):
        self._timer.stop()
        self.listening = False
        self.source.stop()
        self._frame = None
        self._rect = None
        self.hud.reset()
        self.validator.reset()

    def set_listening(self, on: bool) -> bool:
        """Arm or disarm detection. All it needs is a picture to look at."""
        if on:
            if not self.source.running:
                self.error.emit("还没有连接游戏窗口")
                return False
            self.trigger.reset()
            self.paused = False
        self.listening = bool(on)
        if not on:
            self.watcher.stop()
            self.arbiter.end_song()
            self.party.reset()
            self.party_changed.emit(0)
            if self.player is not None:
                self.player.stop()
            self._readings = []
            self.readings_changed.emit([])
        return self.listening

    def set_paused(self, on: bool) -> bool:
        """Pause stops the running track and holds off auto-play until resumed."""
        self.paused = bool(on)
        self.watcher.stop()
        self.arbiter.end_song()
        if self.paused and self.player is not None:
            self.player.stop()
        self.trigger.reset()
        return self.paused

    def _spots_for(self, shape, rect):
        """The ally spots for this frame size: seeded, refined, then remembered.

        The panel is at a fixed place on screen, so nobody has to mark it. What
        the in-game UI size setting changes is *where* that fixed place lands, so
        a geometry that has actually read well is kept per resolution rather than
        re-derived every launch, and a window resize gets its own instead of
        inheriting a stale one.

        The letterbox counts as a resize: the geometry hangs off the picture box,
        so when the measured box moves the seed has to move with it. Without that
        the first frame decides everything - and a recording that opens on a fade
        measures its top bar at 312 px instead of 80 and seeds the whole session
        232 px below the panel.
        """
        key = "%dx%d" % (shape[1], shape[0])
        top = 0 if rect is None else rect[1]
        self._reseeded = not (key == self._spot_key
                              and abs(top - self._rect_top) <= RECT_RESEED_PX)
        if not self._reseeded:
            return self.settings.slot_tuples()
        self._spot_key = key
        self._rect_top = top
        self._geometry_proven = False
        self.party.reset()
        self._party = 0
        seed = gauge_spots(shape, rect=rect)
        saved = [(x.cx, x.cy, x.radius) for x in
                 (self.settings.spots_cache.get(key) or self.settings.slots)]
        spots = saved if self._fits(saved, seed) else seed
        self.settings.apply_spots(key, spots)
        self._tune_tries = 0
        self._tune_wait = 0
        return self.settings.slot_tuples()

    def _align_spots(self, frame, spots, rect):
        """Slide the spots onto the panel when the grid's guess is clearly wrong.

        The in-game UI size setting moves the whole ally list and screen width
        alone cannot see it, but the portraits can. Three things have to hold
        before anything moves, because a portrait score can be fooled by anything
        drawn over the panel: every visible row has to point the same way (the
        panel is rigid, it does not stretch), the shift has to be bigger than the
        portrait peak's own bias, and the place being moved to has to actually
        read a gauge where the current one does not.
        """
        grid = row_ys(frame.shape, rect=rect)
        peaks = row_peaks(frame, grid, rect)
        off = frame.shape[1] * REALIGN_OFF_FRAC_W
        settle = frame.shape[1] * REALIGN_SETTLE_FRAC_W
        seen = [peak - y for (peak, score), y in zip(peaks, grid)
                if score >= ROW_MIN_STRENGTH]
        if not seen or max(seen) - min(seen) > off:
            self._tune_wait = max(0, self._tune_wait - 1)
            return spots
        shift = sum(seen) / len(seen)
        # The shift is measured against the grid, so the target is the grid
        # moved, not the current spots moved.
        target = [(cx, y + shift, r) for (cx, _cy, r), y in zip(spots, grid)]
        if all(abs(t[1] - s[1]) <= settle for t, s in zip(target, spots)):
            return spots
        if self._tune_tries < self.REALIGN_TRIES and self._tune_wait == 0:
            self._tune_tries += 1
            self._tune_wait = self.REALIGN_GAP_FRAMES
            pitch = grid[1] - grid[0]
            found = [r for r in propose_rings(frame) if r.confidence >= 0.5]
            settled = []
            for cx, cy, r in target:
                near = [g for g in found if abs(g.cy - cy) <= 0.6 * pitch]
                settled.append((near[0].cx, near[0].cy, near[0].radius)
                               if near else (cx, cy, r))
            target = settled
        if self._read_score(frame, target) <= self._read_score(frame, spots):
            return spots
        if self.listening:
            # Only an armed tool commits a position. Disarmed it may look at a
            # better place for this frame's preview, but rewriting the user's
            # calibration because a window was left open is not a decision they
            # made - and until now that write was stopped only by accident, by the
            # same distance gate that was hiding the dead zone.
            self.settings.apply_spots(self._spot_key, target)
            self.calibrationChanged.emit()
        return target

    @staticmethod
    def _read_score(frame, spots) -> float:
        """How much gauge a position reports - the only proof worth moving for.

        The portraits say the panel is elsewhere, but a portrait peak can be made
        out of sunlit rock, and a panel-sized guess that reads nothing is a worse
        place to look than the grid. On the user's own 276 s recording the peaks
        moved the spots 130 times and left them blind for 230 frames in which the
        grid was reading a downed ally; five of those moves were worth it.
        """
        return sum(r.confidence for r in detect_spots(frame, spots) if r is not None)

    # -- the stop decision ----------------------------------------------------
    def _observations(self, spots, readings) -> list[SlotObservation]:
        """One record per ally we are watching, stamped with the capture clock.

        The annulus is only re-measured where the gauge read failed, and only
        while a song is playing: that is the case that has to be told apart from
        an ally who is simply not down, and it costs a 70x70 pixel crop.
        """
        # §5.3: losing the anchor is not a change in the game. While the picture box
        # is not being found, a slot that reads nothing may be an ally standing up or
        # may be our geometry pointing at the sky, and only the first of those is
        # allowed to end a song.
        position_ok = (self._rect is not None and not self._reseeded
                       and self._pose_state != LOST)
        out = []
        for i, ((_cx, cy, cr), r) in enumerate(zip(spots, readings)):
            share, lit = None, 0
            if r is None:
                got = ring_presence(self._frame, _cx, cy, cr)
                if got is not None:
                    share, lit = got
            out.append(SlotObservation(
                slot_id=i, t=self._clock(), frame_id=self.source.frames,
                bars=None if r is None else r.bars,
                confidence=0.0 if r is None else float(r.confidence),
                standing=r is None and ally_bar(self._frame, cy, self._rect),
                position_ok=position_ok, fresh=self._frame_fresh,
                ring_share=share, panel_lit=lit,
                accept_empty_ring=self.settings.revive_empty_ring_counts))
        return out

    def _song_ended(self, obs: list[SlotObservation]) -> int | None:
        """Which ally, if any, is standing firmly enough to be worth stopping for.

        "v1" is the rollback path: thirty consecutive frames of the same three
        valued flag, counted in frames rather than in time.
        """
        if self.settings.stop_policy != "v2":
            flags = [True if o.downed else (False if o.standing else None)
                     for o in obs]
            return 0 if self.watcher.update(flags) else None
        for o in obs:
            if self.arbiter.observe(o):
                return o.slot_id
        return None

    @staticmethod
    def _fits(spots, seed):
        """Keep a geometry that is close to this frame's grid, and drop one that
        was measured on another resolution - a stale position is worse than the
        grid, because it reads nothing and looks like an ally who is standing."""
        if len(spots) != len(seed):
            return False
        pitch = abs(seed[1][1] - seed[0][1]) if len(seed) > 1 else seed[0][2] * 4.0
        reach = seed[0][2] * 3.0
        return all(abs(sx - cx) <= reach and abs(sy - cy) <= 0.5 * pitch
                   for (sx, sy, _sr), (cx, cy, _r) in zip(spots, seed))


    # -- the loop -----------------------------------------------------------
    @property
    def observations(self) -> list[SlotObservation]:
        """What each watched slot said on the last frame of a playing song.

        Empty while nothing is playing: the records are only built when the stop
        decision needs them, and an empty list tells the interface that.
        """
        return list(self._evidence)

    def _tick(self):
        frame_id = self.source.frames
        # Whether this tick brought a new picture or re-served the last one. A
        # stalled capture has to stop being evidence, not start being a revive.
        self._frame_fresh = frame_id != self._last_frame_id
        if self._frame_fresh:
            self._last_frame_id = frame_id
            self._frame = self.source.latest()
            if self._frame is not None and not self._geometry_proven:
                # A seed that has never found the panel must not be defended by the
                # vote. A recording that opens on a fade measures 232 px of black
                # where the game later shows 80, and holding that box for a second
                # and a half costs the whole opening.
                self.hud.reset()
            pose = None if self._frame is None else self.hud.measure(self._frame)
            self._rect = None if pose is None else pose.rect
            self._pose_state = "NONE" if pose is None else pose.state
            if self._frame is not None:
                self.validator.check(self._frame)
        frame = self._frame
        if frame is None:
            return
        now = self._clock()
        if now - self._last_preview >= 1.0 / PREVIEW_FPS_CAP:
            self._last_preview = now
            self.frame_ready.emit(frame)
        # The panel is at a fixed place, so these positions are ours to know; the
        # user never marks anything. Reading costs ~1 ms, so it runs every frame
        # even while disarmed and the preview shows what each spot reports.
        spots = self._align_spots(frame, self._spots_for(frame.shape, self._rect),
                                  self._rect)
        if self.listening:
            readings, drifted = detect_with_recovery(frame, spots)
            if drifted:
                spots = [tuple(s) for s in spots]
                for i in drifted:
                    got = readings[i]
                    if got.confidence < ADOPT_MIN_CONFIDENCE:
                        continue
                    spots[i] = (got.cx, got.cy, got.radius)
                self.settings.apply_spots(self._spot_key, spots)
                self.calibrationChanged.emit()
        else:
            readings = detect_spots(frame, spots)
        # How many allies there are is checked at the same spots the gauges are
        # read from, so a panel the aligner has moved down follows along, and the
        # count can never disagree with what the rows themselves report.
        rows = count_rows(frame, self._rect, spots)
        if rows or any(r is not None for r in readings):
            # Something is where we are looking. From here on the picture box we
            # confirmed is worth defending against a momentary disagreement.
            self._geometry_proven = True
        allies = self.party.update(rows)
        if allies != self._party:
            self._party = allies
            self.party_changed.emit(allies)
        readings = readings[:allies]
        self._readings = readings
        # Judged before the frame is announced, so the interface can show this
        # frame's reason: what each watched slot is currently allowed to say.
        # Building it costs a 70x70 crop per unread slot, so only while a song is
        # going - that is when "why has it not stopped yet" is the question.
        self._evidence = (self._observations(spots, readings)
                          if self.watcher.watching else [])
        self.readings_changed.emit(readings)
        if self.diag is not None:
            self.diag.frame(frame, self._rect, spots, readings, allies,
                            self.source.frames, stale=self.validator.stale,
                            rows=self.party.last_rows,
                            pending=self.party.pending, pose=self._pose_state,
                            fresh=self._frame_fresh)
        if not allies:
            # Solo, or a panel that is not reading. There is nothing to trigger on,
            # but a song that is already going keeps going: not seeing the allies
            # is not the same as seeing them stand up.
            self.trigger.reset()
            return
        self.trigger.allies = allies

        # The track runs until an ally is seen standing again. Dropping from three
        # segments to two is a revive in progress and deliberately keeps playing.
        if self.watcher.watching:
            who = self._song_ended(self._evidence)
            if who is not None:
                if self.diag is not None:
                    self.diag.event("stop", frame, self._rect, party=allies,
                                    slot=who, policy=self.settings.stop_policy,
                                    flags=[o.evidence() for o in self._evidence],
                                    held=self.watcher.held, ages=self.arbiter.ages())
                self.watcher.stop()
                self.arbiter.end_song()
                if self.player is not None:
                    self.player.stop()
                self.stopped.emit(who)

        if not self.listening or self.paused:
            return
        # One song at a time. While it is playing there is nothing to trigger: the
        # readings can go blank for a moment and come back, and calling play_next
        # again would restart the track from zero over the top of it.
        if self.watcher.watching:
            return
        bars = [None if r is None else r.bars for r in readings]
        if self.party.pending:
            # Someone who might be there is not the same as someone who is: while a
            # bigger party is only half-proved the count is not a settled answer,
            # and the last thing wanted here is deciding "everyone is down" against
            # a roster that may grow a row on the next frame.
            self.trigger.reset()
            return
        if not self.trigger.update(bars, self._clock()):
            return
        if self.player is None:
            return
        track = self.player.play_next()
        if track is None:
            self.error.emit("触发了但没有可播放的音乐")
            return
        self.watcher.start()
        self.arbiter.start_song(range(allies))
        if self.diag is not None:
            self.diag.event("fire", frame, self._rect, party=allies, bars=bars,
                            track=track.name, held=self.trigger.held)
        self.fired.emit(track.name)

    # -- calibration helpers ------------------------------------------------
    @property
    def readings(self) -> list[Ring | None]:
        return list(self._readings)
