"""The loop that ties capture, detection, triggering and playback together.

Runs on the Qt GUI thread off a QTimer. The capture backend already has its own
thread, so this only ever grabs the newest frame and reads a couple of fixed
spots in it; nothing here blocks.

Preview and listening are separate states on purpose: you need to see the game
to calibrate the gauges, and you should not be triggering music while doing it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

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
from .geometry import GeometryManager
from .hud_tracker import LOCKED, OCCLUDED, HudTracker
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

# How long a proposed position may go unseen before its collected votes are thrown
# away. A panel that came back on its own, or a candidate that was only ever one
# frame's opinion, must not stay eligible to be believed later.
CANDIDATE_STALE_S = 1.0


@dataclass(frozen=True)
class PreviewSnapshot:
    """One frame, complete: the picture, where the panel was put in it, and what
    each gauge there reported.

    The interface used to assemble this out of three signals that arrived at three
    moments - the picture when the capture produced it, the circles whenever the
    calibration last changed, the counts on their own tick - so a preview could,
    and did, show a new frame under old circles. One immutable object per frame is
    the only way to promise they agree.

    `frame` is safe to hold on to: the capture publishes a fresh array per frame
    (`capture.py`), and every signal here is a same-thread call, so nothing writes
    under the QImage the widget builds from it.
    """

    frame_id: int
    timestamp_s: float
    frame: object
    render_rect: tuple | None
    spots: tuple
    readings: tuple
    party_count: int
    hud_state: str
    geometry_state: str
    geometry_source: str


class Engine(QObject):
    # Following the portraits is free; looking at the gauges to nail the last few
    # pixels costs about a second, so it is tried a few times per move.
    REALIGN_TRIES = 3
    REALIGN_GAP_FRAMES = 100

    frame_ready = Signal(object)          # newest captured frame (BGR ndarray)
    preview_snapshot = Signal(object)     # PreviewSnapshot: frame + spots + readings
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
        self._raw_rect = None
        # One second of frames is the vote on where the picture box is; a fade or
        # an overlay in a bar cannot move the geometry on its own.
        self.hud = HudTracker(window=max(5, int(settings.target_fps)),
                              relocate_frames=self._relocate_frames())
        # Diagnostic only: a stalled capture is reported, never used to
        # accept or refuse evidence. See nrdoom/frame_quality.py.
        self.validator = FrameValidator()
        self._geom_sig: tuple | None = None
        self._persisted_sig: tuple | None = None
        # Where to look, and what it would take to start believing that. Holds the
        # active layout (what this frame reads), the verified one (what is kept) and
        # any candidate in between.
        self.geom = GeometryManager()
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
        # A new capture is a new window: nothing measured before it describes this
        # one, and the first frame has to re-seed rather than inherit the old box.
        self._geom_sig = None
        self._persisted_sig = None

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

    def _signature(self, shape, rect) -> tuple:
        """Everything the layout is derived from, as one comparable value.

        The rendered width belongs here, not only the top edge: `gauge_spots()`
        multiplies the panel's column, its row pitch and the gauge radius out of it,
        so a picture that really became 80 px narrower is a different layout even in
        a capture frame of exactly the same size. The old test looked at the capture
        size and the top edge only, which meant the one case the width lock in
        `HudTracker` creates - a stable change of the *side* bars - was invisible.
        """
        if rect is None:
            return (shape[1], shape[0], 0, 0, shape[1])
        return (shape[1], shape[0], rect[0], rect[1], rect[2])

    def _moved(self, sig) -> bool:
        last = self._geom_sig
        if last is None or last[:2] != sig[:2]:
            return True                       # a different capture size
        # Not an integer comparison: the letterbox is measured by sampling every
        # 8th line, so a box that has not moved does come back a pixel apart, and
        # reseeding on that would throw away the roster every few frames.
        return any(abs(a - b) > RECT_RESEED_PX for a, b in zip(last[2:], sig[2:]))

    def _rememberable(self) -> bool:
        """Is the picture box we are looking at stable enough to keep a layout for?

        ACQUIRING and LOST are the moments when the measurement is somebody else's
        frame - a fade, a cinematic, a window being dragged. Geometry derived from
        one is fine to look through for the next frame and must not be written
        anywhere that outlives it.
        """
        return self._pose_state in (LOCKED, OCCLUDED)

    def _keep(self, spots, source: str, frames: int = 0) -> None:
        """Remember the layout that is now believed, for the box it was proved on.

        Only ever called with a *verified* layout. Whatever a frame happens to be
        looking at right now is not what gets filed - that is the difference between
        using a position and keeping one, and it is the whole reason a single grazing
        read used to outlive its frame.
        """
        if self._geom_sig is None or not self._rememberable():
            return
        self.settings.remember_layout((self._geom_sig[0], self._geom_sig[1]),
                                      self._geom_sig[2:], spots, source, frames,
                                      self._clock())
        self._persisted_sig = self._geom_sig

    def _expire(self, now: float) -> None:
        """Stop collecting evidence for a proposal nobody has seen for a while.

        Votes are only ever added by the frame that sees them, so a candidate that
        has gone quiet is one the game stopped agreeing with - and leaving it alive
        would let a stale handful of votes carry on accruing against a panel that
        has since moved back.
        """
        cand = self.geom.candidate
        if cand is not None and now - cand.last_s > CANDIDATE_STALE_S:
            self.geom.refuse(now)
            self.settings.use_spots(self.geom.active)

    def _spots_for(self, shape, rect):
        """The ally spots for this frame: seeded, refined, then remembered.

        The panel is at a fixed place on screen, so nobody has to mark it. What
        the in-game UI size setting changes is *where* that fixed place lands, so
        a geometry that has actually read well is kept per resolution rather than
        re-derived every launch, and a window resize gets its own instead of
        inheriting a stale one.

        The letterbox counts as a resize: the geometry hangs off the picture box,
        so when the box moves the seed has to move with it. Without that the first
        frame decides everything - and a recording that opens on a fade measures its
        top bar at 312 px instead of 80 and seeds the whole session 232 px below the
        panel. The other half of that, which the first version of the fix left open:
        a box that moves *legitimately* has to be followed, or locking the width
        would freeze a layout that is now forty pixels too wide.
        """
        sig = self._signature(shape, rect)
        self._reseeded = self._moved(sig)
        if not self._reseeded:
            if self._rememberable() and sig != self._persisted_sig:
                # The box has only just been confirmed - the session's one reseed
                # happened while it was still ACQUIRING, and nothing has moved since
                # - so this is the first moment a layout is worth keeping at all.
                self._keep(self.geom.verified, self.geom.source)
            return self.geom.active or self.settings.slot_tuples()
        self._spot_key = "%dx%d" % (shape[1], shape[0])
        first = self._geom_sig is None
        self._geom_sig = sig
        self._geometry_proven = False
        self.party.reset()
        self._party = 0
        seed = gauge_spots(shape, rect=rect)
        proved = self.settings.layout_for((sig[0], sig[1]), sig[2:])
        if proved is not None:
            saved = [(s.cx, s.cy, s.radius) for s in proved.slots]
        elif first:
            # Nothing has been proved for any box yet in this session, so whatever
            # the config was loaded with is the starting hint: that is how a geometry
            # earned by an older build, or marked by hand, survives the change of
            # store.
            saved = self.settings.slot_tuples()
        else:
            # This session has already looked at a different box, so the live
            # positions belong to *that* one, and `_fits` is loose enough (three
            # radii) to wave a layout eighty pixels away straight through. That is
            # how a stale geometry used to outlive a resize.
            saved = []
        if saved and self._fits(saved, seed):
            spots, source = saved, (proved.source if proved else "cache")
        else:
            spots, source = seed, "seed"
        now = self._clock()
        self.geom.settle(spots, source, now)
        self.settings.use_spots(self.geom.active)
        self._keep(self.geom.verified, source)
        self._tune_tries = 0
        self._tune_wait = 0
        return self.geom.active

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
                # Two constraints, not one. `propose_rings` walks the whole search
                # band and hands its answers back sorted top to bottom, so the old
                # `near[0]` was simply the highest ring on a row whose vertical
                # offset was tolerated - anything bright on that row could become
                # the calibration, hundreds of pixels away from the panel. A gauge
                # is a rigid sprite: the column is where the ally list is, so a
                # candidate only counts within its own neighbourhood, and the
                # closest of those wins rather than the first one found.
                near = [g for g in found
                        if abs(g.cy - cy) <= 0.6 * pitch and abs(g.cx - cx) <= 2.0 * r]
                if not near:
                    settled.append((cx, cy, r))
                    continue
                best = min(near, key=lambda g: (abs(g.cy - cy), abs(g.cx - cx)))
                settled.append((best.cx, best.cy, best.radius))
            target = settled
        now = self._clock()
        score = self._read_score(frame, target)
        if score <= self._read_score(frame, spots):
            return spots
        # Look there from this frame on, and start collecting the evidence that would
        # make it the layout we keep. Before this, one frame's opinion of the portraits
        # *was* the calibration, and a portrait peak fooled by sunlit rock took the
        # whole ally list with it.
        self.geom.propose(target, "realign", self.source.frames, now, score,
                          self._read_score(frame, self.geom.verified))
        if self.listening:
            # Disarmed, this is still only a look: the preview gets the positions
            # from the snapshot, and the user's saved geometry stays where it was.
            self.settings.use_spots(self.geom.active)
        return self.geom.active

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

    def _settle_geometry(self, now: float, frame) -> None:
        """Say what a proposal has earned, and whether the kept layout still has it.

        Both directions live here on purpose: the portrait aligner and the jitter
        search are two ways of discovering the same thing, and neither of them gets
        to keep its own private idea of what has been proved.

        One question is asked of every position - how much gauge is actually at that
        place - and the same answer decides both halves. Asking the candidate one way
        and the kept layout another is how a position could be believed into the
        store and judged blind out of it in the same frame.
        """
        if not self._frame_fresh:
            # A re-served picture is the same observation again. Counting it twice is
            # how a stalled capture used to reach "several frames of agreement".
            return
        self._expire(now)
        cand = self.geom.candidate
        if cand is None and not self.geom.fallback:
            return
        kept = self._read_score(frame, self.geom.verified)
        if cand is not None:
            # The frames that arrive *after* the move are the ones that decide it:
            # once the spots are where the panel is, nothing proposes any more, so
            # without this a correct move could never reach three votes.
            self.geom.confirm(self.source.frames, now,
                              self._read_score(frame, self.geom.active), kept)
            if self.geom.earned(now):
                votes = self.geom.candidate.votes
                self.geom.accept(now)
                self.settings.use_spots(self.geom.verified)
                self._keep(self.geom.verified, self.geom.source, frames=votes)
                self.calibrationChanged.emit()
                return
        restored = self.geom.report_read(now, kept)
        if restored is not None:
            self.settings.use_spots(restored)
            self._keep(restored, "rollback")
            self.calibrationChanged.emit()

    # -- the stop decision ----------------------------------------------------
    def _observations(self, spots, readings) -> list[SlotObservation]:
        """One record per ally we are watching, stamped with the capture clock.

        The annulus is only re-measured where the gauge read failed, and only
        while a song is playing: that is the case that has to be told apart from
        an ally who is simply not down, and it costs a 70x70 pixel crop.
        """
        # §5.3 and P1-3: losing the anchor is not a change in the game. A slot that
        # reads nothing may be an ally standing up, or may be our geometry pointing
        # at the sky - and only the first of those is allowed to end a song. The box
        # has to be the one the vote settled on, and the positions have to be ones the
        # tool has stopped arguing about: a candidate still collecting evidence is a
        # position that is being re-decided, which is the same kind of not-knowing.
        position_ok = (self._rect is not None and not self._reseeded
                       and self._pose_state == LOCKED
                       and self.geom.state() == "verified")
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

    # -- what the interface can ask for ---------------------------------------
    def redetect(self) -> bool:
        """Forget where the panel was and go and find it again.

        Clears the working positions and the picture-box vote, but not the layouts
        that were proved and written down: a user who moved the window wants a new
        measurement, not an amnesia that has to re-learn every resolution they own.
        """
        if self._frame is None:
            return False
        self.geom.refuse(self._clock())
        self.hud.reset()
        self._geom_sig = None
        self._persisted_sig = None
        self._geometry_proven = False
        self.party.reset()
        self._party = 0
        return True

    def reset_layout(self) -> int:
        """Drop the proved geometry for this picture box, and only this one."""
        if self._geom_sig is None:
            return 0
        gone = self.settings.forget_layout((self._geom_sig[0], self._geom_sig[1]),
                                           self._geom_sig[2:])
        self.redetect()
        return gone

    def restore_layout(self) -> bool:
        """Go back to the position that was trusted before the current one."""
        restored = self.geom.restore(self._clock())
        if restored is None:
            return False
        self.settings.use_spots(restored)
        self._keep(restored, "manual")
        self.calibrationChanged.emit()
        return True

    def _log_geometry(self, now: float) -> None:
        """Write the state changes the manager collected, each with its own crop.

        "Why did the box move" is the question this exists to answer, and the answer
        is a sequence: proposed, then verified or refused, then committed or rolled
        back. Guessing it afterwards from the frame lines is what the raw-versus-locked
        rect fields are here for, but a sequence beats a reconstruction.
        """
        for kind, when, spots in self.geom.drain():
            self.diag.event("geometry_" + kind, self._frame, self._rect,
                            t=round(when, 3), pose=self._pose_state,
                            geom=self.geom.state(),
                            spots=[[round(v, 1) for v in s] for s in spots],
                            sig=None if self._geom_sig is None
                               else list(self._geom_sig))

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
            self._raw_rect = None if pose is None else pose.measured
            self._pose_state = "NONE" if pose is None else pose.state
            if self._frame is not None:
                self.validator.check(self._frame)
        frame = self._frame
        if frame is None:
            return
        now = self._clock()
        due = now - self._last_preview >= 1.0 / PREVIEW_FPS_CAP
        if due:
            self._last_preview = now
            # Kept for the feed label, which counts delivered captures and names the
            # backend. The picture the user looks for goes out with the snapshot
            # below, after this frame's positions are known.
            self.frame_ready.emit(frame)
        # The panel is at a fixed place, so these positions are ours to know; the
        # user never marks anything. Reading costs ~1 ms, so it runs every frame
        # even while disarmed and the preview shows what each spot reports.
        spots = self._align_spots(frame, self._spots_for(frame.shape, self._rect),
                                  self._rect)
        if self.listening:
            readings, drifted = detect_with_recovery(frame, spots)
            if drifted:
                moved = [(i, readings[i]) for i in drifted
                         if readings[i].confidence >= ADOPT_MIN_CONFIDENCE]
                if moved:
                    cand = [tuple(s) for s in spots]
                    for i, got in moved:
                        cand[i] = (got.cx, got.cy, got.radius)
                    score = self._read_score(frame, cand)
                    self.geom.propose(cand, "recovery", frame_id, now, score,
                                      self._read_score(frame, self.geom.verified))
                    spots = self.geom.active
                    self.settings.use_spots(spots)
            self._settle_geometry(now, frame)
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
        if due:
            self.preview_snapshot.emit(PreviewSnapshot(
                frame_id=frame_id, timestamp_s=now, frame=frame,
                render_rect=self._rect, spots=tuple(tuple(s) for s in spots),
                readings=tuple(readings), party_count=allies,
                hud_state=self._pose_state, geometry_state=self.geom.state(),
                geometry_source=self.geom.source))
        if self.diag is not None:
            cand = self.geom.candidate
            self.diag.frame(frame, self._rect, spots, readings, allies,
                            self.source.frames, stale=self.validator.stale,
                            rows=self.party.last_rows,
                            pending=self.party.pending, pose=self._pose_state,
                            fresh=self._frame_fresh,
                            # Both boxes, not just the one in use: the difference
                            # between "this frame disagreed" and "the tracker moved"
                            # is the whole question when a layout went somewhere it
                            # should not have.
                            raw_rect=None if self._raw_rect is None
                                    else list(self._raw_rect),
                            kept_spots=[[round(v, 1) for v in s]
                                        for s in self.geom.verified],
                            geom=self.geom.state(), geom_src=self.geom.source,
                            candidate=None if cand is None else {
                                "spots": [[round(v, 1) for v in s] for s in cand.spots],
                                "votes": cand.votes,
                                "agreed": len(cand.better_ids),
                                "age": round(now - cand.first_s, 3)},
                            score=round(sum(float(r.confidence)
                                            for r in readings if r), 3))
            self._log_geometry(now)
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
