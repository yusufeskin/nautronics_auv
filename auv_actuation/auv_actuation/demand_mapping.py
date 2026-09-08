"""Maps a commanded body wrench onto ArduSub's pilot-input axes.

The architecture this serves: ArduSub keeps the inner loop - attitude
stabilisation and depth hold, running at 400 Hz on real-time hardware with
failsafes that already work - and this stack owns the outer loop, deciding
where the vehicle should be and how fast it should be going. The boundary
between them is a normalised four-axis demand that looks exactly like a
pilot's joystick.

Two consequences are baked into this module rather than discovered later.

**Only four axes exist.** Surge, sway, heave and yaw. Roll and pitch are
ArduSub's business in STABILIZE and ALT_HOLD: their sticks command a lean
angle inside a configured limit, not a torque, so a roll or pitch torque
asked for here cannot be delivered. Rather than silently dropping it, the
mapping reports it as undelivered so the controller above can stop
integrating against a wrench it is never going to get.

**Yaw is a rate, not a heading.** ArduSub holds heading with its own EKF3 yaw
when the stick is centred, and that yaw is not the one this stack estimates -
it is magnetometer-aided and will disagree. So nothing here ever commands an
absolute heading. The outer loop commands a yaw *rate*, measures the result
with our own estimate, and closes the absolute loop itself. ArduSub's slow
heading-hold drift then becomes an ordinary disturbance for the outer loop to
reject rather than a competing opinion about where north is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Demand", "DemandLimits", "WrenchDemandMapper"]


@dataclass
class DemandLimits:
    """Full-scale calibration: the wrench produced at full stick deflection.

    MEASURE these in the pool. Command a fixed demand on one axis, wait for
    the vehicle to stop accelerating, and read the terminal velocity off the
    DVL; with the drag coefficient from the same experiment that gives force.
    Until then the numbers below are placeholders and the loop gains they
    imply are meaningless.
    """

    surge_full_scale_n: float = 40.0
    sway_full_scale_n: float = 30.0
    heave_full_scale_n: float = 40.0
    yaw_full_scale_nm: float = 8.0

    # MANUAL_CONTROL carries each axis as an integer in [-1000, 1000].
    max_count: int = 1000

    # In ALT_HOLD the vertical axis is a climb-rate demand around a neutral
    # point that means "hold this depth". In STABILIZE it is raw throttle and
    # neutral means "no vertical thrust", which on a positively buoyant
    # vehicle is not the same as holding depth.
    neutral_z: int = 500

    # Demands below this are sent as zero. ArduSub applies its own stick
    # deadband; asking for less than that produces nothing but noise on the
    # link. Set to 0 to disable.
    deadband_counts: int = 0


@dataclass
class Demand:
    x: int          # surge, forward positive
    y: int          # sway, right positive
    z: int          # throttle stick, around neutral_z, higher is ascend
    r: int          # yaw rate, clockwise positive

    # True when any axis was clipped. The controller above must stop
    # integrating while this is set, or it will wind up against a limit it
    # cannot push through.
    saturated: bool

    # The wrench that will actually be produced, after clipping and after
    # dropping the axes ArduSub does not accept. Publishing this lets the
    # controller compare what it asked for against what it got - the standard
    # anti-windup arrangement, and the only honest way to report that roll and
    # pitch torque went nowhere.
    realised_force: np.ndarray
    realised_torque: np.ndarray

    # Set when a roll or pitch torque was requested and discarded.
    dropped_roll_pitch: bool = False


@dataclass
class WrenchDemandMapper:
    limits: DemandLimits = field(default_factory=DemandLimits)

    # Threshold above which a discarded roll/pitch torque is worth reporting,
    # so that numerical dust from a controller does not raise a warning.
    roll_pitch_report_threshold_nm: float = 0.1

    def map(self, force, torque) -> Demand:
        """Convert a body-frame wrench (FLU) into an ArduSub demand (FRD)."""
        f = np.asarray(force, dtype=float)
        t = np.asarray(torque, dtype=float)

        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(t)):
            return self.neutral()

        # Normalise each axis against its full-scale calibration.
        surge = self._normalise(f[0], self.limits.surge_full_scale_n)
        # ArduSub's pilot axes are not a single consistent frame, which is
        # exactly why every one of them is spelled out here rather than
        # derived. Lateral is right-positive against this stack's FLU
        # left-positive, and yaw is clockwise-positive against FLU's
        # counter-clockwise - both flip. The vertical axis is a throttle
        # stick, not an FRD axis: pushing it up ascends, which already agrees
        # with FLU's up-positive z, so it does NOT flip.
        sway = self._normalise(-f[1], self.limits.sway_full_scale_n)
        heave = self._normalise(f[2], self.limits.heave_full_scale_n)
        yaw = self._normalise(-t[2], self.limits.yaw_full_scale_nm)

        saturated = any(abs(v) > 1.0 for v in (surge, sway, heave, yaw))

        surge_c = self._counts(surge)
        sway_c = self._counts(sway)
        heave_c = self._counts(heave)
        yaw_c = self._counts(yaw)

        # What the vehicle will actually feel, converted back to FLU so the
        # controller can compare it against what it asked for in its own
        # frame.
        realised_force = np.array([
            self._denormalise(surge_c, self.limits.surge_full_scale_n),
            -self._denormalise(sway_c, self.limits.sway_full_scale_n),
            self._denormalise(heave_c, self.limits.heave_full_scale_n),
        ])
        realised_torque = np.array([
            0.0,    # roll: ArduSub stabilises it, we cannot command torque
            0.0,    # pitch: likewise
            -self._denormalise(yaw_c, self.limits.yaw_full_scale_nm),
        ])

        dropped = bool(
            max(abs(t[0]), abs(t[1])) > self.roll_pitch_report_threshold_nm
        )

        return Demand(
            x=surge_c,
            y=sway_c,
            z=self._clamp(self.limits.neutral_z + heave_c),
            r=yaw_c,
            saturated=saturated,
            realised_force=realised_force,
            realised_torque=realised_torque,
            dropped_roll_pitch=dropped,
        )

    def neutral(self) -> Demand:
        """The command that asks the vehicle for nothing.

        In ALT_HOLD this means hold the current depth and stop translating -
        not "cut the thrusters", which on a positively buoyant vehicle would
        mean surfacing. This is what the bridge sends on a watchdog timeout,
        so it has to be the safe thing rather than the inert thing.
        """
        return Demand(
            x=0,
            y=0,
            z=self.limits.neutral_z,
            r=0,
            saturated=False,
            realised_force=np.zeros(3),
            realised_torque=np.zeros(3),
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalise(value: float, full_scale: float) -> float:
        if full_scale <= 0.0:
            return 0.0
        return value / full_scale

    def _counts(self, normalised: float) -> int:
        clipped = max(-1.0, min(1.0, normalised))
        counts = int(round(clipped * self.limits.max_count))
        if abs(counts) < self.limits.deadband_counts:
            return 0
        return counts

    def _denormalise(self, counts: int, full_scale: float) -> float:
        return (counts / self.limits.max_count) * full_scale

    def _clamp(self, counts: int) -> int:
        return max(0, min(self.limits.max_count, counts))
