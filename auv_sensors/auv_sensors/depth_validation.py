"""Pressure-to-depth conversion and validation, free of ROS dependencies.

Depth is the one absolute, non-drifting state this vehicle measures: yaw and
horizontal position both drift, z does not. That makes the depth channel the
backbone the rest of the estimate hangs off, and makes its zero reference
worth being fussy about - a surface pressure captured with the vehicle
already half submerged biases every depth for the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

__all__ = ["DepthConverter", "DepthLimits", "DepthSample", "DepthVerdict"]

STANDARD_GRAVITY = 9.80665
NOMINAL_ATMOSPHERIC_PA = 101325.0


@dataclass
class DepthLimits:
    water_density_kg_m3: float = 1000.0

    # Plausible absolute pressure band. Outside this the sensor is faulty or
    # unplugged rather than deep.
    min_pressure_pa: float = 50_000.0
    max_pressure_pa: float = 400_000.0   # ~30 m of fresh water

    # The vehicle cannot change depth faster than this. A step beyond it is a
    # sensor glitch, not a dive.
    max_depth_rate_mps: float = 3.0
    max_dt_s: float = 1.0

    max_clock_skew_s: float = 0.5

    # Depth measurement variance in m^2. A Bar30-class sensor resolves better
    # than this; the margin covers surface wave action and thruster wash over
    # the port.
    depth_variance_m2: float = 4.0e-4    # (0.02 m)^2

    # Refuse to report depth until a surface zero has been taken, rather than
    # silently using the nominal atmosphere. Set False for bench work.
    require_surface_zero: bool = True


@dataclass
class DepthSample:
    stamp: float
    pressure_pa: float


@dataclass
class DepthVerdict:
    accepted: bool
    reason: str = ""
    degraded_reason: str = ""
    depth_m: float = 0.0        # positive down


@dataclass
class DepthConverter:
    limits: DepthLimits = field(default_factory=DepthLimits)

    _surface_pressure_pa: float = NOMINAL_ATMOSPHERIC_PA
    _zeroed: bool = False
    _last_stamp: float | None = None
    _last_depth: float | None = None

    @property
    def zeroed(self) -> bool:
        return self._zeroed

    @property
    def surface_pressure_pa(self) -> float:
        return self._surface_pressure_pa

    def zero_at(self, pressure_pa: float) -> bool:
        """Adopt ``pressure_pa`` as the surface reference.

        Called from the deck with the vehicle floating, before arming.
        Returns False for an implausible reading so a mistimed call - taken
        with the vehicle already under - cannot poison the run.
        """
        if not math.isfinite(pressure_pa):
            return False
        if not (self.limits.min_pressure_pa <= pressure_pa <= self.limits.max_pressure_pa):
            return False
        # More than ~0.5 m of water above nominal atmosphere means this was
        # not taken at the surface.
        if abs(pressure_pa - NOMINAL_ATMOSPHERIC_PA) > 5000.0:
            return False
        self._surface_pressure_pa = pressure_pa
        self._zeroed = True
        return True

    def check(self, sample: DepthSample, now: float) -> DepthVerdict:
        skew = sample.stamp - now
        if abs(skew) > self.limits.max_clock_skew_s:
            direction = "ahead of" if skew > 0 else "behind"
            return DepthVerdict(False, f"timestamp {abs(skew):.2f} s {direction} local clock")

        pressure = sample.pressure_pa
        if not math.isfinite(pressure):
            return DepthVerdict(False, "non-finite pressure")

        if not (self.limits.min_pressure_pa <= pressure <= self.limits.max_pressure_pa):
            return DepthVerdict(
                False,
                f"pressure {pressure:.0f} Pa outside the plausible "
                f"{self.limits.min_pressure_pa:.0f}-{self.limits.max_pressure_pa:.0f} Pa band",
            )

        if self.limits.require_surface_zero and not self._zeroed:
            return DepthVerdict(
                False,
                "no surface zero taken; call the zero service with the vehicle floating",
            )

        depth = (pressure - self._surface_pressure_pa) / (
            self.limits.water_density_kg_m3 * STANDARD_GRAVITY
        )

        if self._last_stamp is not None and self._last_depth is not None:
            dt = sample.stamp - self._last_stamp
            if 0.0 < dt <= self.limits.max_dt_s:
                rate = abs(depth - self._last_depth) / dt
                if rate > self.limits.max_depth_rate_mps:
                    return DepthVerdict(
                        False,
                        f"depth changing at {rate:.1f} m/s, above the "
                        f"{self.limits.max_depth_rate_mps:.1f} m/s limit",
                    )

        self._last_stamp = sample.stamp
        self._last_depth = depth

        degraded = ""
        if not self._zeroed:
            degraded = "using nominal atmospheric pressure; no surface zero taken"
        elif depth < -0.25:
            # Above the recorded surface by more than sensor noise: either the
            # zero was taken underwater or the barometric pressure has moved.
            degraded = f"depth {depth:.2f} m is above the surface reference"

        return DepthVerdict(True, degraded_reason=degraded, depth_m=depth)
