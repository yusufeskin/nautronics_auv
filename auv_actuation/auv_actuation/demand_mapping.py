"""Maps an axis demand onto ArduSub's RC channels, for ALT_HOLD.

This stack drives ArduSub the way a pilot with a joystick would: one PWM
value per axis, sent as RC channel overrides. ArduSub does the motor mixing,
holds roll and pitch level, holds depth while the throttle is centred and
holds heading while yaw is centred. Nothing here allocates thrust.

What each channel means in ALT_HOLD, and therefore what goes on it:

    ch5 forward    thrust fraction          <- surge, [-1, 1]
    ch6 lateral    thrust fraction          <- sway, [-1, 1]
    ch3 throttle   climb rate               <- climb_rate, m/s
    ch4 yaw        turn rate                <- yaw_rate, rad/s
    ch1 pitch      lean angle               <- always neutral
    ch2 roll       lean angle               <- always neutral

**Conventions.** ArduSub's sticks are not one consistent frame. Lateral is
right-positive against this stack's FLU left-positive, and yaw is
clockwise-positive against FLU's counter-clockwise, so both flip. Throttle up
ascends, which already agrees with FLU's up-positive z. Every flip is written
out here and nowhere else.

**Deadzones.** ArduSub ignores a band around neutral on each stick (THR_DZ
for throttle, RCn_DZ for the others). A demand mapped linearly from 1500
would lose everything inside that band, so a non-zero demand starts at the
edge of the deadzone and the remaining travel is scaled over it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ChannelLimits", "PwmDemand", "AxisDemandMapper"]


@dataclass
class ChannelLimits:
    neutral_pwm: int = 1500
    # Full deflection either side of neutral: 1100..1900.
    span_pwm: int = 400

    # The rates ArduSub produces at full stick. READ THESE OFF THE VEHICLE:
    # PILOT_SPEED_UP and PILOT_SPEED_DN (cm/s) for climb, and verify the yaw
    # scaling by measurement. Wrong values here do not destabilise anything -
    # the outer loop still closes on our estimate - but they make the rate
    # caps in pose_controller.yaml mean something other than they say.
    max_climb_rate_up: float = 0.5        # m/s at full stick up
    max_climb_rate_down: float = 0.5      # m/s at full stick down
    max_yaw_rate: float = 1.0             # rad/s at full stick

    # ArduSub's deadzones, in PWM either side of neutral.
    throttle_deadzone_pwm: int = 100      # THR_DZ
    yaw_deadzone_pwm: int = 0             # RC4_DZ
    forward_deadzone_pwm: int = 0         # RC5_DZ
    lateral_deadzone_pwm: int = 0         # RC6_DZ


@dataclass
class PwmDemand:
    pitch: int        # ch1
    roll: int         # ch2
    throttle: int     # ch3
    yaw: int          # ch4
    forward: int      # ch5
    lateral: int      # ch6

    # True when any axis asked for more than full stick and was clipped.
    saturated: bool = False


@dataclass
class AxisDemandMapper:
    limits: ChannelLimits

    def map(self, surge: float, sway: float, climb_rate: float,
            yaw_rate: float) -> PwmDemand:
        """Convert one axis demand (FLU, see module docstring) to PWM."""
        values = (surge, sway, climb_rate, yaw_rate)
        if not all(math.isfinite(v) for v in values):
            return self.neutral()

        lim = self.limits
        climb_scale = lim.max_climb_rate_up if climb_rate >= 0.0 else lim.max_climb_rate_down

        fractions = {
            "forward": surge,
            "lateral": -sway,                                   # FLU left+ -> right+
            "throttle": _ratio(climb_rate, climb_scale),
            "yaw": _ratio(-yaw_rate, lim.max_yaw_rate),         # CCW+ -> CW+
        }
        saturated = any(abs(f) > 1.0 for f in fractions.values())

        return PwmDemand(
            pitch=lim.neutral_pwm,
            roll=lim.neutral_pwm,
            throttle=self._pwm(fractions["throttle"], lim.throttle_deadzone_pwm),
            yaw=self._pwm(fractions["yaw"], lim.yaw_deadzone_pwm),
            forward=self._pwm(fractions["forward"], lim.forward_deadzone_pwm),
            lateral=self._pwm(fractions["lateral"], lim.lateral_deadzone_pwm),
            saturated=saturated,
        )

    def neutral(self) -> PwmDemand:
        """All sticks centred.

        In ALT_HOLD this means hold depth, hold heading, stop pushing
        horizontally - the vehicle stays roughly where it is. That is what a
        watchdog timeout sends, so it has to be the safe command, not merely
        the inert one.
        """
        n = self.limits.neutral_pwm
        return PwmDemand(pitch=n, roll=n, throttle=n, yaw=n, forward=n, lateral=n)

    def _pwm(self, fraction: float, deadzone: int) -> int:
        lim = self.limits
        if fraction == 0.0:
            return lim.neutral_pwm
        magnitude = min(1.0, abs(fraction))
        deadzone = max(0, min(deadzone, lim.span_pwm))
        offset = deadzone + (lim.span_pwm - deadzone) * magnitude
        sign = 1.0 if fraction > 0.0 else -1.0
        return int(round(lim.neutral_pwm + sign * offset))


def _ratio(value: float, full_scale: float) -> float:
    if full_scale <= 0.0:
        return 0.0
    return value / full_scale

