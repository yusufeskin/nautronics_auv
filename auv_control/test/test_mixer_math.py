"""Unit tests for the thruster mixing / thrust-to-PWM math."""

import numpy as np
import pytest

from auv_control import mixer_math
from auv_control.mixer_math import input_to_pwm, mix, MIXING_MATRIX


def full_pwm(control_vector):
    """Mirror ThrusterMixer.cmd_vel_callback: mix, then convert every thruster."""
    return [input_to_pwm(v) for v in mix(MIXING_MATRIX, np.array(control_vector))]


def test_zero_input_is_neutral_pwm():
    assert input_to_pwm(0.0) == mixer_math.PWM_NEUTRAL == 1500


def test_zero_twist_gives_neutral_on_all_eight_thrusters():
    assert full_pwm([0.0] * 6) == [1500] * 8


def test_full_forward_and_reverse_hit_pwm_limits():
    assert input_to_pwm(1.0) == 2000
    assert input_to_pwm(-1.0) == 1000


# Calibration table documented at the top of thruster_mixer.py
CALIBRATION_TABLE = {
    1.0: 2000, 0.9: 1933, 0.8: 1898, 0.7: 1860, 0.6: 1821, 0.5: 1788,
    0.4: 1760, 0.3: 1724, 0.2: 1675, 0.1: 1596, 0.0: 1500,
    -0.1: 1428, -0.2: 1357, -0.3: 1299, -0.4: 1246, -0.5: 1210,
    -0.6: 1149, -0.7: 1122, -0.8: 1089, -0.9: 1038, -1.0: 1000,
}


def test_matches_documented_calibration_table():
    actual = {v: input_to_pwm(v) for v in CALIBRATION_TABLE}
    assert actual == CALIBRATION_TABLE


def test_forward_and_reverse_use_separate_max_thrust():
    # 0.5 forward -> 2.34 kgf -> between (1.69, 1750) and (2.54, 1800)
    # 0.5 reverse -> -1.76 kgf -> between (-1.86, 1200) and (-1.37, 1250)
    assert input_to_pwm(0.5) == int(1750 + (2.34 - 1.69) / (2.54 - 1.69) * 50)
    assert input_to_pwm(-0.5) == int(1200 + (-1.76 + 1.86) / (-1.37 + 1.86) * 50)


@pytest.mark.parametrize('input_val, expected', [
    (1.5, 2000), (10.0, 2000), (-1.5, 1000), (-10.0, 1000),
])
def test_out_of_range_input_is_clamped(input_val, expected):
    assert input_to_pwm(input_val) == expected


def test_pwm_is_monotonic_in_input():
    inputs = np.linspace(-1.0, 1.0, 201)
    pwms = [input_to_pwm(v) for v in inputs]
    assert all(a <= b for a, b in zip(pwms, pwms[1:]))
    assert all(isinstance(p, int) for p in pwms)


def test_custom_pwm_limits_are_respected():
    assert input_to_pwm(1.0, pwm_min=1100, pwm_max=1900) == 1900
    assert input_to_pwm(-1.0, pwm_min=1100, pwm_max=1900) == 1100


def test_thruster_data_is_sorted_for_interp():
    # np.interp silently returns garbage if xp is not increasing.
    assert np.all(np.diff(mixer_math.THRUSTER_DATA_THRUST) > 0)
    assert np.all(np.diff(mixer_math.THRUSTER_DATA_PWM) > 0)
    assert mixer_math.THRUSTER_DATA_THRUST[0] == -mixer_math.MAX_REVERSE_THRUST
    assert mixer_math.THRUSTER_DATA_THRUST[-1] == mixer_math.MAX_FORWARD_THRUST


def test_mixing_matrix_axis_mapping():
    assert MIXING_MATRIX.shape == (8, 6)
    # surge -> T5 (+), sway -> T6 (-), heave -> T3, roll -> T2, pitch -> T1 (-), yaw -> T4
    expected = {0: (4, 1.0), 1: (5, -1.0), 2: (2, 1.0), 3: (1, 1.0), 4: (0, -1.0), 5: (3, 1.0)}
    for axis, (thruster, sign) in expected.items():
        unit = np.zeros(6)
        unit[axis] = 1.0
        out = mix(MIXING_MATRIX, unit)
        assert out[thruster] == sign
        assert np.count_nonzero(out) == 1


def test_mix_does_not_scale_when_within_limits():
    out = mix(MIXING_MATRIX, np.array([0.5, 0.0, 0.0, 0.0, 0.0, -0.25]))
    assert out[4] == 0.5
    assert out[3] == -0.25


def test_mix_normalizes_and_keeps_ratios_when_saturated():
    out = mix(MIXING_MATRIX, np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert out[4] == pytest.approx(1.0)
    assert out[5] == pytest.approx(-0.5)
    assert np.max(np.abs(out)) == pytest.approx(1.0)


def test_full_pipeline_surge_and_pitch():
    pwm = full_pwm([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    assert pwm[4] == 2000         # T5 full forward
    assert pwm[0] == 1210         # T1 = -0.5 (pitch weight is -1)
    assert pwm[1:4] == [1500, 1500, 1500]
    assert pwm[6:] == [1500, 1500]
