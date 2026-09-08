"""Depth gate logic tests.

Depth is the only non-drifting state on the vehicle, so the surface-zero
discipline is enforced here rather than left to whoever is on the deck.
"""

import pytest

from auv_sensors.depth_validation import (
    NOMINAL_ATMOSPHERIC_PA,
    STANDARD_GRAVITY,
    DepthConverter,
    DepthLimits,
    DepthSample,
)

DENSITY = 1000.0


def pressure_at(depth_m, surface_pa=NOMINAL_ATMOSPHERIC_PA):
    return surface_pa + DENSITY * STANDARD_GRAVITY * depth_m


def make_converter(**overrides):
    overrides.setdefault("require_surface_zero", False)
    return DepthConverter(limits=DepthLimits(**overrides))


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.5, 3.0, 10.0])
def test_pressure_converts_to_depth(depth):
    converter = make_converter()
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    verdict = converter.check(DepthSample(0.0, pressure_at(depth)), now=0.0)
    assert verdict.accepted
    assert verdict.depth_m == pytest.approx(depth, abs=1e-6)


def test_depth_is_measured_from_the_recorded_surface_not_the_nominal_one():
    """Weather moves atmospheric pressure by more than a centimetre of water."""
    converter = make_converter()
    actual_surface = NOMINAL_ATMOSPHERIC_PA + 1500.0   # a high-pressure day
    assert converter.zero_at(actual_surface)

    verdict = converter.check(
        DepthSample(0.0, pressure_at(2.0, surface_pa=actual_surface)), now=0.0
    )
    assert verdict.depth_m == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------
# Surface zero discipline
# --------------------------------------------------------------------------


def test_publishing_is_refused_until_a_surface_zero_is_taken():
    converter = DepthConverter(limits=DepthLimits(require_surface_zero=True))
    verdict = converter.check(DepthSample(0.0, pressure_at(1.0)), now=0.0)
    assert not verdict.accepted
    assert "surface zero" in verdict.reason


def test_zeroing_underwater_is_refused():
    """A zero taken with the vehicle already under biases the whole run."""
    converter = make_converter()
    assert not converter.zero_at(pressure_at(2.0))
    assert not converter.zeroed


def test_zeroing_on_a_nonsense_reading_is_refused():
    converter = make_converter()
    assert not converter.zero_at(0.0)
    assert not converter.zero_at(float("nan"))


def test_zeroing_at_the_surface_is_accepted():
    converter = make_converter()
    assert converter.zero_at(NOMINAL_ATMOSPHERIC_PA + 200.0)
    assert converter.zeroed


def test_unzeroed_operation_is_degraded_when_explicitly_allowed():
    converter = make_converter(require_surface_zero=False)
    verdict = converter.check(DepthSample(0.0, pressure_at(1.0)), now=0.0)
    assert verdict.accepted
    assert "no surface zero" in verdict.degraded_reason


def test_reading_above_the_surface_reference_is_degraded():
    converter = make_converter()
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    verdict = converter.check(DepthSample(0.0, pressure_at(-1.0)), now=0.0)
    assert verdict.accepted
    assert "above the surface" in verdict.degraded_reason


# --------------------------------------------------------------------------
# Rejection rules
# --------------------------------------------------------------------------


def test_non_finite_pressure_is_rejected():
    converter = make_converter()
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    verdict = converter.check(DepthSample(0.0, float("nan")), now=0.0)
    assert not verdict.accepted
    assert "non-finite" in verdict.reason


def test_pressure_outside_the_plausible_band_is_rejected():
    """An unplugged sensor reads zero, not 'very shallow'."""
    converter = make_converter()
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    verdict = converter.check(DepthSample(0.0, 0.0), now=0.0)
    assert not verdict.accepted
    assert "plausible" in verdict.reason


def test_impossible_depth_rate_is_rejected():
    converter = make_converter(max_depth_rate_mps=3.0)
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    converter.check(DepthSample(0.0, pressure_at(1.0)), now=0.0)
    # 2 m in 50 ms is 40 m/s; the vehicle does not do that.
    verdict = converter.check(DepthSample(0.05, pressure_at(3.0)), now=0.05)
    assert not verdict.accepted
    assert "m/s limit" in verdict.reason


def test_depth_rate_check_is_skipped_across_a_long_gap():
    converter = make_converter(max_depth_rate_mps=3.0, max_dt_s=1.0)
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    converter.check(DepthSample(0.0, pressure_at(1.0)), now=0.0)
    verdict = converter.check(DepthSample(10.0, pressure_at(3.0)), now=10.0)
    assert verdict.accepted


def test_clock_skew_is_rejected():
    converter = make_converter(max_clock_skew_s=0.5)
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    verdict = converter.check(DepthSample(100.0, pressure_at(1.0)), now=0.0)
    assert not verdict.accepted
    assert "local clock" in verdict.reason


def test_a_rejected_sample_does_not_become_the_rate_reference():
    """Otherwise one glitch makes the next good reading look like a jump."""
    converter = make_converter(max_depth_rate_mps=3.0)
    converter.zero_at(NOMINAL_ATMOSPHERIC_PA)
    converter.check(DepthSample(0.0, pressure_at(1.0)), now=0.0)
    converter.check(DepthSample(0.05, pressure_at(3.0)), now=0.05)  # rejected

    verdict = converter.check(DepthSample(0.10, pressure_at(1.05)), now=0.10)
    assert verdict.accepted
