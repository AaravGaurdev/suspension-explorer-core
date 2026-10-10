import pytest

from kinematics.core.load_cases import (
    AxleLocation,
    Maneuver,
    VehicleParameters,
    calculate_corner_load,
)
from kinematics.core.primitives.point_ref import Side


@pytest.fixture
def vehicle() -> VehicleParameters:
    return VehicleParameters(
        mass_kg=300.0,
        wheelbase_mm=2500.0,
        cg_height_mm=500.0,
        front_static_fraction=0.5,
        front_track_mm=1500.0,
        rear_track_mm=1500.0,
    )


def test_static_corner_is_one_quarter_vehicle_weight(
    vehicle: VehicleParameters,
) -> None:
    result = calculate_corner_load(
        vehicle, Maneuver("static"), AxleLocation.FRONT, Side.LEFT
    )
    assert result.force_n == pytest.approx((0.0, 0.0, 300.0 * 9.80665 / 4.0))


def test_braking_transfers_vertical_load_forward(vehicle: VehicleParameters) -> None:
    braking = Maneuver(
        "braking", longitudinal_g=-1.0, front_longitudinal_force_fraction=0.7
    )
    front = calculate_corner_load(vehicle, braking, AxleLocation.FRONT, Side.LEFT)
    rear = calculate_corner_load(vehicle, braking, AxleLocation.REAR, Side.LEFT)
    assert front.force_n[2] > rear.force_n[2]
    combined = 2.0 * (front.force_n[2] + rear.force_n[2])
    assert combined == pytest.approx(300.0 * 9.80665)


def test_left_corner_loads_right_tire(vehicle: VehicleParameters) -> None:
    corner = Maneuver("left", lateral_g=1.0)
    left = calculate_corner_load(vehicle, corner, AxleLocation.FRONT, Side.LEFT)
    right = calculate_corner_load(vehicle, corner, AxleLocation.FRONT, Side.RIGHT)
    assert right.force_n[2] > left.force_n[2]
    assert right.force_n[1] == pytest.approx(left.force_n[1])
