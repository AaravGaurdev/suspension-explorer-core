import numpy as np
import pytest

from kinematics.core.enums import PointID
from kinematics.core.loads import AppliedLoad, MemberLoadPath, solve_static_loads
from kinematics.core.primitives.geometry import Point3
from kinematics.core.state import SuspensionState


def test_solves_member_and_pickup_forces_for_balanced_basis() -> None:
    positions = {
        PointID.AXLE_INBOARD: Point3((0.0, 0.0, 0.0)),
        PointID.WHEEL_CONTACT_CENTER: Point3((0.0, 0.0, 0.0)),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3((1.0, 0.0, 0.0)),
        PointID.UPPER_WISHBONE_INBOARD_REAR: Point3((0.0, 1.0, 0.0)),
        PointID.LOWER_WISHBONE_INBOARD_FRONT: Point3((0.0, 0.0, 1.0)),
    }
    state = SuspensionState(
        positions=positions,
        free_points={PointID.AXLE_INBOARD, PointID.WHEEL_CONTACT_CENTER},
    )
    paths = tuple(
        MemberLoadPath(axis, PointID.AXLE_INBOARD, pickup)
        for axis, pickup in (
            ("x", PointID.UPPER_WISHBONE_INBOARD_FRONT),
            ("y", PointID.UPPER_WISHBONE_INBOARD_REAR),
            ("z", PointID.LOWER_WISHBONE_INBOARD_FRONT),
        )
    )

    result = solve_static_loads(
        state,
        (AppliedLoad(PointID.WHEEL_CONTACT_CENTER, (100.0, -200.0, 300.0)),),
        paths,
    )

    assert [member.axial_force for member in result.members] == pytest.approx(
        [-100.0, 200.0, -300.0]
    )
    assert sum((member.pickup_force for member in result.members), np.zeros(3)) == (
        pytest.approx((100.0, -200.0, 300.0))
    )
    assert result.is_in_equilibrium


def test_reports_moment_residual_for_incompatible_load_path() -> None:
    state = SuspensionState(
        positions={
            PointID.AXLE_INBOARD: Point3((0.0, 0.0, 0.0)),
            PointID.WHEEL_CONTACT_CENTER: Point3((0.0, 1.0, 0.0)),
            PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3((1.0, 0.0, 0.0)),
        },
        free_points={PointID.AXLE_INBOARD, PointID.WHEEL_CONTACT_CENTER},
    )

    result = solve_static_loads(
        state,
        (AppliedLoad(PointID.WHEEL_CONTACT_CENTER, (1.0, 0.0, 0.0)),),
        (
            MemberLoadPath(
                "link",
                PointID.AXLE_INBOARD,
                PointID.UPPER_WISHBONE_INBOARD_FRONT,
            ),
        ),
    )

    assert np.linalg.norm(result.moment_residual) > 0.1
    assert not result.is_in_equilibrium


def test_rejects_coincident_member_endpoints() -> None:
    state = SuspensionState(
        positions={
            PointID.AXLE_INBOARD: Point3((0.0, 0.0, 0.0)),
            PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3((0, 0, 0)),
        },
        free_points={PointID.AXLE_INBOARD},
    )
    with pytest.raises(ValueError, match="coincident endpoints"):
        solve_static_loads(
            state,
            (AppliedLoad(PointID.AXLE_INBOARD, (0.0, 0.0, -1.0)),),
            (
                MemberLoadPath(
                    "link",
                    PointID.AXLE_INBOARD,
                    PointID.UPPER_WISHBONE_INBOARD_FRONT,
                ),
            ),
        )
