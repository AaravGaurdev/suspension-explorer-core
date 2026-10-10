"""Quasi-static load transfer through ideal two-force suspension members.

This module deliberately sits beside the kinematic solver.  It evaluates one
already-solved suspension state and does not add compliance, stress, inertia,
or transient dynamics to the geometry model.
"""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from kinematics.core.elements import (
    ElementType,
    RigidLinkElement,
    UprightElement,
    VariableLengthLinkElement,
)
from kinematics.core.primitives.geometry import Point3
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension

KeyT = TypeVar("KeyT", bound=PointKey)
VectorArray = NDArray[np.float64]


@dataclass(frozen=True)
class AppliedLoad:
    """A force and optional free moment applied at a point, in chassis axes.

    Forces are expressed in newtons and moments in newton-millimeters.  A tire
    force should normally be located at ``wheel_contact_center``.
    """

    point: PointKey
    force: tuple[float, float, float]
    moment: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MemberLoadPath:
    """One ideal axial member connecting the loaded body to a support."""

    label: str
    body_point: PointKey
    pickup_point: PointKey


@dataclass(frozen=True)
class MemberForce:
    """Solved axial member force and its endpoint forces.

    ``axial_force`` is positive in tension and negative in compression.
    ``body_force`` is the force exerted by the member on the loaded body;
    ``pickup_force`` is the force exerted by the member on the chassis pickup.
    """

    path: MemberLoadPath
    axial_force: float
    body_force: VectorArray
    pickup_force: VectorArray


@dataclass(frozen=True)
class StaticLoadResult:
    """Member forces and equilibrium diagnostics for one static load case."""

    members: tuple[MemberForce, ...]
    force_residual: VectorArray
    moment_residual: VectorArray
    rank: int
    condition_number: float

    @property
    def is_in_equilibrium(self) -> bool:
        """Whether residuals are small relative to the applied load."""
        return bool(
            np.linalg.norm(self.force_residual) <= 1e-6
            and np.linalg.norm(self.moment_residual) <= 1e-3
        )


def upright_load_paths(
    suspension: Suspension,
    state: SuspensionState[KeyT],
) -> tuple[MemberLoadPath, ...]:
    """Infer direct chassis-to-upright two-force members from an assembly.

    The helper supports conventional corners whose upright is restrained by
    direct rigid or variable-length links.  Rockers, trailing arms, and other
    intermediate rigid bodies require explicitly authored load paths or a
    future multi-body equilibrium model.
    """
    uprights = [
        element
        for element in suspension.elements()
        if isinstance(element, UprightElement)
    ]
    if len(uprights) != 1:
        raise ValueError(
            "Automatic load-path discovery requires exactly one upright element."
        )
    upright_points = set(uprights[0].point_keys)
    fixed_points = state.fixed_points
    paths: list[MemberLoadPath] = []
    for element in suspension.elements():
        if not isinstance(element, RigidLinkElement | VariableLengthLinkElement):
            continue
        endpoints = (element.point_a, element.point_b)
        body_endpoints = [point for point in endpoints if point in upright_points]
        if len(body_endpoints) != 1:
            continue
        body_point = body_endpoints[0]
        pickup_point = endpoints[1] if endpoints[0] == body_point else endpoints[0]
        is_chassis_steering_pickup = (
            isinstance(element, RigidLinkElement)
            and element.type in {ElementType.TRACK_ROD, ElementType.TOE_LINK}
        )
        if pickup_point not in fixed_points and not is_chassis_steering_pickup:
            continue
        paths.append(
            MemberLoadPath(
                label=element.label,
                body_point=body_point,
                pickup_point=pickup_point,
            )
        )
    if not paths:
        raise ValueError("No direct chassis-to-upright two-force members were found.")
    return tuple(paths)


def solve_static_loads(
    state: SuspensionState[KeyT],
    loads: tuple[AppliedLoad, ...],
    load_paths: tuple[MemberLoadPath, ...],
    *,
    reference_point: Point3 | None = None,
) -> StaticLoadResult:
    """Solve rigid-body equilibrium for axial member and chassis pickup forces.

    The least-squares solution is exact for a statically determinate compatible
    layout.  For an indeterminate layout it is the minimum-norm force split;
    stiffness is required to determine the physical split.  Residuals expose
    incompatible or under-constrained load cases rather than hiding them.
    """
    if not loads:
        raise ValueError("At least one applied load is required.")
    if not load_paths:
        raise ValueError("At least one member load path is required.")
    for load in loads:
        if load.point not in state.positions:
            raise KeyError(f"Applied-load point is absent from the state: {load.point}")

    origin = (
        reference_point.data
        if reference_point is not None
        else state.positions[load_paths[0].body_point].data
    )
    body_force = np.zeros(3, dtype=np.float64)
    body_moment = np.zeros(3, dtype=np.float64)
    for load in loads:
        force = _vector(load.force, "force")
        moment = _vector(load.moment, "moment")
        body_force += force
        offset = state.positions[load.point].data - origin
        body_moment += np.cross(offset, force) + moment

    force_columns: list[VectorArray] = []
    moment_columns: list[VectorArray] = []
    for path in load_paths:
        body = state.positions[path.body_point].data
        pickup = state.positions[path.pickup_point].data
        delta = pickup - body
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            raise ValueError(f"Load path '{path.label}' has coincident endpoints.")
        direction = delta / length
        force_columns.append(direction)
        moment_columns.append(np.cross(body - origin, direction))

    body_offsets = (
        float(np.linalg.norm(state.positions[path.body_point].data - origin))
        for path in load_paths
    )
    load_offsets = (
        float(np.linalg.norm(state.positions[load.point].data - origin))
        for load in loads
    )
    characteristic_length = max(1.0, *body_offsets, *load_offsets)
    force_matrix = np.column_stack(force_columns)
    moment_matrix = np.column_stack(moment_columns)
    scaled_matrix = np.vstack((force_matrix, moment_matrix / characteristic_length))
    scaled_rhs = -np.concatenate((body_force, body_moment / characteristic_length))
    axial_forces, _, rank, singular_values = np.linalg.lstsq(
        scaled_matrix, scaled_rhs, rcond=None
    )

    members = tuple(
        MemberForce(
            path=path,
            axial_force=float(axial_force),
            body_force=direction * axial_force,
            pickup_force=-direction * axial_force,
        )
        for path, direction, axial_force in zip(
            load_paths, force_columns, axial_forces, strict=True
        )
    )
    force_residual = force_matrix @ axial_forces + body_force
    moment_residual = moment_matrix @ axial_forces + body_moment
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return StaticLoadResult(
        members=members,
        force_residual=force_residual,
        moment_residual=moment_residual,
        rank=int(rank),
        condition_number=condition_number,
    )


def _vector(values: tuple[float, float, float], name: str) -> VectorArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Applied-load {name} must contain three finite values.")
    return array
