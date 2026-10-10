"""Vehicle-level load cases for quasi-static corner-load estimates."""

from dataclasses import dataclass
from enum import StrEnum

from kinematics.core.primitives.point_ref import Side


class AxleLocation(StrEnum):
    """Vehicle axle selected for a corner calculation."""

    FRONT = "front"
    REAR = "rear"


@dataclass(frozen=True)
class VehicleParameters:
    """Inputs used by the intentionally simple load-transfer model."""

    mass_kg: float
    wheelbase_mm: float
    cg_height_mm: float
    front_static_fraction: float
    front_track_mm: float
    rear_track_mm: float
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("Vehicle mass must be positive.")
        if (
            self.wheelbase_mm <= 0.0
            or min(self.front_track_mm, self.rear_track_mm) <= 0
        ):
            raise ValueError("Wheelbase and track widths must be positive.")
        if self.cg_height_mm < 0.0:
            raise ValueError("CG height cannot be negative.")
        if not 0.0 < self.front_static_fraction < 1.0:
            raise ValueError("Front static fraction must be between zero and one.")


@dataclass(frozen=True)
class Maneuver:
    """One steady-state or factored event in vehicle coordinates."""

    name: str
    longitudinal_g: float = 0.0
    lateral_g: float = 0.0
    bump_factor: float = 1.0
    downforce_n: float = 0.0
    front_downforce_fraction: float = 0.5
    front_longitudinal_force_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.bump_factor < 0.0:
            raise ValueError("Bump factor cannot be negative.")
        for name, value in (
            ("front downforce fraction", self.front_downforce_fraction),
            (
                "front longitudinal-force fraction",
                self.front_longitudinal_force_fraction,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name.capitalize()} must be between zero and one.")


@dataclass(frozen=True)
class CornerLoad:
    """Ground force on one tire in ISO chassis axes, in newtons."""

    name: str
    axle: AxleLocation
    side: Side
    force_n: tuple[float, float, float]
    notes: tuple[str, ...]


def standard_maneuvers() -> tuple[Maneuver, ...]:
    """Return editable starting points rather than design-rule claims."""
    return (
        Maneuver("Static", bump_factor=1.0),
        Maneuver(
            "Hard braking", longitudinal_g=-0.8, front_longitudinal_force_fraction=0.7
        ),
        Maneuver(
            "Hard acceleration",
            longitudinal_g=0.5,
            front_longitudinal_force_fraction=0.1,
        ),
        Maneuver("Left corner", lateral_g=0.8),
        Maneuver("Right corner", lateral_g=-0.8),
        Maneuver(
            "Combined braking + left corner",
            longitudinal_g=-0.6,
            lateral_g=0.6,
            front_longitudinal_force_fraction=0.7,
        ),
        Maneuver("2.5g bump", bump_factor=2.5),
    )


def calculate_corner_load(
    vehicle: VehicleParameters,
    maneuver: Maneuver,
    axle: AxleLocation,
    side: Side,
) -> CornerLoad:
    """Estimate a corner tire load with transparent quasi-static assumptions.

    Longitudinal transfer is shared globally by wheelbase. Lateral transfer is
    allocated to each axle in proportion to its static axle load; this avoids
    pretending that unknown roll stiffness, roll-center, and unsprung-mass data
    are available. Bump factor multiplies vertical load at the selected corner.
    """
    if side is Side.CENTER:
        raise ValueError("A corner load requires left or right side.")
    weight = vehicle.mass_kg * vehicle.gravity_m_s2
    front_fraction = vehicle.front_static_fraction
    axle_fraction = (
        front_fraction if axle is AxleLocation.FRONT else 1.0 - front_fraction
    )

    # Positive longitudinal_g accelerates forward and transfers load rearward.
    longitudinal_transfer = (
        -vehicle.mass_kg
        * maneuver.longitudinal_g
        * vehicle.gravity_m_s2
        * vehicle.cg_height_mm
        / vehicle.wheelbase_mm
    )
    axle_vertical = weight * axle_fraction
    axle_vertical += (
        longitudinal_transfer if axle is AxleLocation.FRONT else -longitudinal_transfer
    )
    axle_downforce = maneuver.downforce_n * (
        maneuver.front_downforce_fraction
        if axle is AxleLocation.FRONT
        else 1.0 - maneuver.front_downforce_fraction
    )
    axle_vertical += axle_downforce

    track = (
        vehicle.front_track_mm if axle is AxleLocation.FRONT else vehicle.rear_track_mm
    )
    axle_lateral_force = (
        vehicle.mass_kg * maneuver.lateral_g * vehicle.gravity_m_s2 * axle_fraction
    )
    lateral_transfer_per_side = axle_lateral_force * vehicle.cg_height_mm / track
    # +Y acceleration loads the right/outside tire; -Y loads the left tire.
    outside_sign = -side.lateral_sign
    vertical = axle_vertical / 2.0 + outside_sign * lateral_transfer_per_side
    vertical *= maneuver.bump_factor

    total_longitudinal = (
        vehicle.mass_kg * maneuver.longitudinal_g * vehicle.gravity_m_s2
    )
    front_force = total_longitudinal * maneuver.front_longitudinal_force_fraction
    axle_longitudinal = (
        front_force if axle is AxleLocation.FRONT else total_longitudinal - front_force
    )
    longitudinal = axle_longitudinal / 2.0
    lateral = axle_lateral_force / 2.0

    notes = (
        "Lateral load transfer is distributed by static axle weight fraction.",
        (
            "No tire saturation, roll stiffness, roll-center, or unsprung-mass "
            "model is included."
        ),
        (
            "Bump factor is a user-selected design multiplier, not a transient "
            "impact simulation."
        ),
    )
    return CornerLoad(
        name=maneuver.name,
        axle=axle,
        side=side,
        force_n=(longitudinal, lateral, vertical),
        notes=notes,
    )
