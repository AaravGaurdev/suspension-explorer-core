"""Browser interface for solar-car suspension load-path studies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from kinematics.core.enums import PointID
from kinematics.core.input import build_suspension
from kinematics.core.load_cases import (
    AxleLocation,
    Maneuver,
    VehicleParameters,
    calculate_corner_load,
    standard_maneuvers,
)
from kinematics.core.loads import AppliedLoad, solve_static_loads, upright_load_paths
from kinematics.core.primitives.point_ref import Side, point_key_name

st.set_page_config(
    page_title="Suspension force calculator",
    layout="centered",
)

ROOT = Path(__file__).parent
DEFAULT_GEOMETRY = ROOT / "examples" / "solar_car_double_wishbone.yaml"

st.markdown(
    """
    <style>
    html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp button,
    .stApp textarea, .stApp select, .stApp h1, .stApp h2, .stApp h3,
    .stApp th, .stApp td { font-family: Arial, Helvetica, sans-serif !important; }
    .stApp { color: #111; background: #fff; }
    .block-container { max-width: 900px; padding: 24px 28px 60px; }
    h1 { font-size: 28px !important; font-weight: 600 !important; }
    h2 { font-size: 20px !important; font-weight: 600 !important; margin-top: 28px !important; }
    h3 { font-size: 16px !important; font-weight: 600 !important; }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stFileUploaderDropzone"],
    button { border-radius: 2px !important; }
    div[data-testid="stFileUploaderDropzone"] {
        background: #fff !important;
        border: 1px solid #aaa !important;
        padding: 8px !important;
    }
    .stApp button { box-shadow: none !important; }
    div[data-testid="stMetric"] { background: none; border: 0; }
    header[data-testid="stHeader"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_geometry(name: str, raw: bytes):
    data = yaml.safe_load(raw.decode("utf-8"))
    suspension = build_suspension(data)
    if suspension.is_axle:
        raise ValueError(
            f"'{name}' is an axle model. This first web release accepts one corner "
            "at a time so chassis reactions are not silently mixed between sides."
        )
    return suspension


def _geometry_inputs() -> list[tuple[str, bytes]]:
    uploads = st.file_uploader(
        "Suspension geometry file (.yaml)",
        type=("yaml", "yml"),
        accept_multiple_files=True,
    )
    st.caption(
        "This file contains the XYZ coordinates of the wheel, joints, and chassis mounts. "
        "Upload more than one file to compare designs."
    )
    if uploads:
        return [(upload.name, upload.getvalue()) for upload in uploads]
    st.write("No file uploaded. Using the included double-wishbone example.")
    return [(DEFAULT_GEOMETRY.name, DEFAULT_GEOMETRY.read_bytes())]


def _vehicle_form() -> VehicleParameters:
    st.header("2. Enter vehicle measurements")
    mass = st.number_input("Total vehicle mass (kg)", 1.0, 2000.0, 300.0, 5.0)
    wheelbase = st.number_input("Wheelbase (mm)", 100.0, 10000.0, 2500.0, 10.0)
    cg_height = st.number_input(
        "Center-of-gravity height (mm)", 0.0, 3000.0, 450.0, 10.0
    )
    front_percent = st.number_input(
        "Weight on the front wheels when stationary (%)", 5.0, 95.0, 50.0, 1.0
    )
    front_track = st.number_input("Front track (mm)", 100.0, 5000.0, 1500.0, 10.0)
    rear_track = st.number_input("Rear track (mm)", 100.0, 5000.0, 1500.0, 10.0)
    return VehicleParameters(
        mass_kg=mass,
        wheelbase_mm=wheelbase,
        cg_height_mm=cg_height,
        front_static_fraction=front_percent / 100.0,
        front_track_mm=front_track,
        rear_track_mm=rear_track,
    )


def _maneuver_form() -> tuple[Maneuver, ...]:
    presets = {item.name: item for item in standard_maneuvers()}
    st.header("3. Choose driving conditions")
    st.write("The calculator runs every checked condition.")
    selected = [name for name in presets if st.checkbox(name, value=True)]
    st.subheader("Optional custom condition")
    add_custom = st.checkbox("Also run a custom condition")
    name = st.text_input("Condition name", "Custom condition")
    longitudinal_g = st.number_input(
        "Forward/back acceleration (g; negative means braking)",
        -5.0,
        5.0,
        0.0,
        0.1,
    )
    lateral_g = st.number_input(
        "Sideways acceleration (g; positive means a left turn)",
        -5.0,
        5.0,
        0.0,
        0.1,
    )
    bump = st.number_input(
        "Vertical wheel-load multiplier (1 = normal, 2 = twice normal)",
        0.0,
        10.0,
        1.0,
        0.1,
    )
    downforce = st.number_input(
        "Aerodynamic downward force on the whole car (N)",
        0.0,
        100000.0,
        0.0,
        10.0,
    )
    front_force_percent = st.number_input(
        "Braking or acceleration force handled by the front wheels (%)",
        0.0,
        100.0,
        50.0,
        5.0,
    )
    maneuvers = [presets[item] for item in selected]
    if add_custom:
        maneuvers.append(
            Maneuver(
                name=name,
                longitudinal_g=longitudinal_g,
                lateral_g=lateral_g,
                bump_factor=bump,
                downforce_n=downforce,
                front_longitudinal_force_fraction=front_force_percent / 100.0,
            )
        )
    return tuple(maneuvers)


def _solve_geometries(geometry_inputs, vehicle, maneuvers, axle, side):
    rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    errors: list[str] = []
    models: dict[str, object] = {}
    for geometry_name, raw in geometry_inputs:
        try:
            suspension = _load_geometry(geometry_name, raw)
            state = suspension.initial_state()
            paths = upright_load_paths(suspension, state)
            models[geometry_name] = (suspension, state, paths)
            for maneuver in maneuvers:
                corner = calculate_corner_load(vehicle, maneuver, axle, side)
                result = solve_static_loads(
                    state,
                    (AppliedLoad(PointID.WHEEL_CONTACT_CENTER, corner.force_n),),
                    paths,
                )
                force_residual = float(
                    sum(value * value for value in result.force_residual) ** 0.5
                )
                moment_residual = float(
                    sum(value * value for value in result.moment_residual) ** 0.5
                )
                cases.append(
                    {
                        "geometry": geometry_name,
                        "case": maneuver.name,
                        "corner_load": corner,
                        "result": result,
                        "state": state,
                        "paths": paths,
                        "valid": result.is_in_equilibrium,
                    }
                )
                for member in result.members:
                    rows.append(
                        {
                            "geometry": geometry_name,
                            "load_case": maneuver.name,
                            "member": member.path.label,
                            "axial_force_n": member.axial_force,
                            "state": "tension"
                            if member.axial_force >= 0
                            else "compression",
                            "pickup": point_key_name(member.path.pickup_point),
                            "pickup_fx_n": member.pickup_force[0],
                            "pickup_fy_n": member.pickup_force[1],
                            "pickup_fz_n": member.pickup_force[2],
                            "force_residual_n": force_residual,
                            "moment_residual_n_mm": moment_residual,
                            "rank": result.rank,
                            "condition_number": result.condition_number,
                            "equilibrium": result.is_in_equilibrium,
                        }
                    )
        except Exception as error:  # A bad upload must not take down other comparisons.
            errors.append(f"{geometry_name}: {error}")
    return pd.DataFrame(rows), cases, errors, models


def _force_figure(case: dict[str, object]) -> go.Figure:
    state = case["state"]
    result = case["result"]
    corner = case["corner_load"]
    fig = go.Figure()
    max_force = max((abs(member.axial_force) for member in result.members), default=1.0)
    for member in result.members:
        a = state.positions[member.path.body_point].data
        b = state.positions[member.path.pickup_point].data
        tension = member.axial_force >= 0
        color = "#1f77b4" if tension else "#d62728"
        width = 3.0 + 7.0 * abs(member.axial_force) / max(max_force, 1.0)
        fig.add_trace(
            go.Scatter3d(
                x=(a[0], b[0]),
                y=(a[1], b[1]),
                z=(a[2], b[2]),
                mode="lines+markers",
                line={
                    "width": width,
                    "color": color,
                },
                marker={"size": 4},
                name=f"{member.path.label}: {member.axial_force:+.0f} N",
                hovertemplate="%{fullData.name}<extra></extra>",
            )
        )
    contact_key = case["paths"][0].body_point
    if hasattr(state, "positions"):
        # The load is drawn from the model contact center when it exists.
        possible = [
            key
            for key in state.positions
            if point_key_name(key).endswith("wheel_contact_center")
        ]
        contact_key = possible[0] if possible else contact_key
    p = state.positions[contact_key].data
    force = corner.force_n
    scale = 180.0 / max((sum(value * value for value in force) ** 0.5), 1.0)
    fig.add_trace(
        go.Scatter3d(
            x=(p[0], p[0] + force[0] * scale),
            y=(p[1], p[1] + force[1] * scale),
            z=(p[2], p[2] + force[2] * scale),
            mode="lines+markers",
            line={"width": 9, "color": "#ff7f0e"},
            marker={"size": (4, 8)},
            name=f"Tire load: ({force[0]:.0f}, {force[1]:.0f}, {force[2]:.0f}) N",
        )
    )
    fig.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "X forward (mm)",
            "yaxis_title": "Y left (mm)",
            "zaxis_title": "Z up (mm)",
        },
        legend={"orientation": "h"},
    )
    return fig


def _html_report(vehicle, axle, side, table: pd.DataFrame) -> bytes:
    summary = table.groupby(["geometry", "member"], as_index=False).agg(
        peak_tension_n=("axial_force_n", "max"),
        peak_compression_n=("axial_force_n", "min"),
        all_cases_in_equilibrium=("equilibrium", "all"),
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>Suspension load report</title>
    <style>body{{font:14px Arial,sans-serif;max-width:1100px;margin:40px auto;color:#111}}
    table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #aaa;padding:7px;text-align:right}}
    th:first-child,td:first-child{{text-align:left}}</style>
    <h1>Suspension force report</h1>
    <p>Wheel calculated: {axle.value} {side.name.lower()}</p>
    <h2>Vehicle measurements</h2><pre>{asdict(vehicle)}</pre>
    <p>This calculation uses rigid links at one fixed suspension position. It does not calculate stress,
    fatigue, deformation, impact behavior, or FEA. Do not use rows that failed the force-balance check.</p>
    <h2>Highest and lowest link forces</h2>{summary.to_html(index=False, float_format=lambda x: f"{x:.2f}")}
    <h2>All results</h2>{table.to_html(index=False, float_format=lambda x: f"{x:.3f}")}
    """
    return html.encode("utf-8")


st.title("Suspension force calculator")
st.write(
    "Enter a suspension design and driving conditions. The calculator estimates the force "
    "in each suspension link and at each chassis mounting point."
)

st.header("1. Choose a suspension design")
geometry_inputs = _geometry_inputs()
st.write("Which wheel should be calculated?")
axle = AxleLocation(st.radio("Position", ("front", "rear")))
side = Side.LEFT if st.radio("Side", ("left", "right")) == "left" else Side.RIGHT
st.caption(
    "This matters because braking and turning put different loads on each wheel. "
    "Geometry coordinates use +X forward, +Y left, and +Z up."
)

vehicle = _vehicle_form()
maneuvers = _maneuver_form()

if not maneuvers:
    st.info("Check at least one driving condition.")
    st.stop()

table, cases, errors, _ = _solve_geometries(
    geometry_inputs, vehicle, maneuvers, axle, side
)
for error in errors:
    st.error(error)
if table.empty:
    st.stop()

valid_cases = sum(bool(case["valid"]) for case in cases)
peak_force = float(table["axial_force_n"].abs().max())
st.header("Results")
st.write(f"Suspension designs calculated: {table['geometry'].nunique()}")
st.write(f"Driving conditions calculated: {table['load_case'].nunique()}")
st.write(f"Largest force found in any suspension link: {peak_force:,.0f} N")
st.write(f"Calculations completed without a force-balance error: {valid_cases}/{len(cases)}")

if valid_cases != len(cases):
    st.error(
        "At least one calculation could not balance all forces and moments. The uploaded "
        "design may be missing a supporting link, or it may use a rocker that this version "
        "cannot calculate. Do not use results from those rows."
    )

st.header("Forces in the suspension links")
st.write("Positive values are tension (pulling). Negative values are compression (pushing).")
member_view = table[
    ["geometry", "load_case", "member", "axial_force_n", "state", "equilibrium"]
].copy()
member_view.columns = [
    "design",
    "driving condition",
    "suspension link",
    "force (N)",
    "pulling or pushing",
    "force balance passed",
]
st.dataframe(member_view, hide_index=True, width="stretch")

st.header("Forces at the chassis mounting points")
st.write("These are the X, Y, and Z forces the suspension applies to the chassis.")
pickup_view = table[
    [
        "geometry",
        "load_case",
        "pickup",
        "pickup_fx_n",
        "pickup_fy_n",
        "pickup_fz_n",
        "equilibrium",
    ]
].copy()
pickup_view.columns = [
    "design",
    "driving condition",
    "chassis mounting point",
    "forward/back force (N)",
    "left/right force (N)",
    "up/down force (N)",
    "force balance passed",
]
st.dataframe(pickup_view, hide_index=True, width="stretch")

st.header("Highest and lowest force in each suspension link")
st.write("Use this table to compare designs across all selected driving conditions.")
envelope = table.groupby(["geometry", "member"], as_index=False).agg(
    peak_tension_n=("axial_force_n", "max"),
    peak_compression_n=("axial_force_n", "min"),
    max_force_residual_n=("force_residual_n", "max"),
    all_cases_in_equilibrium=("equilibrium", "all"),
)
envelope.columns = [
    "design",
    "suspension link",
    "highest tension (N)",
    "highest compression (N)",
    "largest force-balance error (N)",
    "all calculations passed",
]
st.dataframe(envelope, hide_index=True, width="stretch")
st.download_button(
    "Download results CSV",
    table.to_csv(index=False).encode("utf-8"),
    "suspension-load-results.csv",
    "text/csv",
)
st.download_button(
    "Download HTML report",
    _html_report(vehicle, axle, side, table),
    "suspension-load-report.html",
    "text/html",
)

st.header("3D view")
labels = [f"{case['geometry']} — {case['case']}" for case in cases]
selected_label = st.selectbox("Result to display", labels)
selected_case = cases[labels.index(selected_label)]
st.plotly_chart(_force_figure(selected_case), width="stretch")

st.header("Limitations")
st.markdown(
    """
    - It calculates forces for one fixed suspension position. It does not simulate movement over time.
    - It treats every suspension link as perfectly rigid and connected by frictionless joints.
    - It does not calculate stress, bending, buckling, fatigue, deformation, or joint strength.
    - It supports links connected directly between the upright and chassis. Rockers, bell cranks,
      and other multi-part mechanisms are not supported yet.
    - If several links could share a load in different ways, the real split depends on part stiffness,
      which this calculator does not model.
    """
)
