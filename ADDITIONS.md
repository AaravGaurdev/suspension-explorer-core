# What this fork adds

The original project calculates suspension geometry and motion. This fork adds
force calculations and a simple browser interface for comparing designs.

## Force solver

- Accepts a force applied at the tire contact patch.
- Finds the axial force in each supported suspension link.
- Reports whether each link is in tension or compression.
- Converts link forces into X, Y, and Z forces at the chassis mounting points.
- Checks the remaining force and moment error so an unsupported or incomplete
  model is not presented as a valid answer.
- Reports matrix rank and condition number for diagnosing weak or ambiguous
  suspension layouts.
- Can automatically find direct links between the upright and chassis in a
  solved suspension model.

Core implementation: `src/kinematics/core/loads.py`

## Vehicle and driving-condition calculations

- Calculates the tire load for a selected front/rear and left/right wheel.
- Includes stationary weight, braking, acceleration, cornering, a vertical
  wheel-load multiplier, and aerodynamic downforce.
- Includes seven ready-made driving conditions and supports custom conditions.
- Accounts for front/rear weight distribution and basic longitudinal and
  lateral weight transfer.

Core implementation: `src/kinematics/core/load_cases.py`

## Browser interface

- Uses an included double-wishbone example when no file is uploaded.
- Accepts one or more geometry files for comparison.
- Shows forces in every suspension link.
- Shows X, Y, and Z loads at every chassis mounting point.
- Shows the highest tension and compression found across the selected driving
  conditions.
- Provides an interactive 3D force view.
- Exports the complete results as CSV or a standalone HTML report.
- Uses plain-language labels and displays failed force-balance checks.

Interface: `streamlit_app.py`

## Example and deployment files

- `examples/solar_car_double_wishbone.yaml` provides a working direct-link
  double-wishbone model.
- `requirements.txt` and the `web` dependency group in `pyproject.toml` support
  local use and Streamlit Community Cloud deployment.

## Tests

- Solver tests cover balanced force and moment solutions, tension/compression
  signs, multiple loads, and invalid inputs.
- Vehicle-load tests cover stationary, braking, cornering, bump, and mirrored
  left/right cases.
- The public-API test includes the two new core modules.

Tests: `tests/core/test_loads.py`, `tests/core/test_load_cases.py`, and
`tests/test_public_api.py`

## Current scope

This is not limited to solar cars. It can be used for a four-wheeled vehicle
when the uploaded suspension uses direct links from the upright to the chassis,
such as a conventional double wishbone with a directly mounted spring/damper.

The calculator currently solves one wheel assembly at a time. The user selects
front or rear and left or right because braking and cornering change the load at
each wheel differently. It does not yet solve rocker or bell-crank systems,
compliance, stress, fatigue, deformation, or transient impact behavior.
