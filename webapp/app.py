"""Streamlit Community Cloud entry point."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parents[1] / "streamlit_app.py", run_name="__main__")
