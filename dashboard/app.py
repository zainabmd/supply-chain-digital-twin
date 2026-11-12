"""
Supply-Chain Digital Twin: executive control tower (Streamlit entry point).

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.components.layout import render_sidebar  # noqa: E402
from dashboard.components.theme import inject_css, register_template  # noqa: E402

st.set_page_config(page_title="Supply-Chain Digital Twin", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
inject_css()
register_template()

V = Path(__file__).parent / "views"
nav = st.navigation({
    "Executive": [
        st.Page(V / "control_tower.py", title="Control Tower", icon="🧭", default=True),
        st.Page(V / "scenario_lab.py", title="Scenario Simulator", icon="🌪️"),
    ],
    "Network": [
        st.Page(V / "network_twin.py", title="Network Twin", icon="🕸️"),
        st.Page(V / "resilience.py", title="Resilience & SPOF", icon="🛡️"),
    ],
    "Planning": [
        st.Page(V / "forecast.py", title="Demand Forecast", icon="📈"),
        st.Page(V / "inventory_risk.py", title="Stockout Risk", icon="📦"),
        st.Page(V / "optimization.py", title="Inventory Optimisation", icon="⚙️"),
    ],
    "Operations": [
        st.Page(V / "pipeline_ops.py", title="Pipeline & Data Quality", icon="🛠️"),
    ],
})
with st.sidebar:
    st.markdown("<div style='display:flex;gap:10px;align-items:center;margin:2px 0 6px'><span style='font-size:1.6rem'>🧭</span>"
                "<div><div style='font-family:Source Serif 4,Georgia,serif;font-weight:700;color:#0b2545;font-size:1.05rem'>"
                "Control Tower</div><div style='font-size:.72rem;color:#868b97'>Supply-chain digital twin</div></div></div>",
                unsafe_allow_html=True)
st.session_state["_controls"] = render_sidebar()
nav.run()
