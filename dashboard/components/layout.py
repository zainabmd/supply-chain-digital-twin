"""Global sidebar (scenario simulator dials + multi-variable filters) and shared page chrome."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from dashboard.components.data import Filters, scenario_meta
from dashboard.components.theme import hero
from src.network import CATEGORIES, REGIONS
from src.scenarios import SCENARIOS, ScenarioParams


@dataclass(frozen=True)
class Controls:
    params: ScenarioParams
    policy: str
    filters: Filters


def _reset():
    for k in list(st.session_state):
        if k.startswith("sb_"):
            del st.session_state[k]


def render_sidebar() -> Controls:
    with st.sidebar:
        st.markdown("### 🌪️ Disruption scenario")
        names = [n for n, v in SCENARIOS.items() if not v.get("hidden")]
        scenario = st.selectbox("Scenario", names, index=0, key="sb_scenario",
                                format_func=lambda n: f"{SCENARIOS[n]['icon']}  {n}",
                                help="Applies to every page: routes, flows, stockout risk and KPIs are re-simulated.")
        is_base = scenario.startswith("Baseline")
        st.caption(SCENARIOS[scenario]["description"])
        severity = st.slider("Severity · capacity lost at affected nodes", 0, 100, 100, 5, key="sb_severity",
                             format="%d%%", disabled=is_base or scenario == "Demand surge",
                             help="≥95% closes the node; lower values throttle capacity and add congestion delay.")
        duration = st.slider("Duration (weeks)", 1, 8, 6, key="sb_duration", disabled=is_base,
                             help="Inside the 8-week risk outlook.")
        region = st.selectbox("Supplier region (regional outage)", REGIONS, index=0, key="sb_scn_region",
                              disabled=scenario != "Regional supplier outage")

        st.markdown("### ⏱️ Sensitivity")
        lt_mult = st.slider("Lead-time sensitivity", 0.8, 1.6, 1.0, 0.05, key="sb_lt", format="%.2f×",
                            help="Scale every transit time (carrier slack, congestion, buffer erosion).")
        default_shock = int(100 * SCENARIOS[scenario].get("default_shock", 0.0))
        shock = st.slider("Demand shock", -30, 60, default_shock, 5, key=f"sb_shock_{scenario}", format="%+d%%")
        policy = st.radio("Inventory policy", ["current", "optimal"], horizontal=True, key="sb_policy",
                          format_func=lambda p: "Current (z = 1.65)" if p == "current" else "Optimised (SLSQP)")

        st.markdown("### 🔎 Filters")
        sup_regions = st.multiselect("Supplier region", REGIONS, key="sb_sup_regions", placeholder="All regions")
        cats = st.multiselect("Product category", list(CATEGORIES), key="sb_cats", placeholder="All categories")
        dc_regions = st.multiselect("DC region", REGIONS, key="sb_dc_regions", placeholder="All DC regions")
        st.button("↺ Reset to baseline", on_click=_reset, width="stretch")

    params = ScenarioParams(scenario=scenario, severity=severity / 100, duration_weeks=duration,
                            lead_time_multiplier=round(lt_mult, 2), demand_shock=shock / 100, supplier_region=region)
    return Controls(params, policy, Filters(tuple(sup_regions), tuple(cats), tuple(dc_regions)))


def controls() -> Controls:
    c = st.session_state.get("_controls")
    return c if c is not None else Controls(ScenarioParams(), "current", Filters())


def page_hero(eyebrow: str, title: str, subtitle: str) -> None:
    c = controls()
    tags = [f"Scenario: {c.params.scenario}", f"Policy: {c.policy}"]
    if c.params.lead_time_multiplier != 1.0:
        tags.append(f"Lead time ×{c.params.lead_time_multiplier:.2f}")
    if c.params.demand_shock:
        tags.append(f"Demand {c.params.demand_shock:+.0%}")
    hero(eyebrow, title, subtitle, tags)


def scenario_banner(kpis: dict | None = None) -> None:
    c = controls()
    meta = scenario_meta(c.params.scenario)
    active = not c.params.scenario.startswith("Baseline") or c.params.demand_shock or c.params.lead_time_multiplier != 1
    extra = ""
    if kpis:
        extra = (f" · {kpis.get('lanes_rerouted', 0)} lanes rerouted · {kpis.get('lanes_unreachable', 0)} unreachable · "
                 f"avg lead time {kpis.get('avg_lead_time_base', 0):.1f} → {kpis.get('avg_lead_time_scenario', 0):.1f} days")
    sev = "" if c.params.scenario.startswith("Baseline") else f" · severity {c.params.severity:.0%} for {c.params.duration_weeks} wks"
    st.markdown(f'<div class="scn {"active" if active else ""}"><span class="ico">{meta["icon"]}</span><div><b>{c.params.scenario}</b>'
                f'<br><span>{meta["description"]}{sev}{extra}</span></div>'
                f'<span class="pill {"disrupt" if active else "base"}">{"WHAT-IF ACTIVE" if active else "BASELINE"}</span></div>',
                unsafe_allow_html=True)
