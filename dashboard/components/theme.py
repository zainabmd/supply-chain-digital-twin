"""
"Executive Maritime" visual system for the supply-chain control tower.

A board-level planning tool calls for a formal, calm, print-friendly look:
warm paper background, deep navy ink and chrome, an ocean-teal brand accent,
generous whitespace, and colour reserved for meaning.

Data-colour rules (kept separate from the brand chrome):
* SEQUENTIAL  single-hue ocean-blue ramp for magnitude (heatmaps, map intensity)
* DIVERGING   blue (better) ↔ grey midpoint ↔ red (worse), for scenario deltas
* STATUS      good / warning / serious / critical: fixed hues, always shown with a label
* NODE TYPES  fixed categorical slots plus a distinct marker symbol per type
              (secondary encoding, so identity never depends on hue alone)
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- Brand chrome -----------------------------------------------------------
PAPER = "#f6f4ef"
SURFACE = "#ffffff"
BORDER = "#e4dfd3"
NAVY = "#0b2545"
NAVY_2 = "#13315c"
TEAL = "#0f766e"
SAND = "#c9b88a"
INK = "#14213d"
INK_2 = "#4a5468"
MUTED = "#868b97"
GRID = "#ece8de"

# ---- Data colours ----------------------------------------------------------------
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_SCALE = [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec835a"], [1.0, "#b2302f"]]
STATUS = {"Low": "#0ca30c", "Watch": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
STATUS_ORDER = ["Low", "Watch", "High", "Critical"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NODE_STYLE = {   # type: (label, colour, symbol)
    "supplier_t3": ("Tier-3 supplier", "#6250d6", "triangle-up"),
    "supplier_t2": ("Tier-2 supplier", "#2a78d6", "diamond"),
    "plant": ("Assembly plant", "#1baf7a", "square"),
    "port": ("Port", "#eb6834", "circle"),
    "chokepoint": ("Chokepoint", "#d03b3b", "x"),
    "waypoint": ("Sea waypoint", "#b8b4a8", "circle-open"),
    "dc": ("Distribution centre", NAVY, "star"),
}
CATEGORY_COLORS = {c: PALETTE[i] for i, c in enumerate(["Smartphones", "Laptops", "Wearables", "Smart Home", "Power & Batteries"])}
REGION_COLORS = {r: PALETTE[i] for i, r in enumerate(["Asia Pacific", "Europe", "North America", "Latin America",
                                                      "Middle East & Africa"])}


def register_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Source Sans 3', 'Segoe UI', system-ui, sans-serif", color=INK_2, size=12),
        title=dict(font=dict(color=INK, size=14, family="'Source Serif 4', Georgia, serif"), x=0, xanchor="left",
                   yref="container", y=1, yanchor="top", pad=dict(t=10)),
        colorway=PALETTE,
        xaxis=dict(showgrid=False, linecolor="#c3c2b7", ticks="outside", tickcolor="#c3c2b7", zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=NAVY, font=dict(color=INK, size=12)),
        geo=dict(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#ebe6da", showocean=True, oceancolor="#f3f6f9",
                 showcountries=True, countrycolor="#d7d0bf", showcoastlines=True, coastlinecolor="#cfc7b4",
                 showframe=False, projection_type="natural earth", showlakes=False),
        margin=dict(l=8, r=8, t=70, b=8),
        bargap=0.25,
    )
    pio.templates["executive"] = t
    pio.templates.default = "executive"


def style(fig: go.Figure, title: str | None = None, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(title_text=title, height=height, showlegend=legend,
                      margin=dict(t=(34 if title else 8) + (38 if legend else 6)))
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
:root {{ --paper:{PAPER}; --navy:{NAVY}; --teal:{TEAL}; --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --border:{BORDER}; --sand:{SAND}; }}
html, body, [data-testid="stAppViewContainer"] {{ font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif; }}
[data-testid="stAppViewContainer"] {{ background: var(--paper); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1480px; }}
h1, h2, h3 {{ font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #efebe1 0%, #e9e4d7 100%); border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] h3 {{ font-family: 'Source Sans 3', sans-serif; color: var(--navy); font-size: .74rem;
  letter-spacing: .14em; text-transform: uppercase; margin-top: 1rem; border-bottom: 1px solid #d8d1bf; padding-bottom: 4px; }}

/* ---- Hero band: navy, faint nautical-chart grid ---- */
.hero {{ position: relative; overflow: hidden; border-radius: 16px; padding: 22px 26px; margin-bottom: 14px; color: #fff;
  background:
    repeating-linear-gradient(0deg, rgba(255,255,255,.045) 0 1px, transparent 1px 44px),
    repeating-linear-gradient(90deg, rgba(255,255,255,.045) 0 1px, transparent 1px 44px),
    radial-gradient(700px 260px at 92% 0%, rgba(15,118,110,.55), transparent 70%),
    linear-gradient(120deg, {NAVY} 0%, {NAVY_2} 60%, #1d4e89 100%);
  box-shadow: 0 14px 34px rgba(11,37,69,.22); animation: rise .5s ease both; }}
.hero .eyebrow {{ font-size: .72rem; letter-spacing: .2em; text-transform: uppercase; color: var(--sand); font-weight: 700; }}
.hero h1 {{ color: #fff; margin: 4px 0 2px; font-size: 1.75rem; }}
.hero .sub {{ color: #c9d4e6; font-size: .9rem; }}
.hero .meta {{ position:absolute; right: 26px; top: 22px; display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; max-width: 55%; }}
.hero .tag {{ font-size: .72rem; padding: 4px 10px; border-radius: 999px; background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.22); color: #e8eef7; white-space: nowrap; }}
.hero .compass {{ position:absolute; right: -30px; bottom: -48px; font-size: 140px; opacity: .07; transform: rotate(-18deg); }}

/* ---- Scenario banner ---- */
.scn {{ display:flex; gap:12px; align-items:center; padding: 10px 16px; border-radius: 12px; margin: 0 0 12px;
  border: 1px solid var(--border); background: #fff; box-shadow: 0 2px 10px rgba(20,33,61,.05); }}
.scn.active {{ border-color: #f2c26b; background: linear-gradient(90deg, #fff7e6, #fff); }}
.scn .ico {{ font-size: 1.4rem; }} .scn b {{ color: var(--ink); }} .scn span {{ color: var(--ink2); font-size: .86rem; }}
.scn .pill {{ margin-left:auto; font-size:.72rem; font-weight:700; letter-spacing:.08em; padding:3px 10px; border-radius:999px; }}
.pill.base {{ background:#e7f4ee; color:#0b6b3a; border:1px solid #bfe3cf; }}
.pill.disrupt {{ background:#fff1d6; color:#8a5300; border:1px solid #f2c26b; }}

/* ---- KPI cards ---- */
.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; margin: 4px 0 10px; }}
.kpi {{ position:relative; background: var(--paper); background: #fff; border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 16px 12px 18px; box-shadow: 0 1px 2px rgba(20,33,61,.04); transition: transform .18s ease, box-shadow .18s ease;
  animation: rise .45s ease both; }}
.kpi:hover {{ transform: translateY(-2px); box-shadow: 0 12px 26px rgba(20,33,61,.10); }}
.kpi::before {{ content:''; position:absolute; left:0; top:12px; bottom:12px; width:4px; border-radius:0 4px 4px 0; background: var(--tone); }}
.kpi .lbl {{ font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color: var(--muted); font-weight:600; }}
.kpi .val {{ font-family:'Source Serif 4', Georgia, serif; font-size:1.65rem; color: var(--ink); font-weight:700; margin-top:2px; }}
.kpi .sub {{ font-size:.78rem; color: var(--ink2); margin-top:1px; }}
.chip {{ display:inline-block; font-size:.7rem; font-weight:700; padding:1px 7px; border-radius:6px; margin-left:4px; }}
.chip.good {{ background:#e3f4e3; color:#006300; }} .chip.bad {{ background:#fbe4e4; color:#9b1c1c; }} .chip.flat {{ background:#efede7; color:#5d5a52; }}

/* ---- Section titles ---- */
.sec {{ display:flex; align-items:baseline; gap:10px; margin: 20px 0 6px; }}
.sec h3 {{ margin:0; font-size: 1.15rem; }}
.sec .rule {{ flex:1; height:1px; background: linear-gradient(90deg, var(--border), transparent); }}
.sec .note {{ color: var(--muted); font-size: .8rem; }}

/* ---- Alerts list ---- */
.alert {{ display:flex; gap:10px; align-items:flex-start; padding:9px 12px; border-radius:10px; background:#fff;
  border:1px solid var(--border); margin-bottom:8px; transition: background .15s; }}
.alert:hover {{ background:#fbf8f1; }}
.alert .dot {{ width:9px; height:9px; border-radius:50%; margin-top:6px; flex: none; }}
.alert .t {{ font-weight:600; color: var(--ink); font-size:.88rem; }} .alert .d {{ color: var(--ink2); font-size:.8rem; }}
.status {{ font-size:.66rem; font-weight:700; letter-spacing:.06em; padding:1px 6px; border-radius:5px; color:#fff; }}

[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ background:#fff; border:1px solid var(--border); border-radius:14px;
  padding:6px; box-shadow: 0 1px 2px rgba(20,33,61,.04); animation: rise .4s ease both; }}
[data-testid="stExpander"] {{ background:#fff; border:1px solid var(--border); border-radius:12px; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--teal) !important; }}
.stTabs [aria-selected="true"] {{ color: var(--navy) !important; font-weight:700; }}
@keyframes rise {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


TONES = {"navy": NAVY, "teal": TEAL, "good": STATUS["Low"], "warn": STATUS["Watch"], "serious": STATUS["High"],
         "bad": STATUS["Critical"], "blue": "#2a78d6", "violet": "#6250d6", "sand": SAND}


def kpi(label: str, value: str, sub: str = "", tone: str = "navy", delta: str | None = None, delta_good: bool | None = None) -> str:
    chip = ""
    if delta:
        cls = "flat" if delta_good is None else ("good" if delta_good else "bad")
        chip = f'<span class="chip {cls}">{html.escape(delta)}</span>'
    return (f'<div class="kpi" style="--tone:{TONES.get(tone, tone)}"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div><div class="sub">{html.escape(sub)}{chip}</div></div>')


def kpis(cards: list[str]) -> None:
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str, tags: list[str]) -> None:
    tag_html = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>'
                f'<div class="sub">{html.escape(subtitle)}</div><div class="meta">{tag_html}</div>'
                f'<div class="compass">🧭</div></div>', unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="sec"><h3>{html.escape(title)}</h3><div class="rule"></div>'
                f'<span class="note">{html.escape(note)}</span></div>', unsafe_allow_html=True)


def alert(title: str, detail: str, status: str) -> str:
    c = STATUS.get(status, MUTED)
    return (f'<div class="alert"><span class="dot" style="background:{c}"></span><div><div class="t">{html.escape(title)} '
            f'<span class="status" style="background:{c}">{status.upper()}</span></div>'
            f'<div class="d">{html.escape(detail)}</div></div></div>')


def money(x: float) -> str:
    if x != x:
        return "–"
    for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            v = x / d
            return f"${v:,.{0 if abs(v) >= 100 else 1}f}{u}"
    return f"${x:,.0f}"


def num(x: float) -> str:
    if x != x:
        return "–"
    for u, d in (("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            return f"{x / d:,.1f}{u}"
    return f"{x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "–" if x != x else f"{100 * x:.{d}f}%"
