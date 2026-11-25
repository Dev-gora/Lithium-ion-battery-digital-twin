"""Shared visual language.

The interface reads as a laboratory test bench: a graphite instrument rail on the
left, a cool grey bench surface, and silkscreen-style condensed type for titles
with monospaced digits for anything measured.  Colour is restrained so the
traces themselves carry the page; crimson (the colour lithium burns in a flame
test) is reserved for the single most important number on a screen.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ---- tokens -----------------------------------------------------------------
INK = "#131C26"
MUTED = "#5A6A78"
GRID = "#DFE6EC"
HAIRLINE = "#C9D3DB"
SURFACE = "#FFFFFF"
CANVAS = "#E9EDF0"

ACCENT = "#A3123F"       # lithium-flame crimson: emphasis only
MEASURED = "#1B2B3A"     # experimental data
SIMULATED = "#B4690E"    # physics simulation
PREDICTED = "#0F6B63"    # machine-learning prediction
ALERT = "#A3123F"
NEUTRAL = "#93A3B1"

CHEM_COLORS = {"LCO": "#1B2B3A", "LFP": "#0F6B63", "NMC": "#B4690E", "NCA": "#5B3E7A"}

DATASET_COLORS = {
    "NASA": "#1B2B3A",
    "CALCE": "#B4690E",
    "Oxford": "#0F6B63",
    "Samsung": "#5B3E7A",
    "MIT": "#A3123F",
}

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif", size=12, color=INK),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    margin=dict(l=56, r=28, t=54, b=92),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=HAIRLINE, font_size=12,
                    font_family="'IBM Plex Mono', monospace"),
    legend=dict(orientation="h", yanchor="top", y=-0.24, x=0, bgcolor="rgba(0,0,0,0)",
                font=dict(size=10.5)),
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* Drafting paper: a faint 80 px grid under the working surface. */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #EAEEF1;
    background-image:
        linear-gradient(rgba(27,43,58,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(27,43,58,0.045) 1px, transparent 1px);
    background-size: 80px 80px;
}
.stApp, .stApp p, .stApp li, .stApp label {
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    color: #131C26;
}
.block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 1280px; }

h1, h2, h3, h4 {
    font-family: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important; letter-spacing: -0.005em;
}
h1 { font-size: 2.3rem !important; font-weight: 700 !important; line-height: 1.1;
     border-left: 5px solid #A3123F; padding-left: 0.7rem; margin-bottom: 0.4rem; }
h2 { font-size: 1.35rem !important; margin-top: 2.2rem; }
h3 { font-size: 1.05rem !important; }
[data-testid="stCaptionContainer"] p, .stApp .stCaption, small { color: #5A6A78 !important; }

/* ---- instrument rail ---------------------------------------------------- */
[data-testid="stSidebar"] { background: #131C26; border-right: 1px solid #0B1118; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] li {
    color: #AEBAC6 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F4F7F9 !important; border: none; padding-left: 0; font-size: 1rem !important; }
[data-testid="stSidebar"] hr { border-color: #2B3947; }
[data-testid="stSidebarNav"] a { border-radius: 0; padding-left: 0.4rem; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,0.06); }
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(163,18,63,0.20); box-shadow: inset 3px 0 0 #A3123F; }
[data-testid="stSidebarNav"] a[aria-current="page"] *,
[data-testid="stSidebarNavLink"][aria-current="page"] * { color: #FFFFFF !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] p { font-weight: 600; }
/* Navigation is a fixed column drawn by the app, not Streamlit's sidebar. Streamlit
   collapses its sidebar with a scripted Web Animation; a backgrounded tab can freeze
   that animation half-way, which leaves the sidebar off-screen with no way to change
   pages. This rail is plain CSS, so it is always where it should be. */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }

.st-key-rail {
    position: fixed; top: 0; bottom: 0; left: 0; width: 232px; z-index: 1000;
    background: #131C26; border-right: 1px solid #0B1118;
    padding: 3.0rem 0.6rem 0.9rem; overflow-y: auto; }
.rail-id { border-bottom: 1px solid #2B3947; padding: 0 0.2rem 0.7rem; margin-bottom: 0.5rem; }
.rail-id b { display: block; color: #F4F7F9; font-family: 'IBM Plex Sans Condensed', sans-serif;
             font-size: 0.86rem; font-weight: 600; line-height: 1.25; }
.rail-id small { display: block; color: #8496A6; font-family: 'IBM Plex Mono', monospace;
                 font-size: 0.66rem; letter-spacing: 0.02em; margin-top: 0.2rem; }
.st-key-rail [data-testid="stPageLink"] a {
    justify-content: flex-start; gap: 0.5rem; padding: 0.24rem 0.2rem;
    border-radius: 2px; text-decoration: none; }
.st-key-rail [data-testid="stPageLink"] a:hover { background: rgba(255,255,255,0.07); }
.st-key-rail [data-testid="stPageLink"] a[aria-current="page"] {
    background: rgba(163,18,63,0.22); box-shadow: inset 3px 0 0 #A3123F; }
.st-key-rail [data-testid="stIconMaterial"] { color: #AEBAC6; font-size: 18px; }
.st-key-rail [data-testid="stPageLink"] p {
    font-family: 'IBM Plex Sans Condensed', sans-serif; font-size: 0.82rem;
    color: #C7D2DB !important; }
.st-key-rail [data-testid="stPageLink"] a[aria-current="page"] p { color: #FFFFFF !important; }
.st-key-rail .stButton { margin-top: 0.7rem; }
.st-key-rail .stButton button {
    justify-content: flex-start; gap: 0.4rem; background: transparent;
    border: 1px solid #2B3947; color: #C7D2DB; }
.st-key-rail .stButton button p { color: #C7D2DB !important; font-size: 0.74rem; }

[data-testid="stMainBlockContainer"] { padding-left: 256px !important; }
[data-testid="stStatusWidget"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], [data-testid="stBaseButton-header"],
[data-testid="stMainMenu"], [data-testid="stMainMenuButton"] {
    display: none !important; }
[data-testid="stHeader"] { background: transparent; }

/* ---- readouts instead of metric cards ----------------------------------- */
div[data-testid="stMetric"] {
    background: transparent; border: none; border-left: 1px solid #C9D3DB;
    border-radius: 0; padding: 0.1rem 0 0.15rem 0.85rem; }
div[data-testid="stMetricLabel"] p {
    color: #5A6A78; font-size: 0.74rem; letter-spacing: 0.02em; }
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: 1.55rem;
    color: #131C26; letter-spacing: -0.03em; }
div[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; }

/* ---- type-set readout line --------------------------------------------- */
.readout {
    display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.55rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #5A6A78;
    padding: 0.5rem 0 0.55rem; border-bottom: 1px solid #C9D3DB; margin-bottom: 0.9rem; }
.readout b { color: #131C26; font-weight: 600; font-size: 0.86rem; }
.readout i { color: #B6C2CC; font-style: normal; }
.readout .swatch { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.note { color: #5A6A78; font-size: 0.82rem; }

/* ---- badges ------------------------------------------------------------- */
.badge { display: inline-block; padding: 2px 9px; border-radius: 2px;
         font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 500; }
.badge-measured { background: #E4E9EE; color: #1B2B3A; }
.badge-simulated { background: #F7E9D6; color: #8A5009; }
.badge-predicted { background: #DCEFEC; color: #0B544E; }
.badge-alert { background: #F6DEE4; color: #8A0F35; }

/* ---- tabs, buttons, containers ------------------------------------------ */
.stTabs [data-baseweb="tab-list"] { gap: 1.7rem; border-bottom: 1px solid #C9D3DB; }
.stTabs [data-baseweb="tab"] { background: transparent; padding: 0.35rem 0 0.55rem;
    font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 600; font-size: 1rem;
    color: #5A6A78; }
.stTabs [data-baseweb="tab"]:hover { color: #131C26; }
.stTabs [aria-selected="true"] { color: #131C26 !important; box-shadow: inset 0 -2px 0 #A3123F; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button, .stDownloadButton > button {
    border-radius: 2px; font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 600; letter-spacing: 0.01em; border-color: #C9D3DB; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: #A3123F; border: 1px solid #A3123F; color: #FFFFFF; }
.stButton > button[kind="primary"]:hover { background: #8A0F35; border-color: #8A0F35; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border: 1px solid #D3DBE2 !important; border-radius: 2px; }
[data-testid="stDataFrame"] { border: 1px solid #D3DBE2; border-radius: 2px; }
[data-testid="stAlert"] { border-radius: 2px; border-left: 3px solid #A3123F; }
[data-testid="stExpander"] details { border-radius: 2px; border-color: #D3DBE2; }
[data-testid="stDivider"] hr { border-color: #C9D3DB; }
a { color: #A3123F; }

/* ---- page guide --------------------------------------------------------- */
.guide {
    background: #FFFFFF; border: 1px solid #D3DBE2; border-left: 3px solid #A3123F;
    padding: 1.05rem 1.25rem 1.0rem; margin: 0.6rem 0 1.35rem; }
.guide-split { display: grid; grid-template-columns: 1fr 1.06fr; gap: 1.6rem;
               align-items: center; }
.guide-body { min-width: 0; }
.guide-kicker {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.09em;
    text-transform: uppercase; color: #A3123F; margin-bottom: 0.45rem; }
.guide-what { font-size: 0.95rem; line-height: 1.62; color: #1F2A35; margin: 0 0 0.75rem; }
.guide-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #5A6A78; margin: 0.2rem 0 0.35rem; }
.guide-how { list-style: none; margin: 0 0 0.6rem; padding: 0; }
.guide-how li { position: relative; padding-left: 1.1rem; font-size: 0.86rem;
                line-height: 1.55; color: #33414E; margin-bottom: 0.28rem; }
.guide-how li::before { content: '—'; position: absolute; left: 0; color: #A3123F; }
.guide-read { font-size: 0.8rem; line-height: 1.55; color: #5A6A78; margin: 0;
              border-top: 1px dotted #C9D3DB; padding-top: 0.55rem; }
.guide-read span {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem; letter-spacing: 0.09em;
    text-transform: uppercase; color: #131C26; margin-right: 0.5rem; }
.guide-art { text-align: center; padding-left: 1.3rem;
             border-left: 1px dotted #C9D3DB; }
.guide-art .diagram { max-height: 250px; }

.tabnote {
    font-size: 0.85rem; line-height: 1.6; color: #4A5866; margin: 0.35rem 0 1rem;
    padding: 0.5rem 0 0.5rem 0.8rem; border-left: 1px solid #C9D3DB; }
.tabnote span {
    display: block; font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem;
    letter-spacing: 0.09em; text-transform: uppercase; color: #A3123F; margin-bottom: 0.18rem; }

.figure {
    background: #FFFFFF; border: 1px solid #D3DBE2; padding: 0.9rem 1rem 0.55rem;
    margin: 0.4rem 0 1.2rem; }
.figure svg { width: 100%; height: auto; display: block; }
.figure-cap {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.07em;
    text-transform: uppercase; color: #5A6A78; border-top: 1px solid #E4E9EE;
    padding-top: 0.45rem; margin-top: 0.6rem; }
.diagram { width: 100%; height: auto; display: block; }
.diagram text { letter-spacing: 0.01em; }

/* Corner ticks on bordered panels — drawing convention, not a card shadow. */
[data-testid="stVerticalBlockBorderWrapper"] { position: relative; }
[data-testid="stVerticalBlockBorderWrapper"]::before,
[data-testid="stVerticalBlockBorderWrapper"]::after {
    content: ''; position: absolute; width: 7px; height: 7px; border: 0 solid #A3123F; }
[data-testid="stVerticalBlockBorderWrapper"]::before {
    top: -1px; left: -1px; border-top-width: 2px; border-left-width: 2px; }
[data-testid="stVerticalBlockBorderWrapper"]::after {
    bottom: -1px; right: -1px; border-bottom-width: 2px; border-right-width: 2px; }

/* ---- assistant ---------------------------------------------------------- */
[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid #D3DBE2; border-radius: 2px;
    padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #F2F5F7; border-color: #DCE3E8; }
[data-testid="stChatMessage"] p { font-size: 0.9rem; line-height: 1.62; }
[data-testid="stChatMessage"] table { font-size: 0.8rem; }
.example {
    border-left: 3px solid #B4690E; background: #FBF7F0; padding: 0.6rem 0.85rem;
    font-size: 0.86rem; line-height: 1.6; color: #3B342A; margin: 0.2rem 0 0.5rem; }
.example span {
    display: block; font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.1em; color: #8A5009; margin-bottom: 0.25rem; }
[data-testid="stChatInput"] textarea { font-size: 0.9rem; }

@media (max-width: 1100px) {
    .guide-split { grid-template-columns: 1fr; }
    .guide-art { order: -1; }
}
</style>
"""


RAIL_CSS = """
<style>
/* Compact rail: the same fixed column, icons only. Injected after NARROW_CSS so the
   compact width survives the narrow-window rules. */
.st-key-rail { width: 76px; padding: 3.0rem 0.35rem 0.9rem; }
.st-key-rail [data-testid="stPageLink"] a { justify-content: center; gap: 0; }
.st-key-rail [data-testid="stIconMaterial"] { font-size: 20px; }
.st-key-rail .rail-id { display: none; }
/* Page names stay available to screen readers, just not painted. */
.st-key-rail [data-testid="stPageLink"] p,
.st-key-rail .stButton button p {
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip: rect(0 0 0 0); white-space: nowrap; }
.st-key-rail .stButton button { justify-content: center; }
[data-testid="stMainBlockContainer"] { padding-left: 100px !important; }
</style>
"""

NARROW_CSS = """
<style>
/* Narrow windows keep the same rail, just slimmer with smaller names. */
@media (max-width: 760px) {
    .st-key-rail { width: 168px; padding: 3.0rem 0.45rem 0.85rem; }
    .st-key-rail [data-testid="stPageLink"] p { font-size: 0.76rem; }
    [data-testid="stMainBlockContainer"] { padding-left: 186px !important; }
}
@media (max-width: 520px) {
    .st-key-rail { width: 136px; padding: 3.0rem 0.35rem 0.85rem; }
    .st-key-rail [data-testid="stPageLink"] p { font-size: 0.7rem; }
    .st-key-rail .rail-id small { display: none; }
    [data-testid="stMainBlockContainer"] { padding-left: 152px !important; }
}
</style>
"""


def style(fig: go.Figure, title: str | None = None, height: int = 380, y_title: str | None = None,
          x_title: str | None = None) -> go.Figure:
    fig.update_layout(**LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(family="'IBM Plex Sans Condensed', sans-serif",
                                                           size=15, color=INK), x=0, xanchor="left"))
    # Update the primary axes through layout so a secondary y-axis is left alone.
    fig.update_layout(
        yaxis=dict(title_text=y_title, gridcolor=GRID, zeroline=False,
                   linecolor=HAIRLINE, ticks="outside", tickcolor=HAIRLINE),
        xaxis=dict(title_text=x_title, gridcolor=GRID, zeroline=False,
                   linecolor=HAIRLINE, ticks="outside", tickcolor=HAIRLINE),
    )
    return fig


def line(x, y, name: str, color: str = MEASURED, dash: str | None = None, width: float = 1.8):
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
    )


def badge(kind: str, label: str) -> str:
    return f"<span class='badge badge-{kind}'>{label}</span>"
