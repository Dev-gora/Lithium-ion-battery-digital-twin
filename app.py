"""Universal lithium-ion battery digital twin — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.theme import CSS, NARROW_CSS, RAIL_CSS  # noqa: E402
from dashboard.ui import ensure_profile, toggle_rail  # noqa: E402

st.set_page_config(
    page_title="Li-ion Battery Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.pages import (  # noqa: E402
    ageing,
    ai_prediction,
    assistant,
    configuration,
    dashboard,
    data_sources,
    dataset_explorer,
    ml_lab,
    research,
    simulator,
)

pages = [
    st.Page(dashboard.render, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    st.Page(configuration.render, title="Battery configuration", icon=":material/tune:", url_path="configuration"),
    st.Page(simulator.render, title="Battery simulator", icon=":material/bolt:", url_path="simulator"),
    st.Page(ai_prediction.render, title="AI prediction", icon=":material/psychology:", url_path="prediction"),
    st.Page(ageing.render, title="Battery ageing", icon=":material/trending_down:", url_path="ageing"),
    st.Page(data_sources.render, title="Data sources", icon=":material/table_chart:", url_path="sources"),
    st.Page(dataset_explorer.render, title="Dataset explorer", icon=":material/database:", url_path="datasets"),
    st.Page(ml_lab.render, title="AI / ML lab", icon=":material/science:", url_path="ml-lab"),
    st.Page(research.render, title="Research analytics", icon=":material/description:", url_path="research"),
    st.Page(assistant.render, title="Project assistant", icon=":material/forum:", url_path="assistant"),
]

rail = bool(st.session_state.get("sidebar_rail", False))
st.markdown(CSS + NARROW_CSS + (RAIL_CSS if rail else ""), unsafe_allow_html=True)

# The rail is the navigation, always. It is a plain CSS column rather than
# Streamlit's sidebar, whose collapse animation can freeze a background tab with the
# pages off-screen; the button underneath switches between names and icons only.
profile = ensure_profile()
with st.container(key="rail"):
    st.markdown(
        f"<div class='rail-id'><b>{profile.name}</b>"
        f"<small>{profile.cells_series}s{profile.cells_parallel}p · "
        f"{profile.pack_energy_Wh / 1000:.2f} kWh</small></div>",
        unsafe_allow_html=True,
    )
    for page in pages:
        st.page_link(page, label=page.title, icon=page.icon)
    st.button(
        "Icons only" if not rail else "Show names",
        icon=":material/keyboard_double_arrow_left:" if not rail else ":material/keyboard_double_arrow_right:",
        on_click=toggle_rail,
        key="rail_toggle",
        use_container_width=True,
    )

navigation = st.navigation(pages)
navigation.run()
