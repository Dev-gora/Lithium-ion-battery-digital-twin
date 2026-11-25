"""Page — project assistant: explains anything in the app, with examples."""

from __future__ import annotations

import streamlit as st

from dashboard import facts, knowledge
from dashboard.guide import guide
from dashboard.ui import sidebar_profile

SUGGESTIONS = [
    "What is a digital twin in this project?",
    "How is state of health predicted and how accurate is it?",
    "Which datasets are used and what are their limitations?",
    "Explain the simulator with an example.",
    "How accurate is the remaining-life prediction?",
]


def _reply(question: str) -> dict:
    entry = knowledge.answer(question)
    if entry is None:
        body = (
            "I do not have a prepared explanation for that wording. The assistant is retrieval "
            "over a curated knowledge base for this project — it will not invent an answer.\n\n"
            "Try one of these subjects: " + ", ".join(f"**{t}**" for t in knowledge.TITLE_INDEX[:8])
            + ", … or ask about a specific page, dataset or number you see on screen."
        )
        return {"title": "Not covered by the knowledge base", "body": body,
                "example": "", "page": "", "_related": []}
    # Worked examples quote live artifacts, never remembered figures.
    entry = dict(entry)
    try:
        entry["body"] = facts.fill(entry["body"])
        entry["example"] = facts.fill(entry.get("example", ""))
    except facts.PlaceholderError as exc:
        # Never ship a template with unresolved braces: stop and say so.
        return {
            "title": entry["title"],
            "body": "",
            "example": "",
            "page": "",
            "_related": [],
            "_error": (f"Knowledge-base entry `{entry.get('id', '?')}` could not be "
                       f"resolved from the project artifacts: {exc}. The answer is hidden "
                       "rather than shown with placeholder text — rebuild the missing "
                       "artifact (see Research analytics) and ask again."),
        }
    return entry


def _bubble(entry: dict, is_user: bool = False, text: str = "") -> None:
    if is_user:
        with st.chat_message("user"):
            st.markdown(text)
        return
    with st.chat_message("assistant"):
        st.markdown(f"**{entry['title']}**")
        if entry.get("_error"):
            st.error(entry["_error"])
            return
        st.markdown(entry["body"])
        if entry.get("example"):
            st.markdown("**Worked example**")
            st.markdown(
                f"<div class='example'><span>EXAMPLE</span>{entry['example']}</div>",
                unsafe_allow_html=True,
            )
        footer = []
        if entry.get("page"):
            footer.append(f"Check it on: {entry['page']}")
        if entry.get("_related"):
            footer.append("Related: " + " · ".join(entry["_related"]))
        if footer:
            st.caption(" — ".join(footer))


def _ask(question: str) -> None:
    st.session_state.setdefault("chat", [])
    st.session_state["chat"].append({"role": "user", "text": question})


def render() -> None:
    sidebar_profile()
    st.title("Project assistant")
    st.caption("Ask what anything in this app means, how it works, or where a number came from.")

    guide(
        "FIG. 11",
        "How this assistant answers",
        "It searches a knowledge base written from this project's own artifacts — the audited "
        "datasets, the trained models, `metrics.json`, the report files — and returns the matching "
        "explanation with a worked example and the page where you can verify it. It does not "
        "generate text freely, so it cannot invent a number.",
        [
            "Type a question in plain language, or press one of the example questions.",
            "Read the worked example underneath: it puts real project numbers into the definition.",
            "Follow the *Check it on* line to see the same figure on its page.",
        ],
        "Every answer says which artifact and page it comes from. If a question is not in the "
        "knowledge base, it says so instead of guessing.",
    )

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if not st.session_state["chat"]:
        st.markdown("<div class='guide-label'>Try one of these</div>", unsafe_allow_html=True)
        for index, question in enumerate(SUGGESTIONS):
            st.button(question, key=f"chip_{index}", on_click=_ask, args=(question,),
                      use_container_width=True)

    # Answer any turn that arrived without one (chip callbacks append first).
    for message in st.session_state["chat"]:
        if message["role"] == "user" and "entry" not in message:
            message["entry"] = _reply(message["text"])

    for message in st.session_state["chat"]:
        _bubble({}, True, message["text"])
        _bubble(message["entry"])

    prompt = st.chat_input("Ask about SOC, SOH, RUL, a dataset, the simulator, a page…")
    if prompt:
        _ask(prompt)
        st.rerun()

    if st.session_state["chat"]:
        st.divider()
        if st.button("Clear conversation"):
            st.session_state["chat"] = []
            st.rerun()
