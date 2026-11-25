"""Page and tab descriptions.

Every page opens with the same three questions answered, in the same order:
what this is, what you can do here, and how to read what you see.  The drawings
sit to the right of the text so the page is explained before it is used.
"""

from __future__ import annotations

import streamlit as st


def guide(fig: str, title: str, what: str, how: list[str], read: str | None = None,
          svg: str | None = None) -> None:
    steps = "".join(f"<li>{item}</li>" for item in how)
    read_html = f"<p class='guide-read'><span>How to read it</span>{read}</p>" if read else ""
    art = f"<div class='guide-art'>{svg}</div>" if svg else ""
    st.markdown(
        f"<div class='guide{' guide-split' if svg else ''}'>"
        f"<div class='guide-body'>"
        f"<div class='guide-kicker'>{fig} · {title}</div>"
        f"<p class='guide-what'>{what}</p>"
        f"<div class='guide-label'>What you can do here</div>"
        f"<ul class='guide-how'>{steps}</ul>"
        f"{read_html}"
        f"</div>{art}</div>",
        unsafe_allow_html=True,
    )


def tab_note(text: str, label: str = "What this tab shows") -> None:
    """One-line orientation at the top of a tab."""
    st.markdown(f"<p class='tabnote'><span>{label}</span>{text}</p>", unsafe_allow_html=True)


def figure(svg: str, caption: str) -> None:
    st.markdown(f"<div class='figure'>{svg}<div class='figure-cap'>{caption}</div></div>",
                unsafe_allow_html=True)


def quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())
