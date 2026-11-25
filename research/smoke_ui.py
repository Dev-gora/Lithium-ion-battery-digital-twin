"""Render every dashboard page headlessly and report any exception.

Usage:  python research/smoke_ui.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "dashboard",
    "configuration",
    "simulator",
    "ai_prediction",
    "ageing",
    "data_sources",
    "dataset_explorer",
    "ml_lab",
    "research",
    "assistant",
]

SCRIPT = """
import sys
sys.path.insert(0, r"{root}")
from dashboard.pages import {page}
{page}.render()
"""


def main() -> int:
    failures = 0
    for page in PAGES:
        at = AppTest.from_string(SCRIPT.format(root=ROOT, page=page), default_timeout=300)
        try:
            at.run()
        except Exception:
            failures += 1
            print(f"FAIL {page} (harness)")
            traceback.print_exc()
            continue
        if at.exception:
            failures += 1
            print(f"FAIL {page}")
            for exc in at.exception:
                print("    ", exc.message)
            continue
        elements = (len(at.markdown) + len(at.metric) + len(at.dataframe)
                    + len(at.selectbox) + len(at.tabs))
        print(f"ok   {page}: {len(at.title)} title, {elements} widgets")
    print(f"\n{failures} failing page(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
