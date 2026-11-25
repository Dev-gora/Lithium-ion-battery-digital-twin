"""Render the dashboard to a static site that Netlify can serve.

Netlify serves files; Streamlit needs a Python process.  This module closes the gap
without forking the interface: it installs a small HTML backend in place of
``streamlit``, imports the very same page modules, calls each ``render()`` and
writes ``site/`` with the rail navigation, the Plotly charts, the tables and the
download files.

What carries over: every page, every chart (live Plotly, hover and zoom intact),
every table, the schematics, the assistant's knowledge-base answers and the
download buttons (as real files).

What does not: the widgets.  Selectboxes, sliders and buttons keep the values the
pages document as defaults, because a static file has no server to recompute with.
Each page says so once, at the top, instead of pretending the controls are live.

Rebuild with::

    python -m dashboard.static_export

then publish with ``netlify deploy --prod`` (``netlify.toml`` points at ``site/``).
"""

from __future__ import annotations

import html as _html
import re
import shutil
import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ASSETS = SITE / "assets"
DOWNLOADS = SITE / "downloads"

# (module, title, output file, section number) — same order as the Streamlit rail.
PAGES = [
    ("dashboard", "Dashboard", "dashboard", "01"),
    ("configuration", "Battery configuration", "configuration", "02"),
    ("simulator", "Battery simulator", "simulator", "03"),
    ("ai_prediction", "AI prediction", "prediction", "04"),
    ("ageing", "Battery ageing", "ageing", "05"),
    ("data_sources", "Data sources", "sources", "06"),
    ("dataset_explorer", "Dataset explorer", "datasets", "07"),
    ("ml_lab", "AI / ML lab", "ml-lab", "08"),
    ("research", "Research analytics", "research", "09"),
    ("assistant", "Project assistant", "assistant", "10"),
]


# ---- markdown ---------------------------------------------------------------

_TAG = re.compile(r"(<[^>]+>)")


def _inline(text: str) -> str:
    """Inline markdown, leaving raw HTML tags alone (the theme passes HTML through)."""
    out = []
    for part in _TAG.split(text):
        if part.startswith("<") and part.endswith(">"):
            out.append(part)
            continue
        piece = _html.escape(part, quote=False)
        piece = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", piece)
        piece = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", piece)
        piece = re.sub(r"`([^`]+)`", r"<code>\1</code>", piece)
        piece = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r"<a href='\2'>\1</a>", piece)
        out.append(piece)
    return "".join(out)


def _table(rows: list[str], start: int) -> tuple[str, int]:
    header = [c.strip() for c in rows[start].strip().strip("|").split("|")]
    body = []
    index = start + 2
    while index < len(rows) and rows[index].strip().startswith("|"):
        body.append([c.strip() for c in rows[index].strip().strip("|").split("|")])
        index += 1
    head = "".join(f"<th>{_inline(c)}</th>" for c in header)
    cells = "".join(
        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in body
    )
    return f"<table class='dt'><thead><tr>{head}</tr></thead><tbody>{cells}</tbody></table>", index


def md_to_html(text: str) -> str:
    """A compact CommonMark subset: paragraphs, lists, tables, quotes, code, headings."""
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            out.append(f"<pre class='code'><code>{_html.escape(chr(10).join(code))}</code></pre>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = min(3, len(heading.group(1)))
            out.append(f"<h{level + 1}>{_inline(heading.group(2))}</h{level + 1}>")
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(
            r"^\s*\|?[\s:|-]+\|", lines[index + 1]
        ):
            table_html, index = _table(lines, index)
            out.append(table_html)
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index]):
                items.append(_inline(re.sub(r"^\s*[-*]\s+", "", lines[index])))
                index += 1
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while index < len(lines) and re.match(r"^\s*\d+\.\s+", lines[index]):
                items.append(_inline(re.sub(r"^\s*\d+\.\s+", "", lines[index])))
                index += 1
            out.append("<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>")
            continue

        if stripped.startswith(">"):
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            out.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue

        if stripped.startswith("<"):
            block = []
            while index < len(lines) and lines[index].strip():
                block.append(lines[index])
                index += 1
            out.append("\n".join(block))
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^\s*([-*]\s|\d+\.\s|\||>|#|```|<)", lines[index]
        ):
            paragraph.append(lines[index].strip())
            index += 1
        out.append(f"<p>{_inline(' '.join(paragraph))}</p>")

    return "\n".join(out)


# ---- frames -----------------------------------------------------------------


class Frame:
    """A container whose children render in insertion order, after they are filled."""

    def __init__(self, page: "Page"):
        self.page = page
        self.parts: list[object] = []

    def add(self, chunk: object) -> None:
        self.parts.append(chunk)

    def inner(self) -> str:
        return "".join(p if isinstance(p, str) else p.html() for p in self.parts)

    def html(self) -> str:
        return self.inner()

    def __enter__(self):
        self.page.stack.append(self)
        return self

    def __exit__(self, *exc):
        self.page.stack.pop()
        return False

    # Column-level calls: Streamlit's column objects expose the same methods.
    def metric(self, label, value, delta=None, help=None, **kwargs):
        self.add(self.page.metric_html(label, value, delta, help))


class Column(Frame):
    def html(self) -> str:
        return f"<div class='col'>{self.inner()}</div>"


class Row(Frame):
    def __init__(self, page, columns, weights, gap):
        super().__init__(page)
        self.columns = columns
        self.weights = weights
        self.gap = gap or "medium"

    def html(self) -> str:
        tracks = " ".join(f"{w}fr" for w in self.weights)
        body = "".join(cell.html() for cell in self.columns)
        return f"<div class='row gap-{self.gap}' style='grid-template-columns:{tracks}'>{body}</div>"


class Panel(Frame):
    def __init__(self, page, border=False, key=None):
        super().__init__(page)
        self.border = border
        self.key = key

    def html(self) -> str:
        tid = f" id='{self.key}'" if self.key else ""
        if self.border:
            return f"<section class='panel'{tid}>{self.inner()}</section>"
        return f"<div class='stack'{tid}>{self.inner()}</div>"


class Expander(Frame):
    def __init__(self, page, label, expanded=False):
        super().__init__(page)
        self.label = label
        self.expanded = expanded

    def html(self) -> str:
        flag = " open" if self.expanded else ""
        return (
            f"<details class='expander'{flag}><summary>{_inline(str(self.label))}</summary>"
            f"<div class='expander-body'>{self.inner()}</div></details>"
        )


class ChatMessage(Frame):
    def __init__(self, page, role):
        super().__init__(page)
        self.role = role

    def html(self) -> str:
        who = "You" if self.role == "user" else "Assistant"
        return (
            f"<div class='bubble bubble-{self.role}'><div class='who'>{who}</div>"
            f"{self.inner()}</div>"
        )


class Tab(Frame):
    def __init__(self, page, group, index, label):
        super().__init__(page)
        self.group = group
        self.index = index
        self.label = label

    def html(self) -> str:
        return f"<section class='panel tab-panel'>{self.inner()}</section>"


class Tabs(Frame):
    def __init__(self, page, labels):
        super().__init__(page)
        self.group = f"t{page.counter()}"
        self.tabs = [Tab(page, self.group, i, label) for i, label in enumerate(labels)]
        self.parts = list(self.tabs)

    def html(self) -> str:
        checks = "".join(
            f"<input type='radio' name='{self.group}' id='{self.group}-{i}'"
            f"{' checked' if i == 0 else ''}>"
            for i in range(len(self.tabs))
        )
        bars = "".join(
            f"<label for='{self.group}-{i}'>{_inline(str(tab.label))}</label>"
            for i, tab in enumerate(self.tabs)
        )
        rules = []
        for i in range(len(self.tabs)):
            rules.append(
                f"#{self.group}-{i}:checked ~ .tab-panels > .tab-panel:nth-child({i + 1})"
                "{display:block}"
            )
            rules.append(
                f"#{self.group}-{i}:checked ~ .tabbar label[for='{self.group}-{i}']"
                "{color:#131C26;box-shadow:inset 0 -2px 0 #A3123F;background:#fff}"
            )
        panels = "".join(tab.html() for tab in self.tabs)
        return (
            f"<div class='tabs'>{checks}<div class='tabbar'>{bars}</div>"
            f"<style>{''.join(rules)}</style>"
            f"<div class='tab-panels'>{panels}</div></div>"
        )


class Page(Frame):
    def __init__(self, stub, title, slug, number):
        super().__init__(self)
        self.stub = stub
        self.title = title
        self.slug = slug
        self.number = number
        self.stack: list[Frame] = [self]
        self.extra_css: list[str] = []
        self.plotly_loaded = False
        self._counter = 0

    def add(self, chunk) -> None:
        self.stack[-1].parts.append(chunk)

    def counter(self) -> int:
        self._counter += 1
        return self._counter

    # -- blocks -------------------------------------------------------------
    def metric_html(self, label, value, delta=None, help=None) -> str:
        note = f"<div class='kpi-delta'>{_inline(str(delta))}</div>" if delta else ""
        hint = f" title='{_html.escape(str(help))}'" if help else ""
        return (
            f"<div class='kpi'{hint}><div class='kpi-label'>{_inline(str(label))}</div>"
            f"<div class='kpi-value'>{_html.escape(str(value))}</div>{note}</div>"
        )

    def control(self, label, value, kind="select", help=None, disabled=False) -> str:
        marks = " disabled" if disabled else ""
        hint = f"<div class='ctl-help'>{_inline(str(help))}</div>" if help else ""
        shown = _html.escape("" if value is None else str(value))
        if kind == "toggle":
            shown = "On" if value else "Off"
        return (
            f"<div class='ctl{marks}'><div class='ctl-label'>{_inline(str(label))}</div>"
            f"<div class='ctl-box'><span class='ctl-value'>{shown}</span>"
            f"<span class='ctl-fixed'>fixed</span></div>{hint}</div>"
        )

    def alert(self, kind, body) -> None:
        glyph = {"info": "i", "warning": "!", "error": "×", "success": "✓"}.get(kind, "i")
        self.add(
            f"<div class='alert alert-{kind}'><span class='alert-glyph'>{glyph}</span>"
            f"<div class='alert-body'>{md_to_html(str(body))}</div></div>"
        )

    def chart(self, fig, **kwargs) -> None:
        height = int(getattr(fig.layout, "height", None) or 400)
        include = False if self.plotly_loaded else "cdn"
        self.plotly_loaded = True
        html = fig.to_html(
            full_html=False,
            include_plotlyjs=include,
            config={"displayModeBar": False, "responsive": True, "displaylogo": False},
            default_width="100%",
            default_height=f"{height}px",
        )
        self.add(f"<div class='chart'>{html}</div>")

    def table(self, data, max_rows=200, hide_index=False, **kwargs) -> None:
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        total = len(frame)
        shown = frame.head(max_rows).copy()
        if hide_index or isinstance(frame.index, pd.RangeIndex):
            shown = shown.reset_index(drop=True)
            hide_index = True
        body = shown.to_html(
            classes="dt",
            border=0,
            index=not hide_index,
            escape=True,
            na_rep="—",
            float_format=lambda v: f"{v:,.4g}",
        )
        note = ""
        if total > max_rows:
            note = (
                f"<p class='cap'>Showing the first {max_rows:,} of {total:,} rows — "
                "the live app pages through all of them.</p>"
            )
        self.add(f"<div class='tablewrap'>{body}</div>{note}")

    def download(self, label, data, file_name=None, mime=None, **kwargs) -> None:
        name = Path(str(file_name or "download.bin")).name
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        target = DOWNLOADS / f"{self.slug}-{name}"
        payload = data.encode("utf-8") if isinstance(data, str) else data
        target.write_bytes(payload)
        size = target.stat().st_size
        unit = f"{size / 1_048_576:.1f} MB" if size > 1_048_576 else f"{size / 1024:.0f} KB"
        self.add(
            f"<a class='btn' href='downloads/{_html.escape(target.name)}' download>"
            f"{_inline(str(label))}<span class='btn-meta'>{unit}</span></a>"
        )

    def image(self, path, caption=None, **kwargs) -> None:
        source = Path(str(path))
        if not source.exists():
            return
        images = ASSETS / "img"
        images.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, images / source.name)
        cap = f"<div class='figure-cap'>{_inline(str(caption))}</div>" if caption else ""
        self.add(
            f"<figure class='figure'><img src='assets/img/{_html.escape(source.name)}' "
            f"alt='{_html.escape(source.stem)}'>{cap}</figure>"
        )


# ---- streamlit stand-in -----------------------------------------------------


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors Streamlit's error
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Spinner:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Rerun(Exception):
    pass


def _install_stub() -> types.ModuleType:
    """A ``streamlit`` module that renders HTML instead of driving a server."""
    stub = types.ModuleType("streamlit")
    stub.session_state = SessionState()
    pages = types.SimpleNamespace(current=None)

    def target() -> Page:
        return pages.current

    def cache_data(func=None, **_kwargs):
        def decorate(fn):
            return fn

        return decorate(func) if callable(func) else decorate

    def columns(spec, gap=None, **_kwargs):
        page = target()
        weights = list(spec) if isinstance(spec, (list, tuple)) else [1] * int(spec)
        cells = [Column(page) for _ in weights]
        page.add(Row(page, cells, weights, gap))
        return cells

    def container(*_args, border=False, key=None, **_kwargs):
        page = target()
        node = Panel(page, border=border, key=key)
        page.add(node)
        return node

    def expander(label, expanded=False, **_kwargs):
        page = target()
        node = Expander(page, label, expanded)
        page.add(node)
        return node

    def tabs(labels, **_kwargs):
        page = target()
        node = Tabs(page, labels)
        page.add(node)
        return node.tabs

    def chat_message(role="assistant", **_kwargs):
        page = target()
        node = ChatMessage(page, role)
        page.add(node)
        return node

    def markdown(text="", unsafe_allow_html=False, **_kwargs):
        target().add(md_to_html(str(text)))

    def caption(text="", **_kwargs):
        target().add(f"<p class='cap'>{_inline(str(text))}</p>")

    def title(text="", **_kwargs):
        target().add(f"<h1>{_inline(str(text))}</h1>")

    def subheader(text="", **_kwargs):
        target().add(f"<h2>{_inline(str(text))}</h2>")

    def header(text="", **_kwargs):
        target().add(f"<h2>{_inline(str(text))}</h2>")

    def divider(**_kwargs):
        target().add("<hr class='rule'>")

    def metric(label, value, delta=None, help=None, **_kwargs):
        target().add(target().metric_html(label, value, delta, help))

    def selectbox(label, options, index=0, help=None, disabled=False, **_kwargs):
        options = list(options)
        page = target()
        value = options[index] if options else None
        page.add(page.control(label, value, "select", help, disabled))
        return value

    def radio(label, options, index=0, help=None, disabled=False, **_kwargs):
        options = list(options)
        page = target()
        value = options[index] if options else None
        page.add(page.control(label, value, "radio", help, disabled))
        return value

    def multiselect(label, options, default=None, help=None, disabled=False, **_kwargs):
        page = target()
        chosen = list(default if default is not None else [])
        page.add(page.control(label, ", ".join(map(str, chosen)) or "none", "multi", help, disabled))
        return chosen

    def slider(label, min_value, max_value, value=None, step=None, help=None,
               disabled=False, **_kwargs):
        page = target()
        shown = min_value if value is None else value
        page.add(page.control(label, f"{shown}  ({min_value}–{max_value})", "slider", help, disabled))
        return shown

    def select_slider(label, options, value=None, help=None, disabled=False, **_kwargs):
        page = target()
        options = list(options)
        shown = value if value is not None else (options[0] if options else None)
        page.add(page.control(label, shown, "slider", help, disabled))
        return shown

    def number_input(label, min_value=None, max_value=None, value=None, step=None,
                     help=None, disabled=False, **_kwargs):
        page = target()
        page.add(page.control(label, value, "number", help, disabled))
        return value

    def toggle(label, value=False, help=None, disabled=False, **_kwargs):
        page = target()
        page.add(page.control(label, value, "toggle", help, disabled))
        return bool(value)

    def button(label, key=None, on_click=None, args=(), disabled=False, **_kwargs):
        page = target()
        # A few buttons produce an artifact that belongs in the static build (the PDF
        # report). Those are pressed once at export time instead of every visit.
        pressed = label in stub._press
        if pressed:
            stub._press.discard(label)
        page.add(
            f"<button class='btn btn-static' type='button' disabled>"
            f"{_inline(str(label))}</button>"
        )
        return pressed

    def download_button(label, data, file_name=None, mime=None, **_kwargs):
        target().download(label, data, file_name, mime)

    def plotly_chart(fig, **_kwargs):
        target().chart(fig)

    def dataframe(data, hide_index=False, **_kwargs):
        target().table(data, hide_index=hide_index, **_kwargs)

    def table(data, **_kwargs):
        target().table(data, **_kwargs)

    def image(path, caption=None, **_kwargs):
        target().image(path, caption)

    def json(body, expanded=True, **_kwargs):
        import json as _json

        text = _json.dumps(body, indent=2, default=str)
        target().add(f"<pre class='code json'>{_html.escape(text)}</pre>")

    def info(body, **_kwargs):
        target().alert("info", body)

    def warning(body, **_kwargs):
        target().alert("warning", body)

    def error(body, **_kwargs):
        target().alert("error", body)

    def success(body, **_kwargs):
        target().alert("success", body)

    def chat_input(_placeholder="", **_kwargs):
        page = target()
        page.add(
            "<div class='chatinput'><span>Ask about SOC, SOH, RUL, a dataset, the "
            "simulator, a page…</span><span class='ctl-fixed'>static</span></div>"
        )
        return None

    def text_input(label, value="", **_kwargs):
        page = target()
        page.add(page.control(label, value, "text"))
        return value

    def write(body="", **_kwargs):
        if isinstance(body, str):
            markdown(body)
        elif isinstance(body, (dict, list, tuple)):
            import json as _json

            target().add(
                f"<pre class='code'>{_html.escape(_json.dumps(body, indent=2, default=str))}</pre>"
            )
        else:
            target().add(f"<pre class='code'>{_html.escape(str(body))}</pre>")

    def set_page_config(**_kwargs):
        return None

    def spinner(*args, **kwargs):
        return _Spinner()

    def rerun():
        raise _Rerun()

    def stop():
        raise _Rerun()

    for fn in (cache_data, columns, container, expander, tabs, chat_message, markdown,
               caption, title, subheader, header, divider, metric, selectbox, radio,
               multiselect, slider, select_slider, number_input, toggle, button,
               download_button, plotly_chart, dataframe, table, image, json, info,
               warning, error, success, chat_input, text_input, write, set_page_config,
               spinner, rerun, stop):
        setattr(stub, fn.__name__, fn)

    stub.cache_resource = cache_data
    stub.cache = cache_data
    stub.__getattr__ = lambda name: (_ for _ in ()).throw(
        AttributeError(f"static_export: streamlit.{name} is not implemented")
    )
    stub._pages = pages
    stub._press = set()
    stub.__path__ = []  # a package, so submodule imports fail loudly instead of resolving
    sys.modules["streamlit"] = stub
    return stub


# ---- shell ------------------------------------------------------------------


def _rail(profile, active: str) -> str:
    links = []
    for module, title, slug, number in PAGES:
        cls = " class='active'" if slug == active else ""
        links.append(
            f"<a href='{slug}.html'{cls}><span class='num'>{number}</span>{_html.escape(title)}</a>"
        )
    return (
        "<nav class='rail' aria-label='Pages'>"
        f"<a class='rail-id' href='index.html'><b>{_html.escape(profile.name)}</b>"
        f"<small>{profile.cells_series}s{profile.cells_parallel}p · "
        f"{profile.pack_nominal_voltage_V:.1f} V · {profile.pack_energy_Wh / 1000:.2f} kWh</small></a>"
        f"<div class='rail-links'>{''.join(links)}</div>"
        "<div class='rail-foot'>Static build<br><span>Controls keep their documented "
        "defaults; the Streamlit app is the live one.</span></div>"
        "</nav>"
    )


FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' fill='#131C26'/>"
    "<rect x='8' y='6' width='16' height='20' rx='2' fill='none' stroke='#E9EDF0' stroke-width='2.5'/>"
    "<rect x='13' y='3' width='6' height='3' fill='#E9EDF0'/>"
    "<rect x='11' y='10' width='10' height='12' fill='#A3123F'/></svg>"
)


def _shell(*, title: str, content: str, rail: str, extra_css: str, stamp: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)} · Li-ion battery digital twin</title>
<meta name="description" content="State of charge, health and remaining useful life for lithium-ion cells: 67 audited laboratory batteries, a physics simulator and leave-one-battery-out validated models.">
<meta name="color-scheme" content="light">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="assets/style.css">
<script src="assets/app.js" defer></script>
{extra_css}
</head>
<body>
{rail}
<main class='content'>
<div class='staticbar'>Static build of the Streamlit dashboard · generated {stamp} ·
controls show their documented default values · the Streamlit app is the live one</div>
{content}
<footer class='foot'>Generated from <code>dashboard/</code> by <code>python -m dashboard.static_export</code>.
Numbers come from <code>experiments/results/metrics.json</code> and the audited datasets; nothing on this
site is invented for the web build.</footer>
</main>
</body>
</html>
"""


def _not_found(rail: str) -> str:
    return _shell(
        title="Page not found",
        rail=rail,
        extra_css="",
        stamp=date.today().isoformat(),
        content=(
            "<h1>Page not found</h1>"
            "<p class='cap'>That address is not part of the static build.</p>"
            "<p>Start from the <a href='index.html'>dashboard</a>.</p>"
        ),
    )


def build() -> int:
    stub = _install_stub()
    sys.path.insert(0, str(ROOT))

    from dashboard import ui  # noqa: E402  (imports the stub by design)
    from dashboard.pages import (  # noqa: E402
        ageing, ai_prediction, assistant, configuration, dashboard as home,
        data_sources, dataset_explorer, ml_lab, research, simulator,
    )

    modules = {
        "dashboard": home,
        "configuration": configuration,
        "simulator": simulator,
        "ai_prediction": ai_prediction,
        "ageing": ageing,
        "data_sources": data_sources,
        "dataset_explorer": dataset_explorer,
        "ml_lab": ml_lab,
        "research": research,
        "assistant": assistant,
    }

    if SITE.exists():
        shutil.rmtree(SITE)
    ASSETS.mkdir(parents=True)
    DOWNLOADS.mkdir(parents=True)
    shutil.copyfile(ROOT / "deploy" / "style.css", ASSETS / "style.css")
    shutil.copyfile(ROOT / "deploy" / "app.js", ASSETS / "app.js")
    (ASSETS / "favicon.svg").write_text(FAVICON, encoding="utf-8")

    stamp = date.today().isoformat()
    profile = ui.ensure_profile()
    written = 0

    for module_name, title, slug, number in PAGES:
        stub.session_state.clear()
        stub.session_state["profile"] = ui.ensure_profile()
        if module_name == "assistant":
            page_module = modules[module_name]
            stub.session_state["chat"] = [
                {"role": "user", "text": question, "entry": page_module._reply(question)}
                for question in assistant.SUGGESTIONS[:3]
            ]
        if module_name == "research":
            stub._press = {"Build PDF"}
        if module_name == "simulator":
            from simulator.profiles import run_discharge

            active = stub.session_state["profile"]
            result = run_discharge(
                active, "Constant C-rate", c_rate=0.5,
                power_W=active.pack_energy_Wh / 4, duration_s=4 * 3600, dt_s=15,
            )
            stub.session_state["sim_result"] = ("discharge", result.frame)

        page = Page(stub, title, slug, number)
        stub._pages.current = page
        try:
            modules[module_name].render()
        except _Rerun:
            pass
        rail = _rail(profile, slug)
        extra = "<style>" + "".join(page.extra_css) + "</style>" if page.extra_css else ""
        document = _shell(title=title, content=page.html(), rail=rail, extra_css=extra,
                          stamp=stamp)
        (SITE / f"{slug}.html").write_text(document, encoding="utf-8")
        if slug == "dashboard":
            (SITE / "index.html").write_text(document, encoding="utf-8")
        written += 1
        print(f"  {slug + '.html':<18} {len(document) / 1024:8.0f} KB")

    (SITE / "404.html").write_text(_not_found(_rail(profile, "")), encoding="utf-8")
    print(f"Static site written to {SITE} ({written} pages)")
    return written


if __name__ == "__main__":
    build()
