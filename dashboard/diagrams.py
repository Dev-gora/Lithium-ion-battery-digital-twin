"""Technical line-art schematics.

Every drawing here is a single-weight outline on white — no fills, no gradients,
no clip art — so the app reads like an engineering drawing rather than a slide
deck.  Functions return SVG strings for ``st.markdown``.
"""

from __future__ import annotations

INK = "#1B2B3A"
FINE = "#B6C2CC"
MUTED = "#5A6A78"
ACCENT = "#A3123F"
MONO = "IBM Plex Mono, monospace"


def _svg(body: str, w: int, h: int) -> str:
    return (
        f"<svg class='diagram' viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg' "
        f"preserveAspectRatio='xMidYMid meet' role='img'>{body}</svg>"
    )


def _t(x: float, y: float, text: str, anchor: str = "start", size: float = 9.5,
       color: str = MUTED, weight: int = 400) -> str:
    # The plates are drawn on a 700-unit canvas and rendered around 500 px wide, so
    # every label is scaled up to stay legible once the viewBox is fitted.
    size = round(size * 1.24, 1)
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' font-family='{MONO}' font-size='{size}' "
        f"fill='{color}' text-anchor='{anchor}' font-weight='{weight}'>{text}</text>"
    )


def _leader(x1: float, y1: float, x2: float, y2: float, color: str = FINE) -> str:
    return (f"<path d='M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}' stroke='{color}' "
            f"stroke-width='1' fill='none'/>")


def _hatch(x: float, y: float, w: float, h: float, step: float = 9) -> str:
    parts = []
    lx = x + 5
    while lx < x + w - 3:
        parts.append(f"<path d='M{lx:.1f},{y + h - 1:.1f} L{min(lx + h - 3, x + w - 1):.1f},"
                     f"{y + 1:.1f}' stroke='{FINE}' stroke-width='0.8'/>")
        lx += step
    return "".join(parts)


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str = INK) -> str:
    return (
        f"<path d='M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}' stroke='{color}' stroke-width='1.2'/>"
        f"<path d='M{x2 - 7:.1f},{y2 - 4:.1f} L{x2:.1f},{y2:.1f} L{x2 - 7:.1f},{y2 + 4:.1f}' "
        f"stroke='{color}' stroke-width='1.2' fill='none'/>"
    )


def _battery_glyph(cx: float, top: float, w: float = 22, h: float = 34) -> str:
    """A cell symbol: can with a raised terminal nub."""
    return (
        f"<rect x='{cx - w / 2:.1f}' y='{top:.1f}' width='{w}' height='{h}' rx='2.5' "
        f"fill='#FFFFFF' stroke='{INK}' stroke-width='1.3'/>"
        f"<rect x='{cx - 5:.1f}' y='{top - 4:.1f}' width='10' height='4' fill='#FFFFFF' "
        f"stroke='{INK}' stroke-width='1.3'/>"
        f"<path d='M{cx - w / 2 + 4:.1f},{top + h / 2:.1f} L{cx + w / 2 - 4:.1f},{top + h / 2:.1f}' "
        f"stroke='{FINE}' stroke-width='1'/>"
    )


# --------------------------------------------------------------------------- cell
def cell_cutaway() -> str:
    """Layered cell construction, cut away to show the electrode stack."""
    p = []
    p.append(f"<rect x='110' y='76' width='300' height='150' rx='9' fill='#FFFFFF' "
             f"stroke='{INK}' stroke-width='1.6'/>")

    layers = [
        ("Al collector · +", False),
        ("Cathode · NMC / LFP / LCO", True),
        ("Separator · PE / PP", None),
        ("Anode · graphite", True),
        ("Cu collector · −", False),
    ]
    y = 92
    for index, (label, hatch) in enumerate(layers):
        if hatch is True:
            p.append(f"<rect x='122' y='{y}' width='276' height='18' fill='#FFFFFF' "
                     f"stroke='{INK}' stroke-width='1'/>")
            p.append(_hatch(122, y, 276, 18))
        elif hatch is None:
            p.append(f"<rect x='122' y='{y}' width='276' height='18' fill='none' "
                     f"stroke='{FINE}' stroke-width='1' stroke-dasharray='2 3'/>")
        else:
            p.append(f"<rect x='122' y='{y}' width='276' height='18' fill='#FFFFFF' "
                     f"stroke='{INK}' stroke-width='1'/>")
            p.append(f"<path d='M122,{y + 9} L398,{y + 9}' stroke='{FINE}' stroke-width='0.8'/>")
        p.append(_leader(398, y + 9, 432, y + 9))
        p.append(_t(440, y + 12, label))
        y += 22

    # tabs and terminals
    p.append(f"<rect x='150' y='58' width='30' height='18' fill='#FFFFFF' stroke='{ACCENT}' "
             f"stroke-width='1.4'/>")
    p.append(f"<rect x='330' y='58' width='30' height='18' fill='#FFFFFF' stroke='{INK}' "
             f"stroke-width='1.4'/>")
    p.append(_t(165, 50, "+ tab", "middle", 9, ACCENT))
    p.append(_t(345, 50, "− tab", "middle"))

    # overall dimension
    p.append(f"<path d='M110,250 L410,250' stroke='{FINE}' stroke-width='1'/>")
    p.append(f"<path d='M110,244 L110,256 M410,244 L410,256' stroke='{FINE}' stroke-width='1'/>")
    p.append(_t(260, 266, "cell length — form-factor dependent", "middle", 8.5))
    p.append(_t(110, 288, "cross-section is schematic; layer count and thickness vary by chemistry"))
    return _svg("".join(p), 700, 300)


# --------------------------------------------------------------------------- pack
def pack_topology(series: int = 4, parallel: int = 3) -> str:
    """Series/parallel lattice with busbars, terminals and a BMS tap."""
    s, par = max(1, min(int(series), 4)), max(1, min(int(parallel), 4))
    p = []
    bus_top, bus_bot = 76.0, 286.0
    bus_left, bus_right = 108.0, 560.0
    # Spread the parallel columns across the busbar instead of bunching them left.
    if par == 1:
        cols = [(bus_left + bus_right) / 2]
    else:
        span = 330.0
        cols = [bus_left + 52 + i * span / (par - 1) for i in range(par)]
    rows = [94.0 + i * 46 for i in range(s)]

    p.append(f"<path d='M{bus_left},{bus_top} L{bus_right},{bus_top}' stroke='{INK}' "
             f"stroke-width='2' fill='none'/>")
    p.append(f"<path d='M{bus_left},{bus_bot} L{bus_right},{bus_bot}' stroke='{INK}' "
             f"stroke-width='2' fill='none'/>")
    p.append(_t(bus_right - 4, bus_top - 9, "+ bus", "end"))
    p.append(_t(bus_right - 4, bus_bot + 17, "− bus", "end"))

    for cx in cols:
        p.append(f"<path d='M{cx:.1f},{bus_top} L{cx:.1f},{rows[0]:.1f}' stroke='{FINE}'/>")
        p.append(_battery_glyph(cx, rows[0]))
        for upper, lower in zip(rows, rows[1:]):
            p.append(f"<path d='M{cx:.1f},{upper + 34:.1f} L{cx:.1f},{lower:.1f}' "
                     f"stroke='{FINE}'/>")
            p.append(_battery_glyph(cx, lower))
        p.append(f"<path d='M{cx:.1f},{rows[-1] + 34:.1f} L{cx:.1f},{bus_bot}' stroke='{FINE}'/>")

    # terminals
    p.append(f"<path d='M{bus_left},{bus_top} L88,{bus_top} L88,58' stroke='{INK}' "
             f"stroke-width='1.2' fill='none'/>")
    p.append(f"<circle cx='88' cy='52' r='4' fill='#FFFFFF' stroke='{ACCENT}' stroke-width='1.4'/>")
    p.append(_t(88, 38, "+", "middle", 12, ACCENT, 600))
    p.append(f"<path d='M{bus_left},{bus_bot} L88,{bus_bot} L88,304' stroke='{INK}' "
             f"stroke-width='1.2' fill='none'/>")
    p.append(f"<circle cx='88' cy='310' r='4' fill='#FFFFFF' stroke='{INK}' stroke-width='1.4'/>")
    p.append(_t(88, 330, "−", "middle", 12, INK, 600))

    # BMS tap, parked just beyond the end of the busbars
    p.append(f"<rect x='578' y='166' width='76' height='48' rx='3' fill='#FFFFFF' "
             f"stroke='{INK}' stroke-width='1.4'/>")
    p.append(_t(616, 188, "BMS", "middle", 10, INK, 600))
    p.append(_t(616, 202, "sense", "middle", 8))
    p.append(f"<path d='M{bus_right},{bus_top} L616,{bus_top} L616,166' stroke='{FINE}' "
             f"stroke-dasharray='3 3' fill='none'/>")
    p.append(f"<path d='M{bus_right},{bus_bot} L616,{bus_bot} L616,214' stroke='{FINE}' "
             f"stroke-dasharray='3 3' fill='none'/>")

    note = f"{series}s{parallel}p"
    if series > 4 or parallel > 4:
        note += " (lattice schematic: at most 4 × 4 cells drawn)"
    p.append(_t(344, 352, f"{note} — voltage adds in series, capacity in parallel",
                "middle", 9))
    return _svg("".join(p), 700, 368)


# --------------------------------------------------------------------------- ECM
def equivalent_circuit() -> str:
    """Thevenin equivalent circuit used by the simulator."""
    p = []
    top, bot = 78.0, 198.0
    p.append(f"<path d='M84,{top} L626,{top} L626,{bot} L84,{bot} Z' fill='none' "
             f"stroke='{INK}' stroke-width='1.3'/>")
    p.append(f"<circle cx='84' cy='{top}' r='4' fill='#FFFFFF' stroke='{ACCENT}' stroke-width='1.5'/>")
    p.append(f"<circle cx='84' cy='{bot}' r='4' fill='#FFFFFF' stroke='{INK}' stroke-width='1.5'/>")

    # R0 zigzag
    zig = ["M130,78"]
    for i in range(6):
        zig.append(f"L{142 + i * 9},{(70 if i % 2 == 0 else 86)}")
    zig.append("L196,78")
    p.append(f"<path d='{' '.join(zig)}' fill='none' stroke='{INK}' stroke-width='1.4'/>")
    p.append(_leader(160, 88, 160, 118))
    p.append(_t(160, 130, "R0 · ohmic", "middle"))
    p.append(_t(160, 142, "rises with ageing", "middle", 8, FINE))

    # RC branch
    p.append(f"<path d='M250,78 L250,138' stroke='{INK}' stroke-width='1.3'/>")
    p.append(f"<path d='M470,78 L470,138' stroke='{INK}' stroke-width='1.3'/>")
    p.append(f"<path d='M250,138 L292,138' stroke='{INK}' stroke-width='1.3'/>")
    zig2 = ["M292,138"]
    for i in range(5):
        zig2.append(f"L{302 + i * 9},{(130 if i % 2 == 0 else 146)}")
    zig2.append("L350,138")
    p.append(f"<path d='{' '.join(zig2)}' fill='none' stroke='{INK}' stroke-width='1.4'/>")
    p.append(f"<path d='M350,138 L412,138' stroke='{INK}' stroke-width='1.3'/>")
    p.append(f"<path d='M412,118 L412,158 M424,118 L424,158' stroke='{INK}' stroke-width='1.8'/>")
    p.append(f"<path d='M424,138 L470,138' stroke='{INK}' stroke-width='1.3'/>")
    p.append(_leader(418, 160, 418, 180))
    p.append(_t(418, 194, "R1 ∥ C1 · charge transfer", "middle"))

    # OCV source on the return leg
    p.append(f"<path d='M300,198 L318,198 M318,182 L318,214 M332,188 L332,208 M332,198 L360,198' "
             f"stroke='{INK}' stroke-width='1.5' fill='none'/>")
    p.append(_leader(325, 214, 325, 238))
    p.append(_t(325, 252, "OCV(SOC, T)", "middle"))

    # terminal voltage arrow
    p.append(f"<path d='M56,{top} L56,{bot - 1}' stroke='{FINE}' stroke-width='1'/>")
    p.append(f"<path d='M52,{top + 7} L56,{top} L60,{top + 7}' stroke='{FINE}' fill='none'/>")
    p.append(f"<path d='M52,{bot - 8} L56,{bot - 1} L60,{bot - 8}' stroke='{FINE}' fill='none'/>")
    p.append(_t(48, 142, "V_t", "end", 10, INK, 600))
    p.append(_t(640, 60, "terminal", "end", 8.5))
    p.append(_t(640, 214, "current path", "end", 8.5))
    return _svg("".join(p), 700, 262)


# --------------------------------------------------------------------------- pipeline
def pipeline(stages: list[tuple[str, str]] | None = None) -> str:
    """Left-to-right data flow."""
    stages = stages or [
        ("RAW FILES", "NASA · CALCE\nOxford · Samsung"),
        ("LOADERS", "per-dataset\nparsers"),
        ("FEATURES", "one row\nper cycle"),
        ("MODELS", "RF · GB\nXGBoost"),
        ("APP", "Streamlit\npages"),
    ]
    p = []
    w, h, y = 112, 62, 66
    for i, (head, note) in enumerate(stages):
        x = 18 + i * 136
        stroke = ACCENT if i == len(stages) - 1 else INK
        p.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='3' fill='#FFFFFF' "
                 f"stroke='{stroke}' stroke-width='1.5'/>")
        p.append(_t(x + w / 2, y + 26, head, "middle", 9.5, INK, 600))
        for j, line in enumerate(note.split("\n")):
            p.append(_t(x + w / 2, y + 42 + j * 12, line, "middle", 7.8))
        if i < len(stages) - 1:
            p.append(_arrow(x + w + 4, y + h / 2, x + w + 20, y + h / 2, FINE))
    p.append(_t(18, 158, "nothing in this chain writes to data/raw — the laboratory files stay read-only",
                "start", 8.5))
    return _svg("".join(p), 700, 176)


# --------------------------------------------------------------------------- ageing
def ageing_curve() -> str:
    """Capacity-fade trajectory with the 80% end-of-life line."""
    p = []
    left, right, top, bottom = 78.0, 646.0, 40.0, 214.0

    def x(cycle: float) -> float:
        return left + cycle / 1200 * (right - left)

    def y(soh: float) -> float:
        return top + (100 - soh) / 30 * (bottom - top)

    p.append(f"<path d='M{left},{top} L{left},{bottom} L{right},{bottom}' fill='none' "
             f"stroke='{INK}' stroke-width='1.2'/>")
    for tick in (0, 300, 600, 900, 1200):
        p.append(f"<path d='M{x(tick):.1f},{bottom} L{x(tick):.1f},{bottom + 5}' stroke='{INK}'/>")
        p.append(_t(x(tick), bottom + 18, str(tick), "middle", 8.5))
    for tick in (100, 90, 80, 70):
        p.append(f"<path d='M{left - 5},{y(tick):.1f} L{left},{y(tick):.1f}' stroke='{INK}'/>")
        p.append(_t(left - 10, y(tick) + 3, f"{tick}%", "end", 8.5))

    points = [(0, 100), (90, 99.3), (180, 98.6), (270, 97.8), (360, 96.4), (450, 94.2),
              (540, 91.4), (630, 88.0), (720, 84.0), (810, 80.2), (900, 77.0)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(c):.1f},{y(s):.1f}"
                    for i, (c, s) in enumerate(points))
    p.append(f"<path d='{path}' fill='none' stroke='{INK}' stroke-width='2'/>")

    p.append(f"<path d='M{left},{y(80):.1f} L{right},{y(80):.1f}' stroke='{ACCENT}' "
             f"stroke-width='1' stroke-dasharray='5 4'/>")
    p.append(_t(right, y(80) - 6, "end of life · 80% SOH", "end", 8.5, ACCENT))

    p.append(_leader(x(250), y(98.1), 300, 66))
    p.append(_t(306, 62, "SEI growth · slow, near-linear", "start", 8.5))
    p.append(_leader(x(700), y(85.2), 520, 118))
    p.append(_t(514, 124, "knee · plating / dry-out", "end", 8.5))
    p.append(_t(left + 6, top - 14, "SOH / %", "start", 8.5))
    p.append(_t(right, bottom + 34, "cycle number", "end", 8.5))
    return _svg("".join(p), 700, 262)


# --------------------------------------------------------------------------- data map
def dataset_map() -> str:
    """Four laboratories feeding one harmonised table."""
    p = []
    sets = [
        ("NASA PCoE", "18650 · .mat records"),
        ("CALCE CS2", "prismatic · Arbin sheets"),
        ("Oxford", "pouch · check-ups"),
        ("Samsung 35E", "18650 · tab-separated"),
    ]
    for i, (name, note) in enumerate(sets):
        y = 26 + i * 54
        p.append(f"<rect x='20' y='{y}' width='186' height='42' rx='3' fill='#FFFFFF' "
                 f"stroke='{INK}' stroke-width='1.3'/>")
        p.append(_t(32, y + 18, name, "start", 9.5, INK, 600))
        p.append(_t(32, y + 33, note, "start", 8))
        p.append(_arrow(212, y + 21, 286, y + 21, FINE))

    p.append(f"<rect x='292' y='54' width='158' height='146' rx='3' fill='#FFFFFF' "
             f"stroke='{ACCENT}' stroke-width='1.6'/>")
    p.append(_t(371, 84, "HARMONISED", "middle", 9.5, INK, 600))
    p.append(_t(371, 98, "CYCLE TABLE", "middle", 9.5, INK, 600))
    p.append(f"<path d='M306,110 L436,110' stroke='{FINE}'/>")
    p.append(_t(371, 132, "67 batteries", "middle", 8.5))
    p.append(_t(371, 148, "32,369 cycle rows", "middle", 8.5))
    p.append(_t(371, 164, "SOH · capacity", "middle", 8.5))
    p.append(_t(371, 180, "voltage · current · T", "middle", 8.5))

    p.append(_arrow(454, 127, 516, 127, FINE))
    p.append(f"<rect x='520' y='92' width='158' height='76' rx='3' fill='#FFFFFF' "
             f"stroke='{INK}' stroke-width='1.3'/>")
    p.append(_t(599, 116, "DASHBOARD", "middle", 9.5, INK, 600))
    p.append(_t(599, 132, "measured ≠ simulated", "middle", 8))
    p.append(_t(599, 148, "≠ predicted", "middle", 8))
    return _svg("".join(p), 700, 232)


# --------------------------------------------------------------------------- validation
def validation() -> str:
    """Leave-one-battery-out: one whole cell is removed per fold."""
    p = []
    held = {0: 4, 1: 1}
    box_w, box_h, step = 74.0, 32.0, 94.0
    for row in range(2):
        y = 50.0 + row * 74
        p.append(_t(30, y + 20, f"fold {row + 1}", "start", 9.5, INK, 600))
        for i in range(5):
            x = 132 + i * step
            out = held[row] == i
            dash = " stroke-dasharray='5 4'" if out else ""
            p.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='2' "
                     f"fill='#FFFFFF' stroke='{ACCENT if out else INK}' stroke-width='1.4'{dash}/>")
            p.append(_t(x + box_w / 2, y + 21, "battery", "middle", 8.5))
            if out:
                p.append(_t(x + box_w / 2, y - 8, "held out", "middle", 8.5, ACCENT, 600))

    p.append(f"<path d='M30,186 L676,186' stroke='{FINE}' stroke-width='1'/>")
    p.append(_t(30, 206, "the model never sees the held-out cell's cycles, so the reported error is a",
                "start", 8.5))
    p.append(_t(30, 222, "forecast error rather than a curve fit", "start", 8.5))
    return _svg("".join(p), 700, 234)


# --------------------------------------------------------------------------- signal
def signal_strip() -> str:
    """Three stacked measurement channels, the way a logger prints them."""
    p = []
    channels = [("voltage / V", INK), ("current / A", MUTED), ("temperature / °C", FINE)]
    for i, (name, color) in enumerate(channels):
        top = 26 + i * 48
        p.append(_t(16, top + 6, name, "start", 8.5))
        p.append(f"<path d='M150,{top + 14} L676,{top + 14}' stroke='#E4E9EF' stroke-width='1'/>")
        if i == 0:
            d = "M150,26 " + " ".join(f"L{150 + k * 12},{26 + (k ** 1.55) * 0.42:.1f}"
                                      for k in range(1, 45))
        elif i == 1:
            d = "M150,80 " + " ".join(
                f"L{150 + k * 12},{80 if k % 8 else 62}" for k in range(1, 45))
        else:
            d = "M150,134 " + " ".join(
                f"L{150 + k * 12},{134 - 10 * (1 - abs(22 - k % 22) / 22):.1f}" for k in range(1, 45))
        p.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='1.5'/>")
    return _svg("".join(p), 700, 172)
