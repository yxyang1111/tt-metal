from __future__ import annotations

import html
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "mla_flash_attention_dev" / "docs" / "assets" / "flash-mla-wh-profile-swimlanes-detailed"
SVG_PATH = ASSET_DIR / "decode_sblock_4core_horizontal_timeline.svg"
PNG_PATH = ASSET_DIR / "decode_sblock_4core_horizontal_timeline.png"


@dataclass(frozen=True)
class Lane:
    name: str
    subtitle: str


@dataclass(frozen=True)
class Task:
    lane_idx: int
    start: float
    end: float
    title: str
    body: str
    fill: str
    stroke: str


@dataclass(frozen=True)
class Arrow:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str
    color: str
    dashed: bool = False
    bend: float = 0.0


WIDTH = 2200
HEIGHT = 1180
LEFT = 270
RIGHT = 70
TOP = 170
BOTTOM = 120
LANE_H = 110
LANE_GAP = 24
TIMELINE_W = WIDTH - LEFT - RIGHT

LANES = [
    Lane("DRAM / Q Source", "KV cache + output-core L1"),
    Lane("Core0", "sender + output-core role"),
    Lane("Core1", "worker / receiver"),
    Lane("Core2", "worker / receiver"),
    Lane("Core3", "worker / receiver"),
    Lane("Parent / Root", "tree reduction + output gather"),
]

COLORS = {
    "read_fill": "#dbeafe",
    "read_stroke": "#2563eb",
    "wait_fill": "#ffedd5",
    "wait_stroke": "#ea580c",
    "comm_fill": "#ede9fe",
    "comm_stroke": "#7c3aed",
    "compute_fill": "#dcfce7",
    "compute_stroke": "#16a34a",
    "write_fill": "#fce7f3",
    "write_stroke": "#db2777",
    "control_fill": "#f3f4f6",
    "control_stroke": "#4b5563",
    "bg": "#ffffff",
    "grid": "#e5e7eb",
    "text": "#111827",
    "muted": "#6b7280",
    "danger": "#dc2626",
}

TASKS = [
    Task(0, 2, 11, "Q ready", "output-core L1 visible\nq_locally_available", COLORS["control_fill"], COLORS["control_stroke"]),
    Task(0, 12, 28, "K source", "paged KV cache\npage table -> K tiles", COLORS["read_fill"], COLORS["read_stroke"]),
    Task(0, 92, 100, "Output write", "root final write /\noutput gather", COLORS["write_fill"], COLORS["write_stroke"]),

    Task(1, 0, 8, "BRISC", "Q local visible /\noutput-core role", COLORS["control_fill"], COLORS["control_stroke"]),
    Task(1, 12, 28, "NCRISC", "paged K read ->\nlocal cb_k_in", COLORS["read_fill"], COLORS["read_stroke"]),
    Task(1, 28, 40, "BRISC", "K multicast sender\nsemaphore ready", COLORS["comm_fill"], COLORS["comm_stroke"]),
    Task(1, 40, 54, "TRISC", "cb_wait_front(Q/K/V)\ninput not fully ready", COLORS["wait_fill"], COLORS["wait_stroke"]),
    Task(1, 54, 70, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
    Task(1, 70, 82, "BRISC", "cb_wait + sender write\nl/m/o -> parent", COLORS["write_fill"], COLORS["write_stroke"]),
    Task(1, 82, 96, "NCRISC", "reserve next chunk\nreader reserve rises", COLORS["wait_fill"], COLORS["wait_stroke"]),

    Task(2, 8, 18, "Reader side", "Q fetch / wait\nk_mcast semaphore", COLORS["control_fill"], COLORS["control_stroke"]),
    Task(2, 30, 42, "NCRISC+BRISC", "recv K + cb_push_back\ncb_k_in/cb_v_in/mask", COLORS["comm_fill"], COLORS["comm_stroke"]),
    Task(2, 42, 54, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
    Task(2, 54, 70, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
    Task(2, 70, 82, "BRISC", "cb_wait + child send", COLORS["write_fill"], COLORS["write_stroke"]),
    Task(2, 82, 92, "TRISC/BRISC", "wait next round /\ncb_pop_front", COLORS["wait_fill"], COLORS["wait_stroke"]),

    Task(3, 8, 18, "Reader side", "Q fetch / wait\nk_mcast semaphore", COLORS["control_fill"], COLORS["control_stroke"]),
    Task(3, 30, 42, "NCRISC+BRISC", "recv K + cb_push_back\ncb_k_in/cb_v_in/mask", COLORS["comm_fill"], COLORS["comm_stroke"]),
    Task(3, 42, 54, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
    Task(3, 54, 70, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
    Task(3, 70, 82, "BRISC", "cb_wait + child send", COLORS["write_fill"], COLORS["write_stroke"]),
    Task(3, 82, 92, "TRISC/BRISC", "wait next round /\ncb_pop_front", COLORS["wait_fill"], COLORS["wait_stroke"]),

    Task(4, 8, 18, "Reader side", "Q fetch / wait\nk_mcast semaphore", COLORS["control_fill"], COLORS["control_stroke"]),
    Task(4, 30, 42, "NCRISC+BRISC", "recv K + cb_push_back\ncb_k_in/cb_v_in/mask", COLORS["comm_fill"], COLORS["comm_stroke"]),
    Task(4, 42, 54, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
    Task(4, 54, 70, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
    Task(4, 70, 82, "BRISC", "cb_wait + child send", COLORS["write_fill"], COLORS["write_stroke"]),
    Task(4, 82, 92, "TRISC/BRISC", "wait next round /\ncb_pop_front", COLORS["wait_fill"], COLORS["wait_stroke"]),

    Task(5, 70, 82, "BRISC", "wait child-ready\nreducer_semaphore", COLORS["wait_fill"], COLORS["wait_stroke"]),
    Task(5, 82, 92, "TRISC", "sdpa_tail / reduce\nmerge l/m/o", COLORS["compute_fill"], COLORS["compute_stroke"]),
    Task(5, 92, 100, "BRISC", "final write /\noutput gather", COLORS["write_fill"], COLORS["write_stroke"]),
]


def unit_to_x(unit: float) -> float:
    return LEFT + (unit / 100.0) * TIMELINE_W


def lane_y(idx: int) -> float:
    return TOP + idx * (LANE_H + LANE_GAP)


def lane_center_y(idx: int) -> float:
    return lane_y(idx) + LANE_H / 2


ARROWS = [
    Arrow(unit_to_x(18), lane_center_y(0), unit_to_x(14), lane_center_y(1), "Q local visible", COLORS["control_stroke"], bend=-40),
    Arrow(unit_to_x(27), lane_center_y(0), unit_to_x(20), lane_center_y(1), "paged K -> Core0", COLORS["read_stroke"], bend=-24),
    Arrow(unit_to_x(39), lane_center_y(1), unit_to_x(31), lane_center_y(2), "K multicast", COLORS["comm_stroke"], bend=-18),
    Arrow(unit_to_x(39), lane_center_y(1), unit_to_x(31), lane_center_y(3), "K multicast", COLORS["comm_stroke"], bend=0),
    Arrow(unit_to_x(39), lane_center_y(1), unit_to_x(31), lane_center_y(4), "K multicast", COLORS["comm_stroke"], bend=18),
    Arrow(unit_to_x(81), lane_center_y(2), unit_to_x(73), lane_center_y(5), "child l/m/o", COLORS["write_stroke"], bend=-24),
    Arrow(unit_to_x(81), lane_center_y(3), unit_to_x(73), lane_center_y(5), "child l/m/o", COLORS["write_stroke"], bend=0),
    Arrow(unit_to_x(81), lane_center_y(4), unit_to_x(73), lane_center_y(5), "child l/m/o", COLORS["write_stroke"], bend=24),
    Arrow(unit_to_x(99), lane_center_y(5), unit_to_x(94), lane_center_y(0), "final output", COLORS["write_stroke"], bend=-22),
    Arrow(unit_to_x(92), lane_center_y(5), unit_to_x(84), lane_center_y(1), "space release late /\nbackpressure", COLORS["danger"], dashed=True, bend=-60),
]


def svg_text(x: float, y: float, text: str, size: int = 18, weight: str = "400", fill: str | None = None, anchor: str = "start") -> str:
    fill = fill or COLORS["text"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-family="Inter, DejaVu Sans, Arial, sans-serif">{html.escape(text)}</text>'
    )


def svg_multiline_text(x: float, y: float, lines: list[str], size: int = 16, fill: str | None = None, anchor: str = "start") -> str:
    fill = fill or COLORS["text"]
    spans = []
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else "1.25em"
        spans.append(
            f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-family="Inter, DejaVu Sans, Arial, sans-serif">' + "".join(spans) + "</text>"
    )


def draw_lane_background(idx: int) -> str:
    y = lane_y(idx)
    fill = "#fafafa" if idx % 2 == 0 else "#f5f7fb"
    return (
        f'<rect x="{LEFT}" y="{y:.1f}" width="{TIMELINE_W}" height="{LANE_H}" '
        f'rx="16" fill="{fill}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
    )


def draw_task(task: Task) -> str:
    x = unit_to_x(task.start)
    y = lane_y(task.lane_idx) + 16
    w = unit_to_x(task.end) - unit_to_x(task.start)
    h = LANE_H - 32
    lines = [task.title] + task.body.split("\n")
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="12" '
        f'fill="{task.fill}" stroke="{task.stroke}" stroke-width="2"/>'
        + svg_multiline_text(x + 14, y + 24, lines, size=15)
    )


def draw_arrow(arrow: Arrow) -> str:
    mx = (arrow.x0 + arrow.x1) / 2
    my = (arrow.y0 + arrow.y1) / 2 + arrow.bend
    dash = ' stroke-dasharray="8 8"' if arrow.dashed else ""
    path = (
        f'M {arrow.x0:.1f} {arrow.y0:.1f} '
        f'Q {mx:.1f} {my:.1f} {arrow.x1:.1f} {arrow.y1:.1f}'
    )
    label_x = (arrow.x0 + arrow.x1) / 2
    label_y = my - 8
    return (
        f'<path d="{path}" fill="none" stroke="{arrow.color}" stroke-width="3"{dash} marker-end="url(#arrowhead)"/>'
        + svg_multiline_text(label_x, label_y, arrow.label.split("\n"), size=14, fill=arrow.color, anchor="middle")
    )


def draw_legend_item(x: float, y: float, fill: str, stroke: str, label: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="24" height="24" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        + svg_text(x + 34, y + 18, label, size=15, fill=COLORS["text"])
    )


def build_svg() -> str:
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["bg"]}"/>')
    parts.append(
        """
<defs>
  <marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#374151"/>
  </marker>
</defs>
""".strip()
    )

    parts.append(svg_text(LEFT, 56, "FlashMLA WH 单个 S block 的 4-core 横向时间泳道图", size=34, weight="700"))
    parts.append(svg_text(LEFT, 88, "以 decode_16k steady-state 为代表；Core0 是 sender/output core，Core1/2/3 是 receiver workers。横向是真正的概念时间轴。", size=18, fill=COLORS["muted"]))

    # Summary panel
    panel_x = WIDTH - 520
    panel_y = 28
    parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="460" height="118" rx="16" fill="#111827" opacity="0.96"/>')
    parts.append(svg_text(panel_x + 24, panel_y + 30, "Representative profile signals", size=18, weight="700", fill="#f9fafb"))
    parts.append(svg_multiline_text(panel_x + 24, panel_y + 58, [
        "reader reserve = 59.71%",
        "writer cb_wait = 99.56%",
        "bubble wait-front = 71.96%",
        "bubble reserve-back = 28.04%",
    ], size=16, fill="#e5e7eb"))

    # Time axis
    axis_y = TOP - 38
    parts.append(f'<line x1="{LEFT}" y1="{axis_y}" x2="{LEFT + TIMELINE_W}" y2="{axis_y}" stroke="{COLORS["text"]}" stroke-width="2"/>')
    ticks = [
        (0, "t0\nQ local"),
        (12, "t1\nK read"),
        (28, "t2\nmcast"),
        (42, "t3\nwait"),
        (54, "t4\ncompute"),
        (70, "t5\nchild send"),
        (82, "t6\nparent merge"),
        (92, "t7\noutput"),
        (100, "t8\nnext chunk"),
    ]
    for unit, label in ticks:
        x = unit_to_x(unit)
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y - 8}" x2="{x:.1f}" y2="{TOP + 5 * (LANE_H + LANE_GAP) + LANE_H}" stroke="{COLORS["grid"]}" stroke-width="1.5" stroke-dasharray="4 6"/>')
        parts.append(svg_multiline_text(x, axis_y - 16, label.split("\n"), size=14, anchor="middle", fill=COLORS["muted"]))

    # Lane labels and backgrounds
    for idx, lane in enumerate(LANES):
        y = lane_y(idx)
        parts.append(draw_lane_background(idx))
        parts.append(svg_text(34, y + 42, lane.name, size=22, weight="700"))
        parts.append(svg_text(34, y + 74, lane.subtitle, size=15, fill=COLORS["muted"]))

    # Tasks
    for task in TASKS:
        parts.append(draw_task(task))

    # Arrows
    for arrow in ARROWS:
        parts.append(draw_arrow(arrow))

    # Legend
    legend_y = HEIGHT - 80
    legend_x = LEFT
    legend_items = [
        ("read_fill", "read_stroke", "read / input"),
        ("comm_fill", "comm_stroke", "communication / multicast"),
        ("wait_fill", "wait_stroke", "wait / reserve / cb_wait"),
        ("compute_fill", "compute_stroke", "TRISC compute / reduce"),
        ("write_fill", "write_stroke", "writer / reduction send / output"),
    ]
    for idx, (fill_key, stroke_key, label) in enumerate(legend_items):
        parts.append(draw_legend_item(legend_x + idx * 330, legend_y, COLORS[fill_key], COLORS[stroke_key], label))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    SVG_PATH.write_text(svg, encoding="utf-8")
    print(f"wrote {SVG_PATH.relative_to(REPO_ROOT)}")

    convert = shutil.which("convert")
    if convert:
        subprocess.run([convert, str(SVG_PATH), str(PNG_PATH)], check=True, cwd=REPO_ROOT)
        print(f"wrote {PNG_PATH.relative_to(REPO_ROOT)}")
    else:
        print("convert not found; skipped PNG export")


if __name__ == "__main__":
    main()
