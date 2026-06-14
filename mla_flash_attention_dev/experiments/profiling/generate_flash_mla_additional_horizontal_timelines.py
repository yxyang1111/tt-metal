from __future__ import annotations

import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "mla_flash_attention_dev" / "docs" / "assets" / "flash-mla-wh-profile-swimlanes-detailed"
MANIFEST_PATH = ASSET_DIR / "additional_horizontal_timelines_manifest.json"

LEFT = 280
RIGHT = 80
TOP = 180
BOTTOM = 130
LANE_H = 110
LANE_GAP = 24
WIDTH = 2320

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
    x0_unit: float
    y0_lane: int
    x1_unit: float
    y1_lane: int
    label: str
    color: str
    dashed: bool = False
    bend: float = 0.0


@dataclass(frozen=True)
class ChartSpec:
    filename_stem: str
    title: str
    subtitle: str
    summary_title: str
    summary_lines: list[str]
    lanes: list[Lane]
    tasks: list[Task]
    arrows: list[Arrow]
    ticks: list[tuple[float, str]]


def chart_height(num_lanes: int) -> int:
    return int(TOP + num_lanes * LANE_H + (num_lanes - 1) * LANE_GAP + BOTTOM)


def timeline_width() -> float:
    return WIDTH - LEFT - RIGHT


def unit_to_x(unit: float) -> float:
    return LEFT + (unit / 100.0) * timeline_width()


def lane_y(idx: int) -> float:
    return TOP + idx * (LANE_H + LANE_GAP)


def lane_center_y(idx: int) -> float:
    return lane_y(idx) + LANE_H / 2


def svg_text(x: float, y: float, text: str, size: int = 18, weight: str = "400", fill: str | None = None, anchor: str = "start") -> str:
    fill = fill or COLORS["text"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-family="Inter, DejaVu Sans, Arial, sans-serif">{html.escape(text)}</text>'
    )


def svg_multiline_text(x: float, y: float, lines: list[str], size: int = 16, fill: str | None = None, anchor: str = "start") -> str:
    fill = fill or COLORS["text"]
    tspans = []
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else "1.25em"
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="Inter, DejaVu Sans, Arial, sans-serif">' + "".join(tspans) + "</text>"
    )


def draw_lane_background(idx: int) -> str:
    y = lane_y(idx)
    fill = "#fafafa" if idx % 2 == 0 else "#f5f7fb"
    return f'<rect x="{LEFT}" y="{y:.1f}" width="{timeline_width():.1f}" height="{LANE_H}" rx="16" fill="{fill}" stroke="{COLORS["grid"]}" stroke-width="1"/>'


def draw_task(task: Task) -> str:
    x = unit_to_x(task.start)
    y = lane_y(task.lane_idx) + 16
    w = unit_to_x(task.end) - unit_to_x(task.start)
    h = LANE_H - 32
    lines = [task.title] + task.body.split("\n")
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="12" fill="{task.fill}" stroke="{task.stroke}" stroke-width="2"/>'
        + svg_multiline_text(x + 14, y + 24, lines, size=15)
    )


def draw_arrow(arrow: Arrow) -> str:
    x0 = unit_to_x(arrow.x0_unit)
    x1 = unit_to_x(arrow.x1_unit)
    y0 = lane_center_y(arrow.y0_lane)
    y1 = lane_center_y(arrow.y1_lane)
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2 + arrow.bend
    dash = ' stroke-dasharray="8 8"' if arrow.dashed else ""
    path = f'M {x0:.1f} {y0:.1f} Q {mx:.1f} {my:.1f} {x1:.1f} {y1:.1f}'
    return (
        f'<path d="{path}" fill="none" stroke="{arrow.color}" stroke-width="3"{dash} marker-end="url(#arrowhead)"/>'
        + svg_multiline_text((x0 + x1) / 2, my - 10, arrow.label.split("\n"), size=14, fill=arrow.color, anchor="middle")
    )


def draw_legend_item(x: float, y: float, fill: str, stroke: str, label: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="24" height="24" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        + svg_text(x + 34, y + 18, label, size=15)
    )


def build_svg(chart: ChartSpec) -> str:
    height = chart_height(len(chart.lanes))
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">')
    parts.append(f'<rect width="{WIDTH}" height="{height}" fill="{COLORS["bg"]}"/>')
    parts.append(
        """
<defs>
  <marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#374151"/>
  </marker>
</defs>
""".strip()
    )
    parts.append(svg_text(LEFT, 56, chart.title, size=34, weight="700"))
    parts.append(svg_text(LEFT, 88, chart.subtitle, size=18, fill=COLORS["muted"]))

    panel_x = WIDTH - 560
    panel_y = 26
    parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="500" height="124" rx="16" fill="#111827" opacity="0.97"/>')
    parts.append(svg_text(panel_x + 24, panel_y + 30, chart.summary_title, size=18, weight="700", fill="#f9fafb"))
    parts.append(svg_multiline_text(panel_x + 24, panel_y + 58, chart.summary_lines, size=16, fill="#e5e7eb"))

    axis_y = TOP - 38
    parts.append(f'<line x1="{LEFT}" y1="{axis_y}" x2="{LEFT + timeline_width():.1f}" y2="{axis_y}" stroke="{COLORS["text"]}" stroke-width="2"/>')
    bottom_grid = lane_y(len(chart.lanes) - 1) + LANE_H
    for unit, label in chart.ticks:
        x = unit_to_x(unit)
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y - 8}" x2="{x:.1f}" y2="{bottom_grid:.1f}" stroke="{COLORS["grid"]}" stroke-width="1.5" stroke-dasharray="4 6"/>')
        parts.append(svg_multiline_text(x, axis_y - 16, label.split("\n"), size=14, anchor="middle", fill=COLORS["muted"]))

    for idx, lane in enumerate(chart.lanes):
        y = lane_y(idx)
        parts.append(draw_lane_background(idx))
        parts.append(svg_text(34, y + 42, lane.name, size=22, weight="700"))
        parts.append(svg_text(34, y + 74, lane.subtitle, size=15, fill=COLORS["muted"]))

    for task in chart.tasks:
        parts.append(draw_task(task))
    for arrow in chart.arrows:
        parts.append(draw_arrow(arrow))

    legend_y = height - 80
    legend_items = [
        ("read_fill", "read_stroke", "read / input"),
        ("comm_fill", "comm_stroke", "communication / forward / reduction"),
        ("wait_fill", "wait_stroke", "wait / reserve / cb_wait"),
        ("compute_fill", "compute_stroke", "TRISC compute / merge"),
        ("write_fill", "write_stroke", "writer / output / handoff"),
    ]
    for idx, (fill_key, stroke_key, label) in enumerate(legend_items):
        parts.append(draw_legend_item(LEFT + idx * 360, legend_y, COLORS[fill_key], COLORS[stroke_key], label))

    parts.append("</svg>")
    return "\n".join(parts)


def render_chart(chart: ChartSpec) -> dict[str, str]:
    svg_path = ASSET_DIR / f"{chart.filename_stem}.svg"
    png_path = ASSET_DIR / f"{chart.filename_stem}.png"
    svg_path.write_text(build_svg(chart), encoding="utf-8")
    convert = shutil.which("convert")
    if convert:
        subprocess.run([convert, str(svg_path), str(png_path)], check=True, cwd=REPO_ROOT)
    return {
        "name": chart.filename_stem,
        "svg": str(svg_path.relative_to(REPO_ROOT)),
        "png": str(png_path.relative_to(REPO_ROOT)),
    }


PREFILL_SBLOCK = ChartSpec(
    filename_stem="prefill_sblock_4core_horizontal_timeline",
    title="FlashMLA WH Prefill 单个 S block 的 4-core 横向时间泳道图",
    subtitle="以 prefill_4k steady-state 为代表；强调 explicit Q/K/V read、forwarding、writer cb_wait 和 compute wait-front。",
    summary_title="Representative profile signals",
    summary_lines=[
        "reader wait = 66.10%",
        "writer cb_wait = 99.62%",
        "PM FPU util = 19.427%",
        "bubble wait-front = 92.6%",
    ],
    lanes=[
        Lane("DRAM / QKV Source", "Q + paged K + explicit V"),
        Lane("Core0", "sender / first worker"),
        Lane("Core1", "receiver / worker"),
        Lane("Core2", "receiver / worker"),
        Lane("Core3", "receiver / worker"),
        Lane("Writer / Output", "cb_out consume + write"),
    ],
    tasks=[
        Task(0, 4, 24, "Q/K/V source", "DRAM reads\npage table + explicit V", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(0, 86, 100, "Output shard", "chunk write /\npartial tile write", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(1, 6, 24, "NCRISC", "read Q/K/V + mask\nsender/forward source", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(1, 24, 36, "BRISC", "forward / valid sem\nnext workers ready", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(1, 36, 52, "TRISC", "cb_wait_front(Q/K/V)\nwait dominates", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(1, 52, 67, "TRISC", "local attention compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(1, 67, 82, "BRISC", "push cb_out /\nwait writer consume", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(1, 82, 94, "NCRISC", "next chunk read /\ncompletion wait", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(2, 24, 38, "NCRISC+BRISC", "forward recv /\ncb_push_back Q/K/V", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(2, 38, 52, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(2, 52, 67, "TRISC", "local attention compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(2, 67, 82, "BRISC", "push cb_out /\nwait writer", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(2, 82, 92, "idle/wait", "next tile pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(3, 24, 38, "NCRISC+BRISC", "forward recv /\ncb_push_back Q/K/V", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(3, 38, 52, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(3, 52, 67, "TRISC", "local attention compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(3, 67, 82, "BRISC", "push cb_out /\nwait writer", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(3, 82, 92, "idle/wait", "next tile pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(4, 24, 38, "NCRISC+BRISC", "forward recv /\ncb_push_back Q/K/V", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(4, 38, 52, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(4, 52, 67, "TRISC", "local attention compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(4, 67, 82, "BRISC", "push cb_out /\nwait writer", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(4, 82, 92, "idle/wait", "next tile pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(5, 66, 86, "BRISC", "cb_wait_front(cb_out)\nwriter dominates", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(5, 86, 100, "Writer", "chunk write /\noutput handoff", COLORS["write_fill"], COLORS["write_stroke"]),
    ],
    arrows=[
        Arrow(22, 0, 16, 1, "Q/K/V -> Core0", COLORS["read_stroke"], bend=-18),
        Arrow(34, 1, 28, 2, "forward", COLORS["comm_stroke"], bend=-18),
        Arrow(34, 1, 28, 3, "forward", COLORS["comm_stroke"], bend=0),
        Arrow(34, 1, 28, 4, "forward", COLORS["comm_stroke"], bend=18),
        Arrow(76, 1, 70, 5, "cb_out", COLORS["write_stroke"], bend=-22),
        Arrow(76, 2, 70, 5, "cb_out", COLORS["write_stroke"], bend=-6),
        Arrow(76, 3, 70, 5, "cb_out", COLORS["write_stroke"], bend=10),
        Arrow(76, 4, 70, 5, "cb_out", COLORS["write_stroke"], bend=24),
        Arrow(99, 5, 94, 0, "output write", COLORS["write_stroke"], bend=-20),
        Arrow(84, 5, 76, 1, "cb_wait backpressure", COLORS["danger"], dashed=True, bend=-48),
    ],
    ticks=[
        (0, "t0\nlaunch"),
        (8, "t1\nread"),
        (24, "t2\nforward"),
        (38, "t3\nwait"),
        (52, "t4\ncompute"),
        (67, "t5\ncb_out"),
        (86, "t6\nwrite"),
        (100, "t7\nnext"),
    ],
)


DECODE_CROSS_SBLOCK = ChartSpec(
    filename_stem="decode_cross_sblock_reduction_horizontal_timeline",
    title="FlashMLA WH Decode 跨 S block reduction / root 横向时间泳道图",
    subtitle="以 decode_32k steady-state 为代表；强调 child send、tree child wait、root merge 与 output gather 的横向时间关系。",
    summary_title="Representative profile signals",
    summary_lines=[
        "sender_cb_wait ≈ 49.95%",
        "tree_child_wait ≈ 50.05%",
        "root/output_gather ≈ 0%",
        "reserve-back bubble = 29.09%",
    ],
    lanes=[
        Lane("S-block A worker", "local compute + child send"),
        Lane("S-block B worker", "local compute + child send"),
        Lane("S-block C worker", "local compute + child send"),
        Lane("Parent reducer", "wait children + merge + upward send"),
        Lane("Root / output core", "final merge + output"),
        Lane("Output / DRAM", "final shard write"),
    ],
    tasks=[
        Task(0, 0, 20, "TRISC", "local SDPA output\nl/m/o ready", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(0, 20, 36, "BRISC", "cb_wait + child send\n-> cb_intermed_out", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(0, 36, 48, "BRISC", "reducer_semaphore_inc\nwait next round", COLORS["comm_fill"], COLORS["comm_stroke"]),

        Task(1, 4, 24, "TRISC", "local SDPA output\nl/m/o ready", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(1, 24, 40, "BRISC", "cb_wait + child send\n-> cb_intermed_out", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(1, 40, 52, "BRISC", "reducer_semaphore_inc\nwait next round", COLORS["comm_fill"], COLORS["comm_stroke"]),

        Task(2, 8, 28, "TRISC", "local SDPA output\nl/m/o ready", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(2, 28, 44, "BRISC", "cb_wait + child send\n-> cb_intermed_out", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(2, 44, 56, "BRISC", "reducer_semaphore_inc\nwait next round", COLORS["comm_fill"], COLORS["comm_stroke"]),

        Task(3, 22, 50, "BRISC", "tree child wait\nreducer_semaphore poll", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(3, 50, 66, "TRISC", "sdpa_reduce / merge", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(3, 66, 80, "BRISC", "send upward /\nparent-ready", COLORS["comm_fill"], COLORS["comm_stroke"]),

        Task(4, 66, 82, "BRISC", "wait child-ready /\nroot cb_wait", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(4, 82, 94, "TRISC", "final tail merge", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(4, 94, 100, "BRISC", "final output write", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(5, 95, 100, "OUT", "shard write /\nvisible", COLORS["write_fill"], COLORS["write_stroke"]),
    ],
    arrows=[
        Arrow(34, 0, 28, 3, "child l/m/o", COLORS["write_stroke"], bend=-24),
        Arrow(38, 1, 34, 3, "child l/m/o", COLORS["write_stroke"], bend=-4),
        Arrow(42, 2, 40, 3, "child l/m/o", COLORS["write_stroke"], bend=18),
        Arrow(78, 3, 72, 4, "merged upward", COLORS["comm_stroke"], bend=-18),
        Arrow(99, 4, 97, 5, "final output", COLORS["write_stroke"], bend=-14),
        Arrow(94, 4, 78, 3, "space release late /\nbackpressure", COLORS["danger"], dashed=True, bend=-56),
    ],
    ticks=[
        (0, "t0\nlocal"),
        (20, "t1\nchild send"),
        (40, "t2\nchild-ready"),
        (50, "t3\nparent merge"),
        (66, "t4\nupward"),
        (82, "t5\nroot merge"),
        (94, "t6\nwrite"),
        (100, "t7\ndone"),
    ],
)


DECODE_FULLSTACK = ChartSpec(
    filename_stem="decode_fullstack_horizontal_timeline",
    title="FlashMLA WH Decode 全链路横向时间泳道总图",
    subtitle="把 Host/Runtime、DRAM/Q source、单个 S block 内 4 个 cores、parent reducer、root/output 串到一张真正的横向时间图里。",
    summary_title="Representative profile signals",
    summary_lines=[
        "reader reserve = 59.71%",
        "writer cb_wait = 99.56%",
        "sender/tree wait ≈ 50/50",
        "bubble reserve-back = 28.04%",
    ],
    lanes=[
        Lane("Host CPU / Runtime", "layout + program args + completion"),
        Lane("DRAM / Q source", "paged KV cache + output-core L1"),
        Lane("Core0", "sender + output-core role"),
        Lane("Core1", "receiver worker"),
        Lane("Core2", "receiver worker"),
        Lane("Core3", "receiver worker"),
        Lane("Parent reducer", "tree child wait + merge"),
        Lane("Root / output core", "final merge + output write"),
        Lane("Output / Host visible", "output shard / completion"),
    ],
    tasks=[
        Task(0, 0, 10, "Host", "grid / root / sender\nprogram args + semaphores", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(0, 92, 100, "Host", "wait completion /\nobserve output visible", COLORS["control_fill"], COLORS["control_stroke"]),

        Task(1, 8, 18, "Q source", "output-core L1 visible\nq_locally_available", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(1, 18, 34, "K source", "paged KV cache\npage table -> K tiles", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(1, 94, 100, "OUT", "final shard write /\noutput visible", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(2, 8, 18, "BRISC", "Q local visible /\noutput-core role", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(2, 18, 34, "NCRISC", "paged K read ->\nlocal cb_k_in", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(2, 34, 46, "BRISC", "K multicast sender\nk_mcast semaphore", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(2, 46, 58, "TRISC", "cb_wait_front(Q/K/V)\ninput not fully ready", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(2, 58, 72, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(2, 72, 84, "BRISC", "cb_wait + child send\nl/m/o -> parent", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(2, 84, 96, "NCRISC", "reserve next chunk\nreader reserve", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(3, 12, 20, "Reader side", "Q fetch / wait", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(3, 36, 48, "NCRISC+BRISC", "recv K + cb_push_back\ncb_k_in/cb_v_in/mask", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(3, 48, 58, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(3, 58, 72, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(3, 72, 84, "BRISC", "child send /\nreducer ready", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(3, 84, 92, "wait", "next round pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(4, 12, 20, "Reader side", "Q fetch / wait", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(4, 36, 48, "NCRISC+BRISC", "recv K + cb_push_back\ncb_k_in/cb_v_in/mask", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(4, 48, 58, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(4, 58, 72, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(4, 72, 84, "BRISC", "child send /\nreducer ready", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(4, 84, 92, "wait", "next round pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(5, 12, 20, "Reader side", "Q fetch / wait", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(5, 36, 48, "NCRISC+BRISC", "recv K + cb_push_back\ncb_k_in/cb_v_in/mask", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(5, 48, 58, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(5, 58, 72, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(5, 72, 84, "BRISC", "child send /\nreducer ready", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(5, 84, 92, "wait", "next round pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(6, 74, 90, "BRISC", "tree child wait\nreducer_semaphore poll", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(6, 90, 98, "TRISC", "merge child l/m/o", COLORS["compute_fill"], COLORS["compute_stroke"]),

        Task(7, 90, 96, "BRISC", "wait parent-ready /\ncb_wait root side", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(7, 96, 100, "TRISC+BRISC", "final merge /\noutput write", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(8, 96, 100, "Output", "host-visible /\ncompletion", COLORS["write_fill"], COLORS["write_stroke"]),
    ],
    arrows=[
        Arrow(9, 0, 10, 2, "launch sender/output", COLORS["control_stroke"], bend=-30),
        Arrow(9, 0, 12, 6, "launch reducer/root", COLORS["control_stroke"], bend=28),
        Arrow(17, 1, 12, 2, "Q local visible", COLORS["control_stroke"], bend=-16),
        Arrow(32, 1, 22, 2, "paged K -> Core0", COLORS["read_stroke"], bend=-22),
        Arrow(44, 2, 38, 3, "K multicast", COLORS["comm_stroke"], bend=-22),
        Arrow(44, 2, 38, 4, "K multicast", COLORS["comm_stroke"], bend=0),
        Arrow(44, 2, 38, 5, "K multicast", COLORS["comm_stroke"], bend=22),
        Arrow(82, 3, 76, 6, "child l/m/o", COLORS["write_stroke"], bend=-26),
        Arrow(82, 4, 78, 6, "child l/m/o", COLORS["write_stroke"], bend=-4),
        Arrow(82, 5, 80, 6, "child l/m/o", COLORS["write_stroke"], bend=18),
        Arrow(96, 6, 92, 7, "merged upward", COLORS["comm_stroke"], bend=-18),
        Arrow(99, 7, 98, 8, "final output", COLORS["write_stroke"], bend=-12),
        Arrow(99, 8, 96, 0, "completion", COLORS["control_stroke"], bend=-28),
        Arrow(96, 7, 86, 2, "space release late /\nbackpressure", COLORS["danger"], dashed=True, bend=-72),
    ],
    ticks=[
        (0, "t0\nlayout"),
        (10, "t1\nlaunch"),
        (18, "t2\nQ local"),
        (34, "t3\nK read"),
        (46, "t4\nmcast"),
        (58, "t5\nwait"),
        (72, "t6\ncompute"),
        (84, "t7\nchild send"),
        (90, "t8\nparent merge"),
        (96, "t9\nroot write"),
        (100, "t10\ndone"),
    ],
)


DECODE_VS_PREFILL = ChartSpec(
    filename_stem="decode_vs_prefill_horizontal_timeline",
    title="FlashMLA WH Decode vs Prefill 总对比横向时间泳道图",
    subtitle="上半部分是长序列 decode steady-state，下半部分是 prefill_4k steady-state；使用同一时间轴直接比较关键等待链路。",
    summary_title="Key contrast signals",
    summary_lines=[
        "decode: reader reserve + child/tree wait",
        "prefill: reader wait + writer cb_wait",
        "decode bubble: reserve-back visible",
        "prefill bubble: wait-front = 92.6%",
    ],
    lanes=[
        Lane("Decode | DRAM / source", "Q local + paged K + output"),
        Lane("Decode | Core0", "sender + output-core"),
        Lane("Decode | Workers", "recv + local compute + child send"),
        Lane("Decode | Reducer/root", "tree wait + merge + output"),
        Lane("Prefill | DRAM / QKV", "Q + paged K + explicit V"),
        Lane("Prefill | Core0", "read + forward source"),
        Lane("Prefill | Workers", "forward recv + local compute"),
        Lane("Prefill | Writer/output", "cb_wait_front + write"),
    ],
    tasks=[
        Task(0, 8, 18, "Q source", "q_locally_available", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(0, 18, 32, "K source", "paged K / page table", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(0, 94, 100, "OUT", "final shard write", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(1, 8, 18, "BRISC", "Q local visible", COLORS["control_fill"], COLORS["control_stroke"]),
        Task(1, 18, 32, "NCRISC", "paged K read", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(1, 32, 44, "BRISC", "K multicast sender", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(1, 44, 56, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(1, 56, 70, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(1, 70, 84, "BRISC", "child send", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(1, 84, 96, "NCRISC", "reader reserve /\nbackpressure", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(2, 34, 46, "NCRISC+BRISC", "recv K + cb_push_back", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(2, 46, 56, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(2, 56, 70, "TRISC", "local SDPA compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(2, 70, 84, "BRISC", "child send", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(2, 84, 92, "wait", "next round pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(3, 74, 90, "BRISC", "tree child wait", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(3, 90, 96, "TRISC", "merge", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(3, 96, 100, "BRISC", "final output write", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(4, 4, 24, "Q/K/V source", "DRAM reads\npage table + explicit V", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(4, 86, 100, "OUT", "chunk write", COLORS["write_fill"], COLORS["write_stroke"]),

        Task(5, 6, 24, "NCRISC", "read Q/K/V", COLORS["read_fill"], COLORS["read_stroke"]),
        Task(5, 24, 38, "BRISC", "forward / valid sem", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(5, 38, 52, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(5, 52, 67, "TRISC", "local attention compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(5, 67, 84, "BRISC", "push cb_out /\nwait writer", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(5, 84, 94, "NCRISC", "next chunk wait", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(6, 24, 38, "NCRISC+BRISC", "forward recv /\ncb_push_back", COLORS["comm_fill"], COLORS["comm_stroke"]),
        Task(6, 38, 52, "TRISC", "cb_wait_front", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(6, 52, 67, "TRISC", "local attention compute", COLORS["compute_fill"], COLORS["compute_stroke"]),
        Task(6, 67, 84, "BRISC", "push cb_out /\nwait writer", COLORS["write_fill"], COLORS["write_stroke"]),
        Task(6, 84, 92, "wait", "next tile pending", COLORS["wait_fill"], COLORS["wait_stroke"]),

        Task(7, 66, 86, "BRISC", "cb_wait_front(cb_out)", COLORS["wait_fill"], COLORS["wait_stroke"]),
        Task(7, 86, 100, "Writer", "chunk write /\noutput handoff", COLORS["write_fill"], COLORS["write_stroke"]),
    ],
    arrows=[
        Arrow(17, 0, 12, 1, "Q local", COLORS["control_stroke"], bend=-16),
        Arrow(30, 0, 22, 1, "paged K", COLORS["read_stroke"], bend=-18),
        Arrow(42, 1, 36, 2, "K multicast", COLORS["comm_stroke"], bend=-18),
        Arrow(82, 2, 78, 3, "child l/m/o", COLORS["write_stroke"], bend=-18),
        Arrow(99, 3, 97, 0, "final output", COLORS["write_stroke"], bend=-16),
        Arrow(94, 3, 86, 1, "reserve/backpressure", COLORS["danger"], dashed=True, bend=-54),

        Arrow(22, 4, 16, 5, "Q/K/V -> Core0", COLORS["read_stroke"], bend=-18),
        Arrow(34, 5, 28, 6, "forward", COLORS["comm_stroke"], bend=-16),
        Arrow(76, 6, 70, 7, "cb_out", COLORS["write_stroke"], bend=-18),
        Arrow(99, 7, 94, 4, "output write", COLORS["write_stroke"], bend=-18),
        Arrow(84, 7, 76, 5, "writer backpressure", COLORS["danger"], dashed=True, bend=-48),
    ],
    ticks=[
        (0, "t0\nstart"),
        (18, "t1\nsource"),
        (32, "t2\ndispatch"),
        (44, "t3\nrecv/wait"),
        (56, "t4\ncompute"),
        (70, "t5\nsend/cb_out"),
        (86, "t6\nmerge/write"),
        (100, "t7\ndone"),
    ],
)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for chart in [PREFILL_SBLOCK, DECODE_CROSS_SBLOCK, DECODE_FULLSTACK, DECODE_VS_PREFILL]:
        manifest.append(render_chart(chart))
        print(f"wrote {chart.filename_stem}")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
