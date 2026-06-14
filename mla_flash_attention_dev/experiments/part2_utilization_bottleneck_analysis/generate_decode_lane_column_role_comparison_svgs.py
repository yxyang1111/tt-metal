#!/usr/bin/env python3
"""
Generate paired decode lane/column role swimlane diagrams:

1. Theoretical steady-state decode pipeline:
   - minimal bubbles / minimal synchronization
   - Q preparation once
   - read K_(t+1) || multicast K_t || compute K_t

2. Actual decode_32k anchored schematic:
   - wait-front / reserve-back on TRISC
   - K reserve on sender NCRISC
   - sender/tree waits on BRISC tail

These are normalized-time role diagrams, not per-core timestamp traces.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "outputs" / "raw" / "part2_utilization_bottleneck_analysis.json"
DEFAULT_OUT_THEORY = ROOT / "outputs" / "visuals" / "decode_lane_column_roles_theoretical.svg"
DEFAULT_OUT_ACTUAL = ROOT / "outputs" / "visuals" / "decode_lane_column_roles_actual_decode32k.svg"
DEFAULT_CASE = "decode_32k"


WIDTH = 1600
HEIGHT = 1180
LEFT_MARGIN = 250
RIGHT_MARGIN = 40
TITLE_Y = 38
SUBTITLE_Y = 60
NOTE_Y = 82
LEGEND_Y = 110
SECTION_BAND_Y = 150
SECTION_BAND_H = 30
ROWS_TOP_Y = 198
ROLE_ROW_H = 122
ROLE_GAP = 18
TRACK_H = 24
TRACK_GAP = 6
TRACKS_TOP_PAD = 14
FOOTER_Y = HEIGHT - 80
CHART_W = WIDTH - LEFT_MARGIN - RIGHT_MARGIN

FG = "#f5f7fb"
SUB_FG = "#9fb0c3"
MUTED = "#8fa3b8"
BG = "#111722"
PANEL = "#0f141d"
BORDER = "#223046"
TRACK_BORDER = "#1c2433"
GHOST = "#2a3340"

SECTIONS = {
    "q_prep": ("Q Preparation", 0.00, 0.12),
    "k_loop": ("K-Chunk Steady-State Loop", 0.12, 0.74),
    "tail": ("Tree Reduction + Final Output", 0.74, 1.00),
}

PALETTE = {
    "q": "#2ec4b6",
    "k_read": "#1982c4",
    "k_wait": "#ffca3a",
    "reserve": "#ff595e",
    "mcast": "#3d5a80",
    "tree_wait": "#9b5de5",
    "qk": "#1b998b",
    "softmax": "#ff9f1c",
    "pv": "#5c7cfa",
    "output": "#00bbf9",
    "ghost": GHOST,
}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def row_by_case(rows: list[dict[str, Any]], case: str) -> dict[str, Any]:
    for row in rows:
        if row.get("case") == case:
            return row
    raise KeyError(case)


@dataclass
class Segment:
    id: str
    label: str
    section_id: str
    start_frac: float
    end_frac: float
    color: str
    style: str = "solid"  # solid | ghost | cycle
    opacity: float = 1.0
    note: str = ""
    text_color: str = FG


@dataclass
class TrackSpec:
    label: str
    segments: list[Segment] = field(default_factory=list)


@dataclass
class RoleSpec:
    label: str
    tracks: list[TrackSpec]


@dataclass
class ArrowSpec:
    source_id: str
    target_id: str
    label: str
    color: str = "#88c0ff"
    dashed: bool = False
    bend: float = 0.0


@dataclass
class CalloutSpec:
    role_label: str
    track_label: str
    x_norm: float
    text: str
    dy: float = -10.0
    color: str = SUB_FG
    anchor: str = "middle"


@dataclass
class DiagramSpec:
    title: str
    subtitle: str
    logic_note: str
    roles: list[RoleSpec]
    arrows: list[ArrowSpec]
    callouts: list[CalloutSpec]
    footnotes: list[str]


def x_norm_to_px(x_norm: float) -> float:
    return LEFT_MARGIN + x_norm * CHART_W


def section_x(section_id: str, frac: float) -> float:
    _, start, end = SECTIONS[section_id]
    x_norm = start + (end - start) * frac
    return x_norm_to_px(x_norm)


def role_top(role_idx: int) -> float:
    return ROWS_TOP_Y + role_idx * (ROLE_ROW_H + ROLE_GAP)


def track_top(role_idx: int, track_idx: int) -> float:
    return role_top(role_idx) + TRACKS_TOP_PAD + track_idx * (TRACK_H + TRACK_GAP)


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def make_cycle_cluster(
    prefix: str,
    section_id: str,
    start_frac: float,
    end_frac: float,
    *,
    base_id: str,
) -> list[Segment]:
    span = end_frac - start_frac
    qk_w = span * 0.42
    sm_w = span * 0.26
    pv_w = span * 0.32
    a = start_frac
    b = a + qk_w
    c = b + sm_w
    return [
        Segment(
            id=f"{base_id}_qk",
            label="QK^T",
            section_id=section_id,
            start_frac=a,
            end_frac=b,
            color=PALETTE["qk"],
            style="cycle",
            note=prefix,
        ),
        Segment(
            id=f"{base_id}_sm",
            label="softmax",
            section_id=section_id,
            start_frac=b,
            end_frac=c,
            color=PALETTE["softmax"],
            style="cycle",
        ),
        Segment(
            id=f"{base_id}_pv",
            label="P@V",
            section_id=section_id,
            start_frac=c,
            end_frac=end_frac,
            color=PALETTE["pv"],
            style="cycle",
        ),
    ]


def legend_items() -> list[tuple[str, str]]:
    return [
        ("Q local / pull", PALETTE["q"]),
        ("K read", PALETTE["k_read"]),
        ("K wait", PALETTE["k_wait"]),
        ("reserve / backpressure", PALETTE["reserve"]),
        ("multicast", PALETTE["mcast"]),
        ("tree / sender wait", PALETTE["tree_wait"]),
        ("output write / gather", PALETTE["output"]),
    ]


def build_theoretical_spec(case: str, seq_len: int, num_chunks: int) -> DiagramSpec:
    common_compute = [
        *make_cycle_cluster("K0", "k_loop", 0.18, 0.36, base_id="common_k0"),
        *make_cycle_cluster("K1", "k_loop", 0.36, 0.54, base_id="common_k1"),
        *make_cycle_cluster("K2", "k_loop", 0.54, 0.72, base_id="common_k2"),
    ]

    roles = [
        RoleSpec(
            label="Q source（兼 worker）",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment(
                            id="theory_qsource_n_q",
                            label="Q local / ready",
                            section_id="q_prep",
                            start_frac=0.20,
                            end_frac=0.85,
                            color=PALETTE["q"],
                        ),
                        Segment(
                            id="theory_qsource_n_ghost",
                            label="",
                            section_id="tail",
                            start_frac=0.00,
                            end_frac=1.00,
                            color=PALETTE["ghost"],
                            style="ghost",
                            opacity=0.24,
                        ),
                        Segment("theory_qsource_n_r0", "mcast-ready", "k_loop", 0.14, 0.18, PALETTE["k_wait"]),
                        Segment("theory_qsource_n_r1", "mcast-ready", "k_loop", 0.28, 0.32, PALETTE["k_wait"]),
                        Segment("theory_qsource_n_r2", "mcast-ready", "k_loop", 0.50, 0.54, PALETTE["k_wait"]),
                    ],
                ),
                TrackSpec("TRISC", list(common_compute)),
                TrackSpec(
                    "BRISC",
                    [
                        Segment(
                            id="theory_qsource_b_pull",
                            label="serve Q pulls",
                            section_id="q_prep",
                            start_frac=0.25,
                            end_frac=0.95,
                            color=PALETTE["q"],
                        ),
                        Segment(
                            id="theory_qsource_b_send",
                            label="send local (m,l,O)",
                            section_id="tail",
                            start_frac=0.08,
                            end_frac=0.28,
                            color=PALETTE["output"],
                        ),
                    ],
                ),
            ],
        ),
        RoleSpec(
            label="Column sender（兼 worker）",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment("theory_sender_n_k0", "read K0", "k_loop", 0.00, 0.14, PALETTE["k_read"]),
                        Segment("theory_sender_n_k1", "read K1", "k_loop", 0.14, 0.28, PALETTE["k_read"]),
                        Segment("theory_sender_n_k2", "read K2", "k_loop", 0.36, 0.50, PALETTE["k_read"]),
                    ],
                ),
                TrackSpec(
                    "TRISC",
                    [
                        *make_cycle_cluster("K0", "k_loop", 0.14, 0.32, base_id="theory_sender_t_k0"),
                        *make_cycle_cluster("K1", "k_loop", 0.32, 0.50, base_id="theory_sender_t_k1"),
                        *make_cycle_cluster("K2", "k_loop", 0.50, 0.68, base_id="theory_sender_t_k2"),
                    ],
                ),
                TrackSpec(
                    "BRISC",
                    [
                        Segment(
                            id="theory_sender_b_qpull",
                            label="pull Q once",
                            section_id="q_prep",
                            start_frac=0.30,
                            end_frac=0.80,
                            color=PALETTE["q"],
                        ),
                        Segment("theory_sender_b_m0", "mcast K0", "k_loop", 0.14, 0.18, PALETTE["mcast"]),
                        Segment("theory_sender_b_m1", "mcast K1", "k_loop", 0.28, 0.32, PALETTE["mcast"]),
                        Segment("theory_sender_b_m2", "mcast K2", "k_loop", 0.50, 0.54, PALETTE["mcast"]),
                        Segment(
                            id="theory_sender_b_send",
                            label="send local (m,l,O)",
                            section_id="tail",
                            start_frac=0.08,
                            end_frac=0.28,
                            color=PALETTE["output"],
                        ),
                    ],
                ),
            ],
        ),
        RoleSpec(
            label="Ordinary worker",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment("theory_worker_n_r0", "mcast-ready", "k_loop", 0.14, 0.18, PALETTE["k_wait"]),
                        Segment("theory_worker_n_r1", "mcast-ready", "k_loop", 0.28, 0.32, PALETTE["k_wait"]),
                        Segment("theory_worker_n_r2", "mcast-ready", "k_loop", 0.50, 0.54, PALETTE["k_wait"]),
                    ],
                ),
                TrackSpec(
                    "TRISC",
                    [
                        *make_cycle_cluster("K0", "k_loop", 0.18, 0.36, base_id="theory_worker_t_k0"),
                        *make_cycle_cluster("K1", "k_loop", 0.36, 0.54, base_id="theory_worker_t_k1"),
                        *make_cycle_cluster("K2", "k_loop", 0.54, 0.72, base_id="theory_worker_t_k2"),
                    ],
                ),
                TrackSpec(
                    "BRISC",
                    [
                        Segment(
                            id="theory_worker_b_qpull",
                            label="pull Q once",
                            section_id="q_prep",
                            start_frac=0.25,
                            end_frac=0.80,
                            color=PALETTE["q"],
                        ),
                        Segment(
                            id="theory_worker_b_send",
                            label="send local (m,l,O)",
                            section_id="tail",
                            start_frac=0.10,
                            end_frac=0.36,
                            color=PALETTE["output"],
                        ),
                    ],
                ),
            ],
        ),
        RoleSpec(
            label="Lane root（兼 worker）",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment(
                            id="theory_root_n_r0",
                            label="mcast-ready",
                            section_id="k_loop",
                            start_frac=0.14,
                            end_frac=0.18,
                            color=PALETTE["k_wait"],
                        ),
                        Segment(
                            id="theory_root_n_r1",
                            label="mcast-ready",
                            section_id="k_loop",
                            start_frac=0.28,
                            end_frac=0.32,
                            color=PALETTE["k_wait"],
                        ),
                        Segment(
                            id="theory_root_n_r2",
                            label="mcast-ready",
                            section_id="k_loop",
                            start_frac=0.50,
                            end_frac=0.54,
                            color=PALETTE["k_wait"],
                        ),
                        Segment(
                            id="theory_root_n_ghost",
                            label="",
                            section_id="tail",
                            start_frac=0.00,
                            end_frac=1.00,
                            color=PALETTE["ghost"],
                            style="ghost",
                            opacity=0.24,
                        ),
                    ],
                ),
                TrackSpec(
                    "TRISC",
                    [
                        *make_cycle_cluster("K0", "k_loop", 0.18, 0.36, base_id="theory_root_t_k0"),
                        *make_cycle_cluster("K1", "k_loop", 0.36, 0.54, base_id="theory_root_t_k1"),
                        *make_cycle_cluster("K2", "k_loop", 0.54, 0.72, base_id="theory_root_t_k2"),
                        Segment("theory_root_t_m0", "merge r0", "tail", 0.30, 0.46, PALETTE["qk"]),
                        Segment("theory_root_t_m1", "merge r1", "tail", 0.52, 0.68, PALETTE["qk"]),
                        Segment("theory_root_t_mt", "tail merge", "tail", 0.70, 0.82, PALETTE["pv"]),
                    ],
                ),
                TrackSpec(
                    "BRISC",
                    [
                        Segment(
                            id="theory_root_b_qpull",
                            label="pull Q once",
                            section_id="q_prep",
                            start_frac=0.25,
                            end_frac=0.75,
                            color=PALETTE["q"],
                        ),
                        Segment("theory_root_b_recv0", "wait child-ready r0", "tail", 0.10, 0.28, PALETTE["tree_wait"]),
                        Segment("theory_root_b_recv1", "wait child-ready r1", "tail", 0.34, 0.52, PALETTE["tree_wait"]),
                        Segment(
                            id="theory_root_b_out",
                            label="final output",
                            section_id="tail",
                            start_frac=0.82,
                            end_frac=0.95,
                            color=PALETTE["output"],
                        ),
                    ],
                ),
            ],
        ),
    ]

    arrows: list[ArrowSpec] = []

    callouts = [
        CalloutSpec(
            role_label="Column sender（兼 worker）",
            track_label="NCRISC",
            x_norm=0.54,
            text="ping-pong CB_K: compute K0 overlaps read K1; read K2 waits until K0 slot is freed",
            dy=-14,
        ),
        CalloutSpec(
            role_label="Ordinary worker",
            track_label="NCRISC",
            x_norm=0.52,
            text="after Q distribution, Q source / workers / lane root all wait the same mcast-ready semaphore",
            dy=-14,
        ),
        CalloutSpec(
            role_label="Lane root（兼 worker）",
            track_label="BRISC",
            x_norm=0.88,
            text="wait child-ready = wait reducer semaphore that child local (m,l,O) has arrived",
            dy=-14,
        ),
    ]

    footnotes = [
        f"Case={case}; seq_len={seq_len}; visible chunks = K0/K1/K2 + ellipsis; total logical K chunks C={num_chunks}.",
        "Receiver-side payload is directly written by sender BRISC multicast; the reserve/push bookkeeping for CB_K is omitted from the theoretical lanes for readability, and the drawn mcast-ready marker is aligned with mcast as one logical handoff window in the abstract timeline.",
        "Because CB_K is double-buffered, the theoretical overlap is current-chunk compute + next-chunk read; the sender does not read K2 before the K0 slot is freed.",
        "Theoretical view keeps only the current dataflow and minimizes exposed bubbles; it is not a different algorithm.",
    ]

    return DiagramSpec(
        title="理想化 Decode 稳态流水（最小空泡 / 最小同步）",
        subtitle="Q preparation once; read K_(t+1) || multicast K_t || compute K_t; tail reduction kept compact.",
        logic_note="`Q source / Column sender / Lane root` are logical roles; they may share physical cores with ordinary workers.",
        roles=roles,
        arrows=arrows,
        callouts=callouts,
        footnotes=footnotes,
    )


def build_actual_spec(phase: dict[str, Any], source: dict[str, Any], *, case: str) -> DiagramSpec:
    seq_len = int(phase.get("seq_len", 32768))
    num_chunks = int(math.ceil(seq_len / 128))

    reader_issue = float(phase["reader_issue_share_pct"])
    reader_reserve = float(phase["reader_reserve_share_pct"])
    reader_wait = max(0.0, 100.0 - reader_issue - reader_reserve)
    total_reader = max(reader_issue + reader_reserve + reader_wait, 1e-9)
    issue_frac = reader_issue / total_reader
    reserve_frac = reader_reserve / total_reader
    wait_frac = reader_wait / total_reader

    wait_front = float(phase["wait_front_share_in_bubble_pct"]) / 100.0
    reserve_back = float(phase["reserve_back_share_in_bubble_pct"]) / 100.0
    compute_frac = max(0.0, 1.0 - wait_front - reserve_back)

    # Actual view reuses the same semantic skeleton as theory, but pushes visible work later
    # and shortens useful islands to reflect decode_32k backpressure.
    ready_windows = [(0.48, 0.52), (0.62, 0.66), (0.76, 0.80)]
    worker_compute_windows = [(0.52, 0.62), (0.66, 0.76), (0.80, 0.90)]
    sender_compute_windows = [(0.48, 0.58), (0.62, 0.72), (0.76, 0.86)]

    def actual_trisc(prefix: str, windows: list[tuple[float, float]], wait_front_end: float, reserve_back_start: float) -> list[Segment]:
        return [
            Segment(f"{prefix}_wf", "wait_front", "k_loop", 0.00, wait_front_end, PALETTE["reserve"]),
            *make_cycle_cluster("K0", "k_loop", windows[0][0], windows[0][1], base_id=f"{prefix}_k0"),
            *make_cycle_cluster("K1", "k_loop", windows[1][0], windows[1][1], base_id=f"{prefix}_k1"),
            *make_cycle_cluster("K2", "k_loop", windows[2][0], windows[2][1], base_id=f"{prefix}_k2"),
            Segment(f"{prefix}_rb", "reserve_back", "k_loop", reserve_back_start, 1.00, PALETTE["k_wait"]),
        ]

    def actual_receiver_ready(prefix: str) -> list[Segment]:
        return [
            Segment(f"{prefix}_w0", "wait K0 ready", "k_loop", 0.00, ready_windows[0][0], PALETTE["reserve"]),
            Segment(f"{prefix}_r0", "mcast-ready", "k_loop", ready_windows[0][0], ready_windows[0][1], PALETTE["k_wait"]),
            Segment(f"{prefix}_w1", "wait K1 ready", "k_loop", ready_windows[0][1], ready_windows[1][0], PALETTE["reserve"]),
            Segment(f"{prefix}_r1", "mcast-ready", "k_loop", ready_windows[1][0], ready_windows[1][1], PALETTE["k_wait"]),
            Segment(f"{prefix}_w2", "wait K2 ready", "k_loop", ready_windows[1][1], ready_windows[2][0], PALETTE["reserve"]),
            Segment(f"{prefix}_r2", "mcast-ready", "k_loop", ready_windows[2][0], ready_windows[2][1], PALETTE["k_wait"]),
        ]

    roles = [
        RoleSpec(
            label="Q source（兼 worker）",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment("actual_qsource_n_q", "Q local / light", "q_prep", 0.20, 0.85, PALETTE["q"]),
                        *actual_receiver_ready("actual_qsource_n"),
                    ],
                ),
                TrackSpec("TRISC", actual_trisc("actual_qsource_t", worker_compute_windows, 0.52, 0.90)),
                TrackSpec(
                    "BRISC",
                    [
                        Segment("actual_qsource_b_pull", "serve Q pulls", "q_prep", 0.25, 0.95, PALETTE["q"]),
                        Segment("actual_qsource_b_wait", "wait local-ready", "tail", 0.06, 0.74, PALETTE["tree_wait"]),
                        Segment("actual_qsource_b_send", "send local (m,l,O)", "tail", 0.74, 0.86, PALETTE["output"]),
                    ],
                ),
            ],
        ),
        RoleSpec(
            label="Column sender（兼 worker）",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment("actual_sender_n_res0", "K reserve", "k_loop", 0.00, 0.18, PALETTE["reserve"]),
                        Segment("actual_sender_n_k0", "read K0", "k_loop", 0.18, 0.48, PALETTE["k_read"]),
                        Segment("actual_sender_n_res1", "", "k_loop", 0.48, 0.52, PALETTE["reserve"]),
                        Segment("actual_sender_n_k1", "read K1", "k_loop", 0.52, 0.62, PALETTE["k_read"]),
                        Segment("actual_sender_n_res2", "", "k_loop", 0.62, 0.66, PALETTE["reserve"]),
                        Segment("actual_sender_n_k2", "read K2", "k_loop", 0.66, 0.76, PALETTE["k_read"]),
                        Segment("actual_sender_n_wait", "K wait", "k_loop", 0.76, 1.00, PALETTE["k_wait"]),
                    ],
                ),
                TrackSpec("TRISC", actual_trisc("actual_sender_t", sender_compute_windows, 0.48, 0.86)),
                TrackSpec(
                    "BRISC",
                    [
                        Segment("actual_sender_b_qpull", "pull Q once", "q_prep", 0.30, 0.80, PALETTE["q"]),
                        Segment("actual_sender_b_m0", "mcast K0", "k_loop", 0.48, 0.52, PALETTE["mcast"]),
                        Segment("actual_sender_b_m1", "mcast K1", "k_loop", 0.62, 0.66, PALETTE["mcast"]),
                        Segment("actual_sender_b_m2", "mcast K2", "k_loop", 0.76, 0.80, PALETTE["mcast"]),
                        Segment("actual_sender_b_wait", "wait local-ready", "tail", 0.00, 0.70, PALETTE["tree_wait"]),
                        Segment("actual_sender_b_send", "send local (m,l,O)", "tail", 0.70, 0.88, PALETTE["output"]),
                    ],
                ),
            ],
        ),
        RoleSpec(
            label="Ordinary worker",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        *actual_receiver_ready("actual_worker_n"),
                    ],
                ),
                TrackSpec("TRISC", actual_trisc("actual_worker_t", worker_compute_windows, 0.52, 0.90)),
                TrackSpec(
                    "BRISC",
                    [
                        Segment("actual_worker_b_qpull", "pull Q once", "q_prep", 0.25, 0.80, PALETTE["q"]),
                        Segment("actual_worker_b_wait", "wait local-ready", "tail", 0.10, 0.82, PALETTE["tree_wait"]),
                        Segment("actual_worker_b_send", "send local (m,l,O)", "tail", 0.82, 0.94, PALETTE["output"]),
                    ],
                ),
            ],
        ),
        RoleSpec(
            label="Lane root（兼 worker）",
            tracks=[
                TrackSpec(
                    "NCRISC",
                    [
                        Segment("actual_root_n_light", "light / non-critical", "q_prep", 0.25, 0.70, PALETTE["ghost"], style="ghost"),
                        *actual_receiver_ready("actual_root_n"),
                    ],
                ),
                TrackSpec(
                    "TRISC",
                    [
                        *actual_trisc("actual_root_t", worker_compute_windows, 0.52, 0.90),
                        Segment("actual_root_t_m0", "merge r0", "tail", 0.38, 0.56, PALETTE["qk"]),
                        Segment("actual_root_t_m1", "merge r1", "tail", 0.78, 0.90, PALETTE["qk"]),
                        Segment("actual_root_t_mt", "tail merge", "tail", 0.90, 0.96, PALETTE["pv"]),
                    ],
                ),
                TrackSpec(
                    "BRISC",
                    [
                        Segment("actual_root_b_qpull", "pull Q once", "q_prep", 0.30, 0.75, PALETTE["q"]),
                        Segment("actual_root_b_wait0", "wait child-ready r0", "tail", 0.00, 0.30, PALETTE["tree_wait"]),
                        Segment("actual_root_b_read0", "read child r0", "tail", 0.30, 0.38, PALETTE["mcast"]),
                        Segment("actual_root_b_wait1", "wait child-ready r1", "tail", 0.38, 0.72, PALETTE["tree_wait"]),
                        Segment("actual_root_b_read1", "read child r1", "tail", 0.72, 0.78, PALETTE["mcast"]),
                        Segment("actual_root_b_out", "final output", "tail", 0.96, 1.00, PALETTE["output"]),
                    ],
                ),
            ],
        ),
    ]

    arrows: list[ArrowSpec] = []

    callouts = [
        CalloutSpec(
            role_label="Column sender（兼 worker）",
            track_label="NCRISC",
            x_norm=0.34,
            text="reader reserve stretches gaps before visible read K_t",
            dy=-12,
        ),
        CalloutSpec(
            role_label="Ordinary worker",
            track_label="TRISC",
            x_norm=0.56,
            text="same K-stage skeleton as theory, but front bubbles dominate",
            dy=-12,
        ),
        CalloutSpec(
            role_label="Lane root（兼 worker）",
            track_label="BRISC",
            x_norm=0.88,
            text="tree_child_wait dominates; child reads and final output stay short",
            dy=-12,
        ),
    ]

    footnotes = [
        (
            f"{case}: kernel={float(phase['kernel_us']):.2f} us; "
            f"reader reserve={fmt_pct(reader_reserve)}; "
            f"K reserve={fmt_pct(float(source['k_reserve_share_pct']))}; "
            f"writer cb_wait={fmt_pct(float(phase['writer_cb_wait_share_pct']))}."
        ),
        (
            "Anchors: wait_front="
            f"{fmt_pct(float(phase['wait_front_share_in_bubble_pct']))}, "
            "reserve_back="
            f"{fmt_pct(float(phase['reserve_back_share_in_bubble_pct']))}, "
            "sender/tree="
            f"{fmt_pct(float(source['sender_cb_wait_share_pct']))}/{fmt_pct(float(source['tree_child_wait_share_pct']))}, "
            "root/output≈0/0."
        ),
        "Actual view reuses the theoretical-stage skeleton; decode_32k counters appear as stretched wait/reserve gaps and delayed visible work islands.",
        "Receiver-side wait K_t / mcast-ready windows are schematic shared receive pressure, not direct per-role traces.",
        (
            "This is a calibrated schematic anchored by aggregate counters, not a per-core timestamp trace. "
            f"Visible K islands are only K0/K1/K2 out of C={num_chunks} logical chunks."
        ),
    ]

    return DiagramSpec(
        title="实际 Decode_32k 运行图（bubble / reserve / sender-tree wait）",
        subtitle=(
            "Same stage skeleton as theory, but decode_32k stalls stretch read/ready/compute/tail handoffs."
        ),
        logic_note="Same logical-role decomposition and same semantic phases as the theoretical view; only lengths, bubbles and overlap differ.",
        roles=roles,
        arrows=arrows,
        callouts=callouts,
        footnotes=footnotes,
    )


def draw_rect(parts: list[str], x: float, y: float, w: float, h: float, *, fill: str, stroke: str = TRACK_BORDER, rx: int = 6, opacity: float = 1.0) -> None:
    parts.append(
        f'  <rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 1.0):.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1" opacity="{opacity:.3f}"/>'
    )


def draw_text(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 11,
    weight: str = "400",
    color: str = FG,
    anchor: str = "start",
) -> None:
    parts.append(
        f'  <text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" dominant-baseline="middle" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{esc(text)}</text>'
    )


def draw_cycle_segment(
    parts: list[str],
    x0: float,
    x1: float,
    y: float,
    h: float,
    segment: Segment,
    *,
    note_drawn: set[str],
) -> None:
    w = max(x1 - x0, 1.0)
    draw_rect(parts, x0, y, w, h, fill=segment.color, rx=6)
    if segment.note and segment.note not in note_drawn:
        draw_text(parts, (x0 + x1) / 2.0, y - 6.0, segment.note, size=8, weight="700", color=SUB_FG, anchor="middle")
        note_drawn.add(segment.note)
    if w > 28:
        draw_text(parts, (x0 + x1) / 2.0, y + h / 2.0, segment.label, size=7, weight="600", color=FG, anchor="middle")


def draw_marker_defs(parts: list[str]) -> None:
    parts.extend(
        [
            "  <defs>",
            '    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            '      <path d="M0,0 L0,6 L9,3 z" fill="#88c0ff"/>',
            "    </marker>",
            "  </defs>",
        ]
    )


def make_legend(parts: list[str]) -> None:
    items = legend_items()
    x = LEFT_MARGIN
    y = LEGEND_Y
    draw_text(parts, x, y - 10, "Legend", size=11, weight="700", color=FG)
    cursor_x = x + 58
    for label, color in items:
        draw_rect(parts, cursor_x, y - 10, 14, 14, fill=color, stroke=BORDER, rx=4)
        draw_text(parts, cursor_x + 20, y, label, size=10, color=SUB_FG)
        cursor_x += 20 + len(label) * 5.6 + 24
    # TRISC cycle legend
    cycle_x = LEFT_MARGIN
    cycle_y = y + 22
    draw_text(parts, cycle_x, cycle_y, "TRISC cycle", size=10, color=SUB_FG)
    sx = cycle_x + 70
    draw_rect(parts, sx, cycle_y - 8, 30, 14, fill=PALETTE["qk"], stroke=BORDER, rx=4)
    draw_rect(parts, sx + 30, cycle_y - 8, 24, 14, fill=PALETTE["softmax"], stroke=BORDER, rx=4)
    draw_rect(parts, sx + 54, cycle_y - 8, 28, 14, fill=PALETTE["pv"], stroke=BORDER, rx=4)
    draw_text(parts, sx + 92, cycle_y, "QK^T / softmax / P@V", size=10, color=SUB_FG)


def render_svg(spec: DiagramSpec) -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    ]
    draw_marker_defs(parts)
    parts.append(f'  <rect width="100%" height="100%" rx="20" fill="{BG}"/>')

    draw_text(parts, LEFT_MARGIN, TITLE_Y, spec.title, size=24, weight="700")
    draw_text(parts, LEFT_MARGIN, SUBTITLE_Y, spec.subtitle, size=12, color=SUB_FG)
    draw_text(parts, LEFT_MARGIN, NOTE_Y, spec.logic_note, size=11, color=SUB_FG)

    make_legend(parts)

    # Section bands and boundaries.
    section_fill = ["#182131", "#131c28", "#182131"]
    for idx, (sid, (label, start, end)) in enumerate(SECTIONS.items()):
        xa = x_norm_to_px(start)
        xb = x_norm_to_px(end)
        draw_rect(parts, xa, SECTION_BAND_Y, xb - xa, SECTION_BAND_H, fill=section_fill[idx], stroke=BORDER, rx=10, opacity=0.95)
        draw_text(parts, (xa + xb) / 2.0, SECTION_BAND_Y + SECTION_BAND_H / 2.0, label, size=11, weight="700", anchor="middle")
        parts.append(
            f'  <line x1="{xa:.2f}" y1="{SECTION_BAND_Y + SECTION_BAND_H:.2f}" x2="{xa:.2f}" y2="{FOOTER_Y - 18:.2f}" stroke="{BORDER}" stroke-width="1"/>'
        )
    parts.append(
        f'  <line x1="{x_norm_to_px(1.0):.2f}" y1="{SECTION_BAND_Y + SECTION_BAND_H:.2f}" '
        f'x2="{x_norm_to_px(1.0):.2f}" y2="{FOOTER_Y - 18:.2f}" stroke="{BORDER}" stroke-width="1"/>'
    )

    segment_boxes: dict[str, tuple[float, float, float, float]] = {}
    role_idx_by_label = {role.label: idx for idx, role in enumerate(spec.roles)}

    for role_idx, role in enumerate(spec.roles):
        y0 = role_top(role_idx)
        draw_rect(parts, LEFT_MARGIN, y0, CHART_W, ROLE_ROW_H, fill=PANEL, stroke=BORDER, rx=14)
        draw_text(parts, 26, y0 + ROLE_ROW_H / 2.0, role.label, size=12, weight="700")

        for track_idx, track in enumerate(role.tracks):
            ty = track_top(role_idx, track_idx)
            draw_text(parts, LEFT_MARGIN - 10, ty + TRACK_H / 2.0, track.label, size=10, color=MUTED, anchor="end")

            note_drawn: set[str] = set()
            for seg in track.segments:
                xa = section_x(seg.section_id, seg.start_frac)
                xb = section_x(seg.section_id, seg.end_frac)
                w = xb - xa
                opacity = seg.opacity
                if seg.style == "ghost":
                    draw_rect(parts, xa, ty, w, TRACK_H, fill=seg.color, stroke=TRACK_BORDER, rx=6, opacity=opacity)
                elif seg.style == "cycle":
                    draw_cycle_segment(parts, xa, xb, ty, TRACK_H, seg, note_drawn=note_drawn)
                else:
                    draw_rect(parts, xa, ty, w, TRACK_H, fill=seg.color, stroke=TRACK_BORDER, rx=6, opacity=opacity)
                    if seg.label and w > 36 and not (
                        len(seg.label) > 14 and w < 90
                    ):
                        size = 8 if w < 70 else 9
                        draw_text(parts, xa + w / 2.0, ty + TRACK_H / 2.0, seg.label, size=size, weight="600", color=seg.text_color, anchor="middle")
                    elif seg.label and (
                        seg.label.startswith("mcast")
                        or "ready" in seg.label
                        or "CB_K" in seg.label
                        or seg.label.startswith("send local")
                        or seg.label.startswith("read child")
                        or seg.label.startswith("tail merge")
                        or seg.label.startswith("final output")
                    ):
                        draw_text(parts, xa + w / 2.0, ty - 6.0, seg.label, size=8, weight="700", color=SUB_FG, anchor="middle")
                if seg.id:
                    segment_boxes[seg.id] = (xa, ty, xb, ty + TRACK_H)

            if track.label == "TRISC":
                _, ks, ke = SECTIONS["k_loop"]
                for frac in [0.12, 0.40, 0.68, 0.86]:
                    xl = x_norm_to_px(ks + (ke - ks) * frac)
                    parts.append(
                        f'  <line x1="{xl:.2f}" y1="{ty:.2f}" x2="{xl:.2f}" y2="{ty + TRACK_H:.2f}" stroke="#f5f7fb18" stroke-width="1"/>'
                    )
                draw_text(parts, x_norm_to_px(0.725), ty + TRACK_H / 2.0, "...", size=14, weight="700", color=SUB_FG, anchor="middle")

    # Arrows.
    for arrow in spec.arrows:
        if arrow.source_id not in segment_boxes or arrow.target_id not in segment_boxes:
            continue
        sx0, sy0, sx1, sy1 = segment_boxes[arrow.source_id]
        tx0, ty0, tx1, ty1 = segment_boxes[arrow.target_id]
        x1 = (sx0 + sx1) / 2.0
        y1 = (sy0 + sy1) / 2.0
        x2 = (tx0 + tx1) / 2.0
        y2 = (ty0 + ty1) / 2.0
        cx = (x1 + x2) / 2.0
        cy = min(y1, y2) - 18.0 - arrow.bend
        dash = ' stroke-dasharray="5 4"' if arrow.dashed else ""
        parts.append(
            f'  <path d="M{x1:.2f},{y1:.2f} Q{cx:.2f},{cy:.2f} {x2:.2f},{y2:.2f}" '
            f'stroke="{arrow.color}" stroke-width="2" fill="none" marker-end="url(#arrowhead)"{dash}/>'
        )
        lx = (x1 + x2) / 2.0
        ly = cy - 8.0
        draw_text(parts, lx, ly, arrow.label, size=9, weight="600", color=arrow.color, anchor="middle")

    # Callouts.
    for callout in spec.callouts:
        role_idx = role_idx_by_label[callout.role_label]
        track_idx = next(i for i, track in enumerate(spec.roles[role_idx].tracks) if track.label == callout.track_label)
        ty = track_top(role_idx, track_idx)
        draw_text(
            parts,
            x_norm_to_px(callout.x_norm),
            ty + callout.dy,
            callout.text,
            size=10,
            weight="600",
            color=callout.color,
            anchor=callout.anchor,
        )

    # Footer.
    for i, foot in enumerate(spec.footnotes):
        draw_text(parts, LEFT_MARGIN, FOOTER_Y + i * 16, foot, size=11, color=SUB_FG)

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--case", default=DEFAULT_CASE)
    ap.add_argument("--out-theory", type=Path, default=DEFAULT_OUT_THEORY)
    ap.add_argument("--out-actual", type=Path, default=DEFAULT_OUT_ACTUAL)
    args = ap.parse_args()

    data = load_json(args.json)
    phase = row_by_case(data["decode_phase_rows"], args.case)
    source = row_by_case(data["decode_source_rows"], args.case)

    seq_len = int(phase.get("seq_len", 32768))
    num_chunks = int(math.ceil(seq_len / 128))

    theory = build_theoretical_spec(args.case, seq_len, num_chunks)
    actual = build_actual_spec(phase, source, case=args.case)

    args.out_theory.parent.mkdir(parents=True, exist_ok=True)
    args.out_theory.write_text(render_svg(theory), encoding="utf-8")
    args.out_actual.write_text(render_svg(actual), encoding="utf-8")

    print(f"Wrote {args.out_theory}")
    print(f"Wrote {args.out_actual}")


if __name__ == "__main__":
    main()
