#!/usr/bin/env python3
"""
Build decode lane-column *runtime* swimlanes (single global wall-clock axis).

Uses Part II aggregate JSON (decode_32k): kernel_us, phase shares, reader/writer
source shares, and TT corrected-runtime PM FPU from A3__tt when present.

Important limitation (also written into the SVG footnotes):
  Device profiling here is grid-aggregated + component counters — there is still
  no per-core instruction timestamp trace.  This figure is therefore a
  *calibrated pipeline schematic*: absolute kernel length matches measured
  `kernel_us`, while sub-interval widths map aggregate shares onto Section 4
  roles.  Q source / column sender / lane root are **workers with extra duties**:
  inter-core = upper 兼任 band + lower peer TRISC; intra-core = same TRISC row
  for all four + role-tinted NCRISC/BRISC overlays.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "outputs" / "raw" / "part2_utilization_bottleneck_analysis.json"
DEFAULT_OUT_INTER = ROOT / "outputs" / "visuals" / "decode_lane_column_roles_inter_core_runtime.svg"
DEFAULT_OUT_INTRA = ROOT / "outputs" / "visuals" / "decode_lane_column_roles_intra_core_runtime.svg"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def row_by_case(rows: list[dict[str, Any]], case: str) -> dict[str, Any]:
    for r in rows:
        if r.get("case") == case:
            return r
    raise KeyError(case)


def pm_fpu_tt_a3(rows: list[dict[str, Any]]) -> float | None:
    for r in rows:
        if r.get("path_id") == "tt_corrected_runtime_a3_tracy_perf_fpu_sync":
            v = r.get("first_pm_fpu_util_pct")
            if v is not None:
                return float(v)
    return None


def peer_trisc_segments_fmha(
    T_us: float,
    tr: float,
    wf_b: float,
    rb_b: float,
    pm_fpu_pct: float,
    *,
    seq_len: int,
    ck: int = 128,
) -> tuple[list[tuple[str, float, str]], dict[str, Any]]:
    """
    Peer TRISC timeline: wait-front, then many logical K-chunk cycles each split
    QK^T / online softmax / P@V (illustrative proportions inside measured PM math
    window), then reserve-back, TRISC idle, non-TRISC kernel window.
    """
    bubble_tr = max(0.0, 1.0 - pm_fpu_pct / 100.0)
    tr_wf = T_us * tr * bubble_tr * wf_b
    tr_act = T_us * tr * (pm_fpu_pct / 100.0)
    tr_rb = T_us * tr * bubble_tr * rb_b
    tr_sum = tr_wf + tr_act + tr_rb
    tr_cap = T_us * tr
    if tr_sum > tr_cap > 0:
        s = tr_cap / tr_sum
        tr_wf *= s
        tr_act *= s
        tr_rb *= s
    tr_idle = max(0.0, tr_cap - tr_wf - tr_act - tr_rb)
    tr_other = max(0.0, T_us - tr_cap)

    num_chunks = max(1, (int(seq_len) + ck - 1) // ck)
    # Visual cycles inside tr_act (each cycle ≈ ceil(C/n_vis) logical chunks in footnote).
    n_vis = max(6, min(24, num_chunks))
    chunks_per_vis = int(math.ceil(num_chunks / n_vis)) if num_chunks > 0 else 1

    segs: list[tuple[str, float, str]] = [
        ("peer: 等 K_t（wait-front / CB）", tr_wf, "#ff595e"),
    ]
    if tr_act > 1e-9 and n_vis > 0:
        cycle = tr_act / float(n_vis)
        fqk, fsm, fpv = 0.42, 0.26, 0.32
        for j in range(n_vis):
            qk = cycle * fqk
            sm = cycle * fsm
            pv = cycle * fpv
            lo = j * chunks_per_vis
            hi = min(num_chunks - 1, (j + 1) * chunks_per_vis - 1)
            tag = f"K[{lo}–{hi}] " if chunks_per_vis > 1 and j in (0, n_vis // 2, n_vis - 1) else ""
            if j == 0:
                segs.append((f"{tag}QK^T", qk, "#1b998b"))
                segs.append((f"{tag}online softmax", sm, "#ff9f1c"))
                segs.append((f"{tag}P@V（V⊂K）", pv, "#5c7cfa"))
            elif j == n_vis // 2:
                segs.append((f"…×{n_vis}轮 QK^T", qk, "#1b998b"))
                segs.append(("softmax", sm, "#ff9f1c"))
                segs.append(("P@V", pv, "#5c7cfa"))
            else:
                segs.append(("QK^T", qk, "#1b998b"))
                segs.append(("softmax", sm, "#ff9f1c"))
                segs.append(("P@V", pv, "#5c7cfa"))

    segs.extend(
        [
            ("peer: reserve-back（chunk 间/下游）", tr_rb, "#ffca3a"),
            ("peer: TRISC idle", tr_idle, "#3a4352"),
            ("kernel 非 TRISC 附着窗", tr_other, "#1a2230"),
        ]
    )

    meta = {
        "tr_wf": tr_wf,
        "tr_act": tr_act,
        "tr_rb": tr_rb,
        "tr_idle": tr_idle,
        "tr_other": tr_other,
        "num_chunks": num_chunks,
        "ck": ck,
        "seq_len": int(seq_len),
        "n_vis": n_vis,
        "chunks_per_vis": chunks_per_vis,
    }
    return segs, meta


def _strip_peer_prefix(segs: list[tuple[str, float, str]]) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for n, w, c in segs:
        if n.startswith("peer: "):
            out.append((n[6:], w, c))
        else:
            out.append((n, w, c))
    return out


def build_inter_core_svg(
    *,
    T_us: float,
    phase: dict[str, Any],
    source: dict[str, Any],
    pm_fpu_pct: float,
) -> str:
    """Four worker rows on 0..T_us; Q / column sender / lane root = 兼任 + peer TRISC."""
    nc = float(phase["ncrisc_share_pct"]) / 100.0
    br = float(phase["brisc_share_pct"]) / 100.0
    tr = float(phase["compute_share_pct"]) / 100.0
    w_cb = float(phase["writer_cb_wait_share_pct"]) / 100.0
    wf_b = float(phase["wait_front_share_in_bubble_pct"]) / 100.0
    rb_b = float(phase["reserve_back_share_in_bubble_pct"]) / 100.0

    seq_len = int(phase.get("seq_len", 32768))
    ck = 128
    peer_segs, tr_meta = peer_trisc_segments_fmha(
        T_us, tr, wf_b, rb_b, pm_fpu_pct, seq_len=seq_len, ck=ck
    )

    k_iss = float(source["k_issue_share_pct"]) + float(source["v_issue_share_pct"])
    k_wait = float(source["k_wait_share_pct"]) + float(source["v_wait_share_pct"])
    k_res = float(source["k_reserve_share_pct"]) + float(source["v_reserve_share_pct"])
    ro = float(source["reader_other_share_pct"])
    sender_cb = float(source["sender_cb_wait_share_pct"]) / 100.0
    tree_cb = float(source["tree_child_wait_share_pct"]) / 100.0

    # Q source: map reader_other (non K/V marker share) onto an upper-bound Q-prep window.
    w_q = min(T_us * nc * (ro / 100.0), 0.08 * T_us)
    w_q = max(w_q, 0.012 * T_us)

    # Column sender NCRISC K pipeline (when NCRISC is attached).
    w_issue = T_us * nc * (k_iss / 100.0)
    w_wait = T_us * nc * (k_wait / 100.0)
    w_res = T_us * nc * (k_res / 100.0)
    s_nc = w_issue + w_wait + w_res
    scale_nc = (T_us * nc) / s_nc if s_nc > 0 else 1.0
    w_issue *= scale_nc
    w_wait *= scale_nc
    w_res *= scale_nc

    br_mcast = max(0.0, min(T_us * br * 0.1, T_us * br * (1.0 - w_cb * sender_cb)))
    br_sender_wait = T_us * br * w_cb * sender_cb
    br_rem = max(0.0, T_us * br - br_mcast - br_sender_wait)

    # Lane root: algorithmic tail (tree-heavy) on a bounded end window.
    t_reduce = max(0.07 * T_us, min(0.22 * T_us, T_us * br * w_cb * 0.55))
    t0 = T_us - t_reduce
    br_tree = t_reduce * 0.62
    br_merge = t_reduce * 0.23
    br_write = t_reduce * 0.15

    width = 1280
    margin_l = 200
    margin_r = 40
    chart_w = width - margin_l - margin_r
    x0 = margin_l
    lane_h = 98
    gap = 12
    title_y = 34
    sub_y = 54
    top_y = 100
    axis_y = top_y - 10
    svg_h = 560

    def x_of(t: float) -> float:
        return x0 + (t / T_us) * chart_w if T_us > 0 else x0

    def draw_segments(y: float, segs: list[tuple[str, float, str]], y0_off: float, h: float) -> None:
        acc = 0.0
        for name, w, color in segs:
            if w <= 0:
                continue
            w = min(w, max(0.0, T_us - acc))
            if w <= 0:
                break
            xa = x_of(acc)
            xb = x_of(acc + w)
            rw = max(xb - xa, 1.0)
            parts.append(
                f'  <rect x="{xa:.2f}" y="{y + y0_off:.2f}" width="{rw:.2f}" height="{h:.2f}" rx="8" fill="{color}" stroke="#223046" stroke-width="1"/>'
            )
            fs = "8" if rw < 100 else "9"
            if rw > 40 and name:
                parts.append(
                    f'  <text x="{xa + rw/2:.2f}" y="{y + y0_off + h/2:.2f}" text-anchor="middle" dominant-baseline="middle" '
                    f'font-size="{fs}" font-weight="600" fill="#f5f7fb">{esc(name)}</text>'
                )
            acc += w
        if acc < T_us - 1e-6:
            xa = x_of(acc)
            rw = max(x_of(T_us) - xa, 0.0)
            if rw > 1:
                parts.append(
                    f'  <rect x="{xa:.2f}" y="{y + y0_off:.2f}" width="{rw:.2f}" height="{h:.2f}" rx="8" fill="#2a3340" opacity="0.55"/>'
                )

    def draw_trisc_chunk_ticks(y_row: float, y0_off: float, h: float) -> None:
        twf = float(tr_meta["tr_wf"])
        tact = float(tr_meta["tr_act"])
        nv = int(tr_meta["n_vis"])
        if nv <= 1 or tact <= 1e-9:
            return
        for i in range(1, nv):
            tx = twf + tact * (float(i) / float(nv))
            xl = x_of(tx)
            parts.append(
                f'  <line x1="{xl:.2f}" y1="{y_row + y0_off:.2f}" x2="{xl:.2f}" y2="{y_row + y0_off + h:.2f}" '
                f'stroke="#f5f7fb22" stroke-width="1.2"/>'
            )

    def dual_role_row(
        y: float,
        left_title: str,
        top_segs: list[tuple[str, float, str]],
        top_split_nc_br: tuple[list[tuple[str, float, str]], list[tuple[str, float, str]]] | None,
    ) -> None:
        """Upper band = 兼任; lower band = same peer TRISC as ordinary worker."""
        inner = lane_h - 18
        h_top = inner * 0.38
        h_bot = inner * 0.62
        y_in = y + 8
        parts.append(
            f'  <rect x="{x0:.1f}" y="{y:.1f}" width="{chart_w:.1f}" height="{lane_h:.1f}" rx="14" fill="#0f141d" stroke="#223046"/>'
        )
        parts.append(
            f'  <text x="22" y="{y + lane_h/2:.1f}" dominant-baseline="middle" font-size="11" font-weight="700" fill="#f5f7fb">{esc(left_title)}</text>'
        )
        if top_split_nc_br is not None:
            segs_nc, segs_br = top_split_nc_br
            h_nc = h_top * 0.52
            h_br = h_top * 0.46
            draw_segments(y, segs_nc, y_in, h_nc)
            draw_segments(y, segs_br, y_in + h_nc + 3, h_br)
        else:
            draw_segments(y, top_segs, y_in, h_top)
        draw_segments(y, peer_segs, y_in + h_top + 8, h_bot)
        draw_trisc_chunk_ticks(y, y_in + h_top + 8, h_bot)

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}" viewBox="0 0 {width} {svg_h}">')
    parts.append('  <rect width="100%" height="100%" rx="20" fill="#111722"/>')
    parts.append(
        f'  <text x="{margin_l}" y="{title_y}" font-size="21" font-weight="700" fill="#f5f7fb">{esc("核间：四类 worker（decode_32k，墙钟 μs）")}</text>'
    )
    _sub = (
        f"S={seq_len}, c_k={ck} → C={tr_meta['num_chunks']} 个 K chunk；"
        "下层 TRISC 内为 QK^T / softmax / P@V 循环示意，竖线≈逻辑 chunk 边界。"
    )
    parts.append(f'  <text x="{margin_l}" y="{sub_y}" font-size="11" fill="#9fb0c3">{esc(_sub)}</text>')

    for frac, lab in [(0, "0"), (0.25, f"{0.25*T_us:.0f}"), (0.5, f"{0.5*T_us:.0f}"), (0.75, f"{0.75*T_us:.0f}"), (1.0, f"{T_us:.0f} μs")]:
        xa = x0 + frac * chart_w
        parts.append(f'  <line x1="{xa:.1f}" y1="{axis_y}" x2="{xa:.1f}" y2="{svg_h - 72:.1f}" stroke="#223046" stroke-width="1"/>')
        parts.append(
            f'  <text x="{xa:.1f}" y="{axis_y - 2}" text-anchor="middle" font-size="11" fill="#9fb0c3">{esc(lab)}</text>'
        )

    y = float(top_y)
    dual_role_row(
        y,
        "Q source（兼 worker）",
        [("Q 物化 + lane pull", w_q, "#2ec4b6"), ("兼任条 idle", max(0.0, T_us - w_q), "#2a3340")],
        None,
    )
    y += lane_h + gap

    dual_role_row(
        y,
        "Column sender（兼 worker）",
        [],
        (
            [
                ("DRAM→CB: K page issue", w_issue, "#1982c4"),
                ("DRAM read 完成等待", w_wait, "#ffca3a"),
                ("reserve（下游反压）", w_res, "#ff595e"),
            ],
            [
                ("组播 K_t 到列", br_mcast, "#3d5a80"),
                ("sender local-ready wait", br_sender_wait, "#9b5de5"),
                ("其它 BRISC", br_rem, "#00bbf9"),
            ],
        ),
    )
    y += lane_h + gap

    # Ordinary worker — full-height peer TRISC only
    parts.append(
        f'  <rect x="{x0:.1f}" y="{y:.1f}" width="{chart_w:.1f}" height="{lane_h:.1f}" rx="14" fill="#0f141d" stroke="#223046"/>'
    )
    parts.append(
        f'  <text x="22" y="{y + lane_h/2:.1f}" dominant-baseline="middle" font-size="11" font-weight="700" fill="#f5f7fb">{esc("Ordinary worker（无额外界面）")}</text>'
    )
    draw_segments(y, peer_segs, 10, lane_h - 20)
    draw_trisc_chunk_ticks(y, 10, lane_h - 20)
    y += lane_h + gap

    lr_top_segs = [
        ("chunk 段：partial 上送（多 K_t）", t0, "#3a4352"),
        ("lane tree wait", br_tree, "#1982c4"),
        ("merge partials", br_merge, "#ff595e"),
        ("normalize + writeback", br_write, "#2ec4b6"),
    ]
    dual_role_row(y, "Lane root（兼 worker）", lr_top_segs, None)
    y += lane_h + gap

    foot = (
        f"kernel_us={T_us:.3f} μs; PM FPU={pm_fpu_pct:.2f}% (TT A3). "
        f"C={tr_meta['num_chunks']}（S={seq_len}, c_k={ck}）；图示 n_vis={tr_meta['n_vis']} 轮，"
        f"每轮≈{tr_meta['chunks_per_vis']} 个逻辑 chunk；QK/softmax/P@V 宽度为 PM math 窗内示意拆分。"
    )
    parts.append(f'  <text x="{margin_l}" y="{svg_h - 36:.1f}" font-size="11" fill="#9fb0c3">{esc(foot)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_intra_core_svg(
    *,
    T_us: float,
    phase: dict[str, Any],
    source: dict[str, Any],
    pm_fpu_pct: float,
) -> str:
    """Four worker rows × NCRISC/TRISC/BRISC; TRISC identical peer path for all roles."""
    nc = float(phase["ncrisc_share_pct"]) / 100.0
    br = float(phase["brisc_share_pct"]) / 100.0
    tr = float(phase["compute_share_pct"]) / 100.0
    w_cb = float(phase["writer_cb_wait_share_pct"]) / 100.0
    wf_b = float(phase["wait_front_share_in_bubble_pct"]) / 100.0
    rb_b = float(phase["reserve_back_share_in_bubble_pct"]) / 100.0
    ro = float(source["reader_other_share_pct"])
    k_iss = float(source["k_issue_share_pct"]) + float(source["v_issue_share_pct"])
    k_wait = float(source["k_wait_share_pct"]) + float(source["v_wait_share_pct"])
    k_res = float(source["k_reserve_share_pct"]) + float(source["v_reserve_share_pct"])
    sender_cb = float(source["sender_cb_wait_share_pct"]) / 100.0
    tree_cb = float(source["tree_child_wait_share_pct"]) / 100.0

    seq_len = int(phase.get("seq_len", 32768))
    ck = 128
    peer_tr_segs, tr_meta = peer_trisc_segments_fmha(
        T_us, tr, wf_b, rb_b, pm_fpu_pct, seq_len=seq_len, ck=ck
    )

    w_q = min(T_us * nc * (ro / 100.0), 0.08 * T_us)
    w_q = max(w_q, 0.012 * T_us)
    w_issue = T_us * nc * (k_iss / 100.0)
    w_wait = T_us * nc * (k_wait / 100.0)
    w_res = T_us * nc * (k_res / 100.0)
    s_nc = w_issue + w_wait + w_res
    scale_nc = (T_us * nc) / s_nc if s_nc > 0 else 1.0
    w_issue *= scale_nc
    w_wait *= scale_nc
    w_res *= scale_nc

    br_stream = T_us * br * w_cb * sender_cb
    br_emit = T_us * br * 0.45
    br_tree_w = T_us * br * 0.45
    br_idle_w = max(0.0, T_us * br - br_emit - br_tree_w)

    t_reduce_lr = max(0.07 * T_us, min(0.22 * T_us, T_us * br * w_cb * 0.55))
    lr_pre = T_us - t_reduce_lr
    lr_tree = t_reduce_lr * 0.62
    lr_merge = t_reduce_lr * 0.23
    lr_wb = t_reduce_lr * 0.15

    w_q_nc = w_q * 0.35
    rem_nc = max(0.0, T_us * nc - w_q_nc)
    w_q_br = w_q * 0.65
    rem_br = max(0.0, T_us * br - w_q_br)

    width = 1320
    margin_l = 210
    margin_r = 36
    chart_w = width - margin_l - margin_r
    x0 = margin_l
    row_h = 112
    track_h = 28
    gap = 16
    title_y = 32
    sub_y = 52
    y0 = 76

    def x_of(t: float) -> float:
        return x0 + (t / T_us) * chart_w if T_us > 0 else x0

    def track_row(y_base: float, t_idx: int) -> float:
        return y_base + 8 + t_idx * (track_h + 4)

    trisc_peer = _strip_peer_prefix(peer_tr_segs)

    tracks_per_role: list[tuple[str, list[tuple[str, list[tuple[str, float, str]]]]]] = [
        (
            "Q source（兼 worker）",
            [
                (
                    "NCRISC",
                    [
                        ("兼任: Q 相关", w_q_nc, "#1982c4"),
                        ("peer: recv/其它", rem_nc * 0.85, "#ff595e"),
                        ("idle", max(0.0, rem_nc * 0.15), "#2a3340"),
                    ],
                ),
                ("TRISC", trisc_peer),
                (
                    "BRISC",
                    [
                        ("兼任: Q pull", w_q_br, "#2ec4b6"),
                        ("peer: emit", rem_br * 0.5, "#00bbf9"),
                        ("peer: tree wait", rem_br * 0.4, "#9b5de5"),
                        ("idle", max(0.0, rem_br * 0.1), "#2a3340"),
                    ],
                ),
            ],
        ),
        (
            "Column sender（兼 worker）",
            [
                (
                    "NCRISC",
                    [
                        ("兼任: DRAM→CB K page issue", w_issue, "#1982c4"),
                        ("兼任: DRAM read wait", w_wait, "#ffca3a"),
                        ("兼任: reserve（反压）", w_res, "#ff595e"),
                        ("idle", max(0.0, T_us * nc - w_issue - w_wait - w_res), "#2a3340"),
                    ],
                ),
                ("TRISC", trisc_peer),
                (
                    "BRISC",
                    [
                        ("兼任: multicast", T_us * br * 0.1, "#3d5a80"),
                        ("兼任: sender wait", br_stream, "#9b5de5"),
                        (
                            "peer: emit + sync",
                            max(0.0, T_us * br - T_us * br * 0.1 - br_stream),
                            "#00bbf9",
                        ),
                    ],
                ),
            ],
        ),
        (
            "Ordinary worker",
            [
                (
                    "NCRISC",
                    [
                        ("idle", T_us * nc * 0.15, "#2a3340"),
                        ("等 K_t：recv/reserve", T_us * nc * 0.85, "#ff595e"),
                    ],
                ),
                ("TRISC", trisc_peer),
                (
                    "BRISC",
                    [
                        ("emit partial（每 K_t）", br_emit, "#00bbf9"),
                        ("tree / sync wait", br_tree_w, "#9b5de5"),
                        ("idle", br_idle_w, "#2a3340"),
                    ],
                ),
            ],
        ),
        (
            "Lane root（兼 worker）",
            [
                ("NCRISC", [("idle", T_us * nc, "#2a3340")]),
                ("TRISC", trisc_peer),
                (
                    "BRISC",
                    [
                        ("peer: child emit", lr_pre * 0.55, "#3d5a80"),
                        ("兼任: idle", lr_pre * 0.45, "#2a3340"),
                        ("兼任: tree", lr_tree, "#1982c4"),
                        ("兼任: merge", lr_merge, "#ff595e"),
                        ("兼任: writeback", lr_wb, "#2ec4b6"),
                    ],
                ),
            ],
        ),
    ]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="620" viewBox="0 0 {width} 620">')
    parts.append('  <rect width="100%" height="100%" rx="20" fill="#111722"/>')
    parts.append(
        f'  <text x="{margin_l}" y="{title_y}" font-size="21" font-weight="700" fill="#f5f7fb">{esc("核内：四类 worker × R/W/C（decode_32k，墙钟 μs）")}</text>'
    )
    _sub2 = (
        f"TRISC：每轮 K_t 为 QK^T→online softmax→P@V（C={tr_meta['num_chunks']}, c_k={ck}）；"
        f"竖线≈ n_vis={tr_meta['n_vis']} 视觉分段。"
    )
    parts.append(f'  <text x="{margin_l}" y="{sub_y}" font-size="11" fill="#9fb0c3">{esc(_sub2)}</text>')

    for frac, lab in [(0, "0"), (0.5, f"{0.5*T_us:.0f}"), (1.0, f"{T_us:.0f} μs")]:
        xa = x0 + frac * chart_w
        parts.append(f'  <line x1="{xa:.1f}" y1="72" x2="{xa:.1f}" y2="580" stroke="#223046" stroke-width="1"/>')
        parts.append(
            f'  <text x="{xa:.1f}" y="68" text-anchor="middle" font-size="11" fill="#9fb0c3">{esc(lab)}</text>'
        )

    yb = float(y0)
    for role_label, tracks in tracks_per_role:
        parts.append(
            f'  <rect x="{x0:.1f}" y="{yb:.1f}" width="{chart_w:.1f}" height="{row_h:.1f}" rx="14" fill="#0f141d" stroke="#223046"/>'
        )
        parts.append(
            f'  <text x="24" y="{yb + row_h/2:.1f}" dominant-baseline="middle" font-size="12" font-weight="700" fill="#f5f7fb">{esc(role_label)}</text>'
        )
        for ti, (tname, segs) in enumerate(tracks):
            yy = track_row(yb, ti)
            parts.append(
                f'  <text x="{margin_l - 8}" y="{yy + track_h/2:.1f}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#8fa3b8">{esc(tname)}</text>'
            )
            cx = x0
            acc = 0.0
            for name, w, color in segs:
                if w <= 0:
                    continue
                w = min(w, T_us - acc)
                if w <= 0:
                    break
                xa = x_of(acc)
                xb = x_of(acc + w)
                rw = max(xb - xa, 1.0)
                parts.append(
                    f'  <rect x="{xa:.2f}" y="{yy:.2f}" width="{rw:.2f}" height="{track_h:.2f}" rx="6" fill="{color}" stroke="#1c2433" stroke-width="1"/>'
                )
                fs = "8" if rw < 90 else "9"
                if rw > 20 and name and name not in ("idle", "—", "idle / light", "pre / idle"):
                    parts.append(
                        f'  <text x="{xa + rw/2:.2f}" y="{yy + track_h/2:.2f}" text-anchor="middle" dominant-baseline="middle" '
                        f'font-size="{fs}" fill="#f5f7fb">{esc(name)}</text>'
                    )
                acc += w
            if acc < T_us - 1e-6:
                xa = x_of(acc)
                xb = x_of(T_us)
                rw = max(xb - xa, 0)
                if rw > 1:
                    parts.append(
                        f'  <rect x="{xa:.2f}" y="{yy:.2f}" width="{rw:.2f}" height="{track_h:.2f}" rx="6" fill="#2a3340" opacity="0.55"/>'
                    )
            if tname == "TRISC":
                twf = float(tr_meta["tr_wf"])
                tact = float(tr_meta["tr_act"])
                nv = int(tr_meta["n_vis"])
                if nv > 1 and tact > 1e-9:
                    for ii in range(1, nv):
                        tx = twf + tact * (float(ii) / float(nv))
                        xl = x_of(tx)
                        parts.append(
                            f'  <line x1="{xl:.2f}" y1="{yy:.2f}" x2="{xl:.2f}" y2="{yy + track_h:.2f}" '
                            f'stroke="#f5f7fb18" stroke-width="1"/>'
                        )
        yb += row_h + gap

    _foot = (
        f"示意：C={tr_meta['num_chunks']} K chunk；TRISC 内 QK/softmax/P@V 为 PM math 窗示意拆分；无 per-core trace。"
    )
    parts.append(f'  <text x="{margin_l}" y="602" font-size="11" fill="#9fb0c3">{esc(_foot)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--case", default="decode_32k")
    ap.add_argument("--out-inter", type=Path, default=DEFAULT_OUT_INTER)
    ap.add_argument("--out-intra", type=Path, default=DEFAULT_OUT_INTRA)
    args = ap.parse_args()

    data = load_json(args.json)
    phase = row_by_case(data["decode_phase_rows"], args.case)
    source = row_by_case(data["decode_source_rows"], args.case)
    T_us = float(phase["kernel_us"])
    pm = pm_fpu_tt_a3(data.get("decode_pm_path_rows", []))
    if pm is None:
        pm = 25.61

    args.out_inter.parent.mkdir(parents=True, exist_ok=True)
    args.out_inter.write_text(build_inter_core_svg(T_us=T_us, phase=phase, source=source, pm_fpu_pct=pm), encoding="utf-8")
    args.out_intra.write_text(build_intra_core_svg(T_us=T_us, phase=phase, source=source, pm_fpu_pct=pm), encoding="utf-8")
    print(f"Wrote {args.out_inter}")
    print(f"Wrote {args.out_intra}")


if __name__ == "__main__":
    main()
