#!/usr/bin/env python3
"""Generate Wormhole SF-MLA runtime diagrams in two styles.

Outputs:
- docs/assets/images/wh_sf_mla_runtime_diagram.png/.svg                 : 4-core floorplan style (default main figure)
- docs/assets/images/wh_sf_mla_runtime_diagram_floorplan.png/.svg       : 4-core floorplan style
- docs/assets/images/wh_sf_mla_runtime_diagram_logical_overlay.png/.svg : 4-core logical->physical overlay style
- docs/assets/images/wh_sf_mla_runtime_diagram_8core.png/.svg           : 8-core floorplan style
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MONO = "DejaVu Sans Mono"

GRID_X = 10
GRID_Y = 12
TILE_W = 1.0
TILE_H = 1.0
GAP = 0.14

FULL_LOGICAL_X_TO_PHYSICAL_X = [1, 2, 3, 4, 6, 7, 8, 9]
FULL_LOGICAL_Y_TO_PHYSICAL_Y = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
CURRENT_LOGICAL_X_TO_PHYSICAL_X = [1, 2, 3, 4, 6, 7, 8, 9]
CURRENT_LOGICAL_Y_TO_PHYSICAL_Y = [1, 2, 3, 4, 5, 7, 8]  # current tt-metal compute_with_storage view used in Part1


def gx(x):
    return x * (TILE_W + GAP)


def gy(y):
    return (GRID_Y - 1 - y) * (TILE_H + GAP)


def gcx(x):
    return gx(x) + TILE_W / 2


def gcy(y):
    return gy(y) + TILE_H / 2


WORKERS = [
    (1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1),
    (1, 2), (2, 2), (3, 2), (4, 2), (6, 2), (7, 2), (8, 2), (9, 2),
    (1, 3), (2, 3), (3, 3), (4, 3), (6, 3), (7, 3), (8, 3), (9, 3),
    (1, 4), (2, 4), (3, 4), (4, 4), (6, 4), (7, 4), (8, 4), (9, 4),
    (1, 5), (2, 5), (3, 5), (4, 5), (6, 5), (7, 5), (8, 5), (9, 5),
    (1, 7), (2, 7), (3, 7), (4, 7), (6, 7), (7, 7), (8, 7), (9, 7),
    (1, 8), (2, 8), (3, 8), (4, 8), (6, 8), (7, 8), (8, 8), (9, 8),
    (1, 9), (2, 9), (3, 9), (4, 9), (6, 9), (7, 9), (8, 9), (9, 9),
    (1, 10), (2, 10), (3, 10), (4, 10), (6, 10), (7, 10), (8, 10), (9, 10),
    (1, 11), (2, 11), (3, 11), (4, 11), (6, 11), (7, 11), (8, 11), (9, 11),
]

DRAM_CHANNELS = {
    0: [(0, 0), (0, 1), (0, 11)],
    1: [(0, 5), (0, 6), (0, 7)],
    2: [(5, 0), (5, 1), (5, 11)],
    3: [(5, 2), (5, 9), (5, 10)],
    4: [(5, 3), (5, 4), (5, 8)],
    5: [(5, 5), (5, 6), (5, 7)],
}

ETH = [
    (9, 0), (1, 0), (8, 0), (2, 0), (7, 0), (3, 0), (6, 0), (4, 0),
    (9, 6), (1, 6), (8, 6), (2, 6), (7, 6), (3, 6), (6, 6), (4, 6),
]
ROUTERS = [(0, 2), (0, 4), (0, 8), (0, 9)]
PCIE = [(0, 3)]
ARC = [(0, 10)]

# From wormhole_b0_80_arch.yaml dram_views, expressed in NOC0 physical coordinates.
DRAM_VIEWS_NOC0 = {
    0: (0, 11),
    1: (0, 1),
    2: (0, 5),
    3: (0, 7),
    4: (5, 1),
    5: (5, 11),
    6: (5, 2),
    7: (5, 9),
    8: (5, 8),
    9: (5, 3),
    10: (5, 7),
    11: (5, 5),
}

# WH FlashMLA 4-core S-blocks, converted from logical 8x7 to physical NOC0 coordinates.
SBLOCKS_4C = [
    {"name": "S1", "color": "#e74c3c", "dram_view": 1, "cores": [(1, 1), (2, 1), (1, 2), (2, 2)]},
    {"name": "S2", "color": "#e67e22", "dram_view": 2, "cores": [(1, 4), (2, 4), (1, 5), (2, 5)]},
    {"name": "S3", "color": "#f1c40f", "dram_view": 0, "cores": [(3, 7), (4, 7), (3, 8), (4, 8)]},
    {"name": "S4", "color": "#2ecc71", "dram_view": 4, "cores": [(6, 1), (7, 1), (6, 2), (7, 2)]},
    {"name": "S5", "color": "#3498db", "dram_view": 9, "cores": [(6, 3), (7, 3), (6, 4), (7, 4)]},
    {"name": "S6", "color": "#9b59b6", "dram_view": 8, "cores": [(6, 7), (7, 7), (6, 8), (7, 8)]},
]

# WH FlashMLA experimental 8-core S-blocks, converted from logical 8x7 to physical NOC0 coordinates.
SBLOCKS_8C = [
    {"name": "S1", "color": "#e74c3c", "dram_view": 1, "cores": [(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (2, 2), (3, 2), (4, 2)]},
    {"name": "S2", "color": "#e67e22", "dram_view": 2, "cores": [(1, 4), (2, 4), (3, 4), (4, 4), (1, 5), (2, 5), (3, 5), (4, 5)]},
    {"name": "S3", "color": "#f1c40f", "dram_view": 0, "cores": [(1, 7), (2, 7), (3, 7), (4, 7), (1, 8), (2, 8), (3, 8), (4, 8)]},
    {"name": "S4", "color": "#2ecc71", "dram_view": 4, "cores": [(6, 1), (7, 1), (8, 1), (9, 1), (6, 2), (7, 2), (8, 2), (9, 2)]},
    {"name": "S5", "color": "#3498db", "dram_view": 9, "cores": [(6, 3), (7, 3), (8, 3), (9, 3), (6, 4), (7, 4), (8, 4), (9, 4)]},
    {"name": "S6", "color": "#9b59b6", "dram_view": 8, "cores": [(6, 7), (7, 7), (8, 7), (9, 7), (6, 8), (7, 8), (8, 8), (9, 8)]},
]

BASE_FILL = {
    "worker": "#172033",
    "dram": "#201824",
    "eth": "#1d2434",
    "router": "#232933",
    "pcie": "#2b2117",
    "arc": "#1c2a22",
}
BASE_EDGE = {
    "worker": "#2c3e63",
    "dram": "#6b4c73",
    "eth": "#495a78",
    "router": "#6b7280",
    "pcie": "#c68b4a",
    "arc": "#6abf69",
}
LIGHT_FILL = {
    "worker": "#e8ecf1",
    "dram": "#ece5f3",
    "eth": "#e3e9f0",
    "router": "#e5e7eb",
    "pcie": "#f5e6d3",
    "arc": "#ddf0dd",
}
LIGHT_EDGE = {
    "worker": "#9ba8ba",
    "dram": "#8b6f9e",
    "eth": "#7b8ca0",
    "router": "#7f868f",
    "pcie": "#c4944e",
    "arc": "#52a852",
}

CLEAN_SBLOCK_COLORS = {
    "#e74c3c": "#c0392b",
    "#e67e22": "#d4770b",
    "#f1c40f": "#9a7d0a",
    "#2ecc71": "#1a7a3e",
    "#3498db": "#2471a3",
    "#9b59b6": "#7d3c98",
}


def sbc(color):
    """Map a saturated S-block color to its muted paper-friendly variant."""
    return CLEAN_SBLOCK_COLORS.get(color, color)
TYPE_LABEL = {"worker": "T", "dram": "D", "eth": "E", "router": "R", "pcie": "P", "arc": "A"}

TYPE_BY_COORD = {}
for coord in WORKERS:
    TYPE_BY_COORD[coord] = "worker"
for coord in ETH:
    TYPE_BY_COORD[coord] = "eth"
for coord in ROUTERS:
    TYPE_BY_COORD[coord] = "router"
for coord in PCIE:
    TYPE_BY_COORD[coord] = "pcie"
for coord in ARC:
    TYPE_BY_COORD[coord] = "arc"
for coords in DRAM_CHANNELS.values():
    for coord in coords:
        TYPE_BY_COORD[coord] = "dram"

CURRENT_LOGICAL_TO_PHYSICAL = {
    (lx, ly): (CURRENT_LOGICAL_X_TO_PHYSICAL_X[lx], CURRENT_LOGICAL_Y_TO_PHYSICAL_Y[ly])
    for lx in range(len(CURRENT_LOGICAL_X_TO_PHYSICAL_X))
    for ly in range(len(CURRENT_LOGICAL_Y_TO_PHYSICAL_Y))
}
CURRENT_PHYSICAL_TO_LOGICAL = {phys: logical for logical, phys in CURRENT_LOGICAL_TO_PHYSICAL.items()}

ACTIVE_LAYOUT_NAME = "4c"
ACTIVE_LAYOUT_DESC = "4-core S-block"
ACTIVE_CORES = 24
SBLOCKS = SBLOCKS_4C
SELECTED_DRAM_TILES = {}
SELECTED_WORKER_TILES = {}
BLOCK_BY_NAME = {}


def activate_layout(layout_name):
    global ACTIVE_LAYOUT_NAME, ACTIVE_LAYOUT_DESC, ACTIVE_CORES
    global SBLOCKS, SELECTED_DRAM_TILES, SELECTED_WORKER_TILES, BLOCK_BY_NAME

    if layout_name == "4c":
        ACTIVE_LAYOUT_NAME = "4c"
        ACTIVE_LAYOUT_DESC = "4-core S-block"
        ACTIVE_CORES = 24
        SBLOCKS = SBLOCKS_4C
    elif layout_name == "8c":
        ACTIVE_LAYOUT_NAME = "8c"
        ACTIVE_LAYOUT_DESC = "8-core S-block"
        ACTIVE_CORES = 48
        SBLOCKS = SBLOCKS_8C
    else:
        raise ValueError(f"Unsupported layout_name={layout_name}")

    SELECTED_DRAM_TILES = {DRAM_VIEWS_NOC0[sb["dram_view"]]: sb for sb in SBLOCKS}
    SELECTED_WORKER_TILES = {}
    for sb in SBLOCKS:
        for idx, coord in enumerate(sb["cores"]):
            SELECTED_WORKER_TILES[coord] = (sb, idx)
    BLOCK_BY_NAME = {sb["name"]: sb for sb in SBLOCKS}


activate_layout("4c")


def block_center(block):
    xs = [c[0] for c in block["cores"]]
    ys = [c[1] for c in block["cores"]]
    return (gcx(min(xs)) + gcx(max(xs))) / 2, (gcy(min(ys)) + gcy(max(ys))) / 2


def build_figure(clean=False):
    bg = "#ffffff" if clean else "#0a0e17"
    fig, ax = plt.subplots(figsize=(20, 18), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(-3.2, gx(GRID_X) + 5.0)
    ax.set_ylim(-6.6, gy(-1) + 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def add_title(ax, title, subtitle):
    grid_mid = (gx(0) + gx(GRID_X - 1) + TILE_W) / 2
    ax.text(
        grid_mid,
        gy(-1) + 2.15,
        title,
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=CJK_FONT, size=18),
        color="#00d2ff",
    )
    ax.text(
        grid_mid,
        gy(-1) + 1.45,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#576574",
        fontfamily=MONO,
    )


def draw_headers(ax, show_logical_headers=False, clean=False):
    if not clean:
        for x in range(GRID_X):
            ax.text(gcx(x), gy(-1) + 0.80, f"x={x}", ha="center", va="center", fontsize=8, color="#576574", fontfamily=MONO)
        for y in range(GRID_Y):
            ax.text(-1.0, gcy(y), f"y={y}", ha="center", va="center", fontsize=8, color="#576574", fontfamily=MONO)

    if show_logical_headers:
        for lx_idx, px in enumerate(CURRENT_LOGICAL_X_TO_PHYSICAL_X):
            ax.text(gcx(px), gy(-1) + 0.45, f"Lx={lx_idx}", ha="center", va="center", fontsize=7.2, color="#48dbfb", fontfamily=MONO)
        for ly_idx, py in enumerate(CURRENT_LOGICAL_Y_TO_PHYSICAL_Y):
            ax.text(-1.95, gcy(py), f"Ly={ly_idx}", ha="center", va="center", fontsize=7.2, color="#48dbfb", fontfamily=MONO)


def draw_floorplan_bands(ax, clean=False):
    if clean:
        dram_fc, dram_ec = "#ede9fe", "#c4b5fd"
        tensix_fc, tensix_ec = "#eff6ff", "#bfdbfe"
        dram_txt_color = "#6d28d9"
        tensix_txt_color = "#1d4ed8"
    else:
        dram_fc, dram_ec = "#6b4c7312", "#6b4c7340"
        tensix_fc, tensix_ec = "#1f3b6d0d", "#1f3b6d22"
        dram_txt_color = "#b58ac3"
        tensix_txt_color = "#6ea3ff"

    ax.add_patch(FancyBboxPatch((gx(0) - 0.18, gy(11) - 0.16), TILE_W + 0.36, gy(0) - gy(11) + TILE_H + 0.32,
                                boxstyle="round,pad=0.08", facecolor=dram_fc, edgecolor=dram_ec,
                                linewidth=1.2, linestyle="--", zorder=0))
    ax.text(gcx(0), gy(-1) + 0.10, "Left IO / DRAM spine", ha="center", va="center", fontsize=8, color=dram_txt_color, fontfamily=MONO)

    ax.add_patch(FancyBboxPatch((gx(5) - 0.18, gy(11) - 0.16), TILE_W + 0.36, gy(0) - gy(11) + TILE_H + 0.32,
                                boxstyle="round,pad=0.08", facecolor=dram_fc, edgecolor=dram_ec,
                                linewidth=1.2, linestyle="--", zorder=0))
    ax.text(gcx(5), gy(-1) + 0.10, "Center DRAM spine", ha="center", va="center", fontsize=8, color=dram_txt_color, fontfamily=MONO)

    left_x = gx(1) - 0.15
    left_w = gx(4) + TILE_W - left_x + 0.15
    ax.add_patch(FancyBboxPatch((left_x, gy(11) - 0.10), left_w, gy(1) - gy(11) + TILE_H + 0.20,
                                boxstyle="round,pad=0.06", facecolor=tensix_fc, edgecolor=tensix_ec,
                                linewidth=1.0, zorder=0))
    ax.text((gx(1) + gx(4) + TILE_W) / 2, gy(-1) + 0.10, "Left Tensix array", ha="center", va="center", fontsize=8, color=tensix_txt_color, fontfamily=MONO)

    right_x = gx(6) - 0.15
    right_w = gx(9) + TILE_W - right_x + 0.15
    ax.add_patch(FancyBboxPatch((right_x, gy(11) - 0.10), right_w, gy(1) - gy(11) + TILE_H + 0.20,
                                boxstyle="round,pad=0.06", facecolor=tensix_fc, edgecolor=tensix_ec,
                                linewidth=1.0, zorder=0))
    ax.text((gx(6) + gx(9) + TILE_W) / 2, gy(-1) + 0.10, "Right Tensix array", ha="center", va="center", fontsize=8, color=tensix_txt_color, fontfamily=MONO)


def draw_tiles(ax, style, clean=False):
    for y in range(GRID_Y):
        for x in range(GRID_X):
            coord = (x, y)
            tile_type = TYPE_BY_COORD.get(coord, "worker")
            if clean:
                fill = LIGHT_FILL[tile_type]
                edge = LIGHT_EDGE[tile_type]
            else:
                fill = BASE_FILL[tile_type]
                edge = BASE_EDGE[tile_type]
            lw = 1.0

            if style == "overlay" and tile_type == "worker":
                if coord in CURRENT_PHYSICAL_TO_LOGICAL:
                    fill = "#18304e"
                    edge = "#3c6ba8"
                else:
                    fill = "#101826"
                    edge = "#22314a"

            if coord in SELECTED_DRAM_TILES:
                raw_color = SELECTED_DRAM_TILES[coord]["color"]
                c = sbc(raw_color) if clean else raw_color
                fill = c + ("15" if clean else "33")
                edge = c
                lw = 2.0
            if coord in SELECTED_WORKER_TILES:
                sb, _ = SELECTED_WORKER_TILES[coord]
                c = sbc(sb["color"]) if clean else sb["color"]
                fill = "#ffffff" if clean else "#1a1a2e"
                edge = c
                lw = 2.0 if clean else 1.7

            ax.add_patch(
                FancyBboxPatch(
                    (gx(x), gy(y)),
                    TILE_W,
                    TILE_H,
                    boxstyle="round,pad=0.05",
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=lw,
                    zorder=2,
                )
            )

            if style == "overlay" and coord in CURRENT_PHYSICAL_TO_LOGICAL and coord not in SELECTED_WORKER_TILES:
                ax.add_patch(
                    FancyBboxPatch(
                        (gx(x) + 0.06, gy(y) + 0.06),
                        TILE_W - 0.12,
                        TILE_H - 0.12,
                        boxstyle="round,pad=0.02",
                        facecolor="none",
                        edgecolor="#48dbfb55",
                        linewidth=0.9,
                        linestyle=":",
                        zorder=3,
                    )
                )

            if coord in SELECTED_DRAM_TILES:
                sb = SELECTED_DRAM_TILES[coord]
                c = sbc(sb["color"]) if clean else sb["color"]
                ax.text(gcx(x), gcy(y) + 0.10, f"DV{sb['dram_view']}", ha="center", va="center", fontsize=6.4, color=c, fontfamily=MONO, fontweight="bold")
                lane_num = sb["name"][1:]
                bottom_label = f"Lane {lane_num}" if clean else sb["name"]
                ax.text(gcx(x), gcy(y) - 0.12, bottom_label, ha="center", va="center", fontsize=5.3, color=c, fontfamily=MONO)
            elif coord in SELECTED_WORKER_TILES:
                sb, idx = SELECTED_WORKER_TILES[coord]
                is_wide = len(sb["cores"]) > 4
                coord_fs = 4.5 if is_wide else 5.2
                label_fs = 5.0 if is_wide else 5.8
                sub_fs = 4.2 if is_wide else 4.8
                out_fs = 3.9 if is_wide else 4.7
                coord_color = "#6b7280" if clean else "#8395a7"
                ax.text(gcx(x), gcy(y) + 0.20, f"({x},{y})", ha="center", va="center", fontsize=coord_fs, color=coord_color, fontfamily=MONO)
                if idx == 0:
                    if clean:
                        ax.text(gcx(x), gcy(y) - 0.02, "Lsrc", ha="center", va="center", fontsize=label_fs, color="#b91c1c", fontfamily=MONO, fontweight="bold")
                    else:
                        sender_label = "SNDR" if is_wide else "SENDER"
                        sub_label = "NCR" if is_wide else "NCRISC"
                        ax.text(gcx(x), gcy(y) + 0.02, sender_label, ha="center", va="center", fontsize=label_fs, color="#ff6b6b", fontfamily=MONO, fontweight="bold")
                        ax.text(gcx(x), gcy(y) - 0.17, sub_label, ha="center", va="center", fontsize=sub_fs, color="#ff6b6b", fontfamily=MONO, alpha=0.8)
                else:
                    worker_color = "#4b5563" if clean else "#feca57"
                    ax.text(gcx(x), gcy(y) + 0.02, f"w{idx}", ha="center", va="center", fontsize=label_fs, color=worker_color, fontfamily=MONO, fontweight="bold")
                    if not clean:
                        ax.text(gcx(x), gcy(y) - 0.17, "TRISC", ha="center", va="center", fontsize=sub_fs, color="#1dd1a1", fontfamily=MONO, alpha=0.8)
                if sb["name"] == "S1":
                    out_color = "#1e40af" if clean else "#48dbfb"
                    ax.text(gcx(x), gcy(y) - 0.34, f"OUT[{idx}]", ha="center", va="center", fontsize=out_fs, color=out_color, fontfamily=MONO, fontweight="bold")
                if style == "overlay" and coord in CURRENT_PHYSICAL_TO_LOGICAL:
                    lx_idx, ly_idx = CURRENT_PHYSICAL_TO_LOGICAL[coord]
                    overlay_fs = 4.2 if is_wide else 4.8
                    ax.text(gx(x) + 0.09, gy(y) + 0.83, f"L{lx_idx},{ly_idx}", ha="left", va="center", fontsize=overlay_fs, color="#48dbfb", fontfamily=MONO)
            else:
                if style == "overlay" and coord in CURRENT_PHYSICAL_TO_LOGICAL:
                    lx_idx, ly_idx = CURRENT_PHYSICAL_TO_LOGICAL[coord]
                    ax.text(gx(x) + 0.08, gy(y) + 0.80, f"L{lx_idx},{ly_idx}", ha="left", va="center", fontsize=5.0, color="#48dbfb", fontfamily=MONO)
                    ax.text(gcx(x), gcy(y) - 0.02, "T", ha="center", va="center", fontsize=6.2, color="#5d8dc8", fontfamily=MONO, alpha=0.65)
                else:
                    type_alpha = 0.55 if clean else 0.65
                    type_color = LIGHT_EDGE[tile_type] if clean else edge
                    ax.text(gcx(x), gcy(y), TYPE_LABEL[tile_type], ha="center", va="center", fontsize=6.5, color=type_color, fontfamily=MONO, alpha=type_alpha)


def draw_sblocks(ax, style, clean=False):
    for sb in SBLOCKS:
        xs = [c[0] for c in sb["cores"]]
        ys = [c[1] for c in sb["cores"]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 0.11
        width = (max_x - min_x + 1) * (TILE_W + GAP) - GAP + 2 * pad
        height = (max_y - min_y + 1) * (TILE_H + GAP) - GAP + 2 * pad
        c = sbc(sb["color"]) if clean else sb["color"]
        ax.add_patch(
            FancyBboxPatch(
                (gx(min_x) - pad, gy(max_y) - pad),
                width,
                height,
                boxstyle="round,pad=0.08",
                facecolor=c + ("08" if clean else "08"),
                edgecolor=c,
                linewidth=2.0,
                linestyle="--",
                zorder=3,
            )
        )
        cx = (gx(min_x) + gx(max_x) + TILE_W) / 2
        cy = gy(min_y) + TILE_H + 0.34
        lane_num = sb["name"][1:]
        if style == "overlay":
            phys_coords = ", ".join(f"({x},{y})" for x, y in sb["cores"])
            ax.text(cx, cy, f"{sb['name']}  {ACTIVE_LAYOUT_NAME}  phys={phys_coords}", ha="center", va="center", fontsize=6.4, color=c, fontfamily=MONO, fontweight="bold")
        elif clean:
            ax.text(cx, cy, f"Lane {lane_num}  (DV{sb['dram_view']})", ha="center", va="center", fontsize=8.4, color=c, fontfamily=MONO, fontweight="bold")
        else:
            ax.text(cx, cy, f"{sb['name']}  (DV{sb['dram_view']}, {ACTIVE_LAYOUT_NAME})", ha="center", va="center", fontsize=8.4, color=c, fontfamily=MONO, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color, lw=1.8, ls="-", alpha=0.85, zorder=5, rad=0.0):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            alpha=alpha,
            mutation_scale=12,
        ),
        zorder=zorder,
    )


def draw_flows(ax, style, clean=False):
    # 1. K chunk: DRAM bank → lane source
    dram_rads = {"S1": 0.0, "S2": -0.08, "S3": -0.14, "S4": 0.0, "S5": 0.0, "S6": 0.10}
    k_arrow_color = "#b91c1c" if clean else "#ff6b6b"
    for sb in SBLOCKS:
        dv = DRAM_VIEWS_NOC0[sb["dram_view"]]
        sender = sb["cores"][0]
        arrow(ax, gcx(dv[0]), gcy(dv[1]), gcx(sender[0]), gcy(sender[1]), k_arrow_color, lw=2.35, alpha=0.92, zorder=8, rad=dram_rads[sb["name"]])

    # 2. Lane-wise K broadcast inside lane group
    k_mcast_color = "#92400e" if clean else "#feca57"
    for sb in SBLOCKS:
        sx, sy = gcx(sb["cores"][0][0]), gcy(sb["cores"][0][1])
        for worker in sb["cores"][1:]:
            tx, ty = gcx(worker[0]), gcy(worker[1])
            arrow(ax, sx, sy, tx, ty, k_mcast_color, lw=1.8, alpha=0.86, zorder=7)

    # 3. Track-wise Q distribution from Lane 1 track source
    s1_center = block_center(BLOCK_BY_NAME["S1"])
    if not clean:
        q_box_x = -2.1
        q_box_y = gcy(1)
        q_text = "Q shards\nalready in\nS1 output L1"
        q_color = "#48dbfb"
        q_fc, q_ec = "#48dbfb11", "#48dbfb55"
        if style == "overlay":
            q_text += "\n(logical batch/q_shards)"
        ax.text(
            q_box_x,
            q_box_y,
            q_text,
            ha="center",
            va="center",
            fontsize=7.2,
            color=q_color,
            fontfamily=MONO,
            bbox=dict(boxstyle="round,pad=0.32", facecolor=q_fc, edgecolor=q_ec),
        )
        arrow(ax, q_box_x + 0.65, q_box_y, gx(1) - 0.08, q_box_y, q_color, lw=2.0, alpha=0.82, zorder=6)
    q_fanout_color = "#1e40af" if clean else "#48dbfb"
    q_rads = {"S2": -0.18, "S3": -0.28, "S4": 0.12, "S5": 0.08, "S6": 0.22}
    for sb in SBLOCKS[1:]:
        dx, dy = block_center(sb)
        arrow(ax, s1_center[0] + 0.18, s1_center[1] - 0.10, dx - 0.18, dy + 0.08, q_fanout_color, lw=1.45, alpha=0.42, zorder=4, rad=q_rads[sb["name"]])

    # 5. Tree reduction
    if not clean:
        reduction_edges = [
            ("S2", "S1", -0.10),
            ("S4", "S3", 0.10),
            ("S6", "S5", 0.10),
            ("S3", "S1", -0.18),
            ("S5", "S1", 0.15),
        ]
        for src_name, dst_name, rad in reduction_edges:
            sx, sy = block_center(BLOCK_BY_NAME[src_name])
            dx, dy = block_center(BLOCK_BY_NAME[dst_name])
            arrow(ax, sx, sy, dx, dy, "#c56cf0", lw=2.1, ls="--", alpha=0.72, zorder=9, rad=rad)

    # 6. Output at track sink
    if not clean:
        out_box_x = gx(GRID_X) + 2.05
        out_box_y = gcy(1)
        out_text = "Final O shards\nstay in S1\noutput-core L1\n(host readback\noutside kernel)"
        out_color = "#ff9ff3"
        out_fc, out_ec = "#ff9ff311", "#ff9ff355"
        if style == "overlay":
            out_text = "Final O shards\nstay in S1 output L1\nphys tiles map to\nlogical L(0..1,0..1)"
        ax.text(
            out_box_x,
            out_box_y,
            out_text,
            ha="center",
            va="center",
            fontsize=7.0,
            color=out_color,
            fontfamily=MONO,
            bbox=dict(boxstyle="round,pad=0.34", facecolor=out_fc, edgecolor=out_ec),
        )
        arrow(ax, s1_center[0] + 0.38, s1_center[1] + 0.08, out_box_x - 0.72, out_box_y, out_color, lw=1.8, alpha=0.76, zorder=6, rad=0.10)


def draw_tile_legend(ax, x0, y0, clean=False):
    if clean:
        tile_legend = [
            ("Tensix", LIGHT_FILL["worker"], LIGHT_EDGE["worker"]),
            ("Lane group core", "#ffffff", "#2471a3"),
            ("DRAM endpoint", LIGHT_FILL["dram"], LIGHT_EDGE["dram"]),
            ("Bound DRAM bank", "#ffffff", "#c0392b"),
            ("Ethernet", LIGHT_FILL["eth"], LIGHT_EDGE["eth"]),
            ("PCIe / Router / ARC", LIGHT_FILL["router"], LIGHT_EDGE["router"]),
        ]
        txt_color = "#1f2937"
    else:
        tile_legend = [
            ("Tensix", "#172033", "#2c3e63"),
            ("Selected S-block core", "#1a1a2e", "#00d2ff"),
            ("DRAM endpoint", "#201824", "#6b4c73"),
            ("Selected DRAM view", "#2b1f35", "#ff6b6b"),
            ("Ethernet", "#1d2434", "#495a78"),
            ("PCIe / Router / ARC", "#232933", "#6b7280"),
        ]
        txt_color = "#c8d6e5"
    for i, (label, fill, edge) in enumerate(tile_legend):
        yy = y0 - i * 0.45
        ax.add_patch(FancyBboxPatch((x0, yy - 0.15), 0.42, 0.30, boxstyle="round,pad=0.03", facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=10))
        ax.text(x0 + 0.55, yy, label, ha="left", va="center", fontsize=7.0, color=txt_color, fontfamily=MONO)


def draw_flow_legend(ax, x0, y0, clean=False):
    if clean:
        flow_items = [
            ("Track-wise Q distribution", "#1e40af", "-"),
            ("K chunk: DRAM bank \u2192 Lsrc", "#b91c1c", "-"),
            ("Lane-wise K broadcast", "#92400e", "-"),
        ]
    else:
        flow_items = [
            ("1  K/V page: DRAM endpoint -> sender", "#ff6b6b", "-"),
            ("2  K multicast inside S-block", "#feca57", "-"),
            ("3  Q fanout from S1 output cores", "#48dbfb", "-"),
            ("4  TRISC local compute on each active core", "#1dd1a1", "-"),
            ("5  Tree reduction back to S1", "#c56cf0", "--"),
            ("6  Final O remains in S1 output L1", "#ff9ff3", "-"),
        ]
    for i, (label, color, ls) in enumerate(flow_items):
        yy = y0 - i * 0.40
        ax.plot([x0, x0 + 0.60], [yy, yy], color=color, lw=3, ls=ls, alpha=0.95)
        ax.annotate("", xy=(x0 + 0.78, yy), xytext=(x0 + 0.60, yy), arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=10))
        ax.text(x0 + 0.95, yy, label, ha="left", va="center", fontsize=7.0, color=color, fontfamily=MONO)


def draw_overlay_mapping_boxes(ax):
    map_x = gx(GRID_X) + 1.0
    map_y = -1.25
    x_map = "  ".join(f"Lx{i}->{px}" for i, px in enumerate(CURRENT_LOGICAL_X_TO_PHYSICAL_X))
    y_map = "  ".join(f"Ly{i}->{py}" for i, py in enumerate(CURRENT_LOGICAL_Y_TO_PHYSICAL_Y))
    ax.text(
        map_x,
        map_y,
        "Current logical compute grid (8x7) overlay\n"
        f"x map: {x_map}\n"
        f"y map: {y_map}\n"
        f"layout: {ACTIVE_LAYOUT_DESC}\n"
        "Interpretation: cyan Lx/Ly labels are the logical coords used by current tt-metal compute grid.",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#48dbfb",
        fontfamily=MONO,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0c1626", edgecolor="#48dbfb44"),
    )

    sb_lines = []
    for sb in SBLOCKS:
        logical = [coord for coord, phys in CURRENT_LOGICAL_TO_PHYSICAL.items() if phys in sb["cores"]]
        logical = sorted(logical, key=lambda t: (t[1], t[0]))
        logical_str = " ".join(f"({lx},{ly})" for lx, ly in logical)
        sb_lines.append(f"{sb['name']}: logical {logical_str}")
    ax.text(
        map_x,
        map_y - 1.95,
        "S-block logical coords in current 8x7 grid\n" + "\n".join(sb_lines),
        ha="left",
        va="top",
        fontsize=7.1,
        color="#c8d6e5",
        fontfamily=MONO,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#111827", edgecolor="#1e293b"),
    )


def draw_floorplan_notes(ax, clean=False):
    if clean:
        return
    txt_color, box_fc, box_ec = "#c8d6e5", "#111827", "#1e293b"
    ax.text(
        4.7,
        -4.1,
        "Physical NOC0 landmarks\n"
        "  x=0 : left IO / DRAM spine\n"
        "  x=5 : center DRAM spine\n"
        "  x=1..4 and 6..9 : Tensix arrays\n"
        "  y=0 and y=6 : Ethernet rows\n"
        f"Active layout: {ACTIVE_LAYOUT_DESC} ({ACTIVE_CORES} active cores)\n"
        "Selected DRAM views for SF-MLA:\n"
        "  DV1=(0,1) DV2=(0,5) DV0=(0,11)\n"
        "  DV4=(5,1) DV9=(5,3) DV8=(5,8)",
        ha="left",
        va="top",
        fontsize=7.2,
        color=txt_color,
        fontfamily=MONO,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=box_fc, edgecolor=box_ec),
    )


def draw_footer(ax, text):
    grid_mid = (gx(0) + gx(GRID_X - 1) + TILE_W) / 2
    ax.text(
        grid_mid,
        -6.0,
        text,
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=CJK_FONT, size=9.5),
        color="#00d2ff",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#111827", edgecolor="#1e293b"),
    )


def save(fig, stem, clean=False):
    bg = "#ffffff" if clean else "#0a0e17"
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight", facecolor=bg, edgecolor="none")
    fig.savefig(f"{stem}.svg", format="svg", bbox_inches="tight", facecolor=bg, edgecolor="none")
    plt.close(fig)


def render_floorplan(stem, layout_name="4c", clean=False):
    activate_layout(layout_name)
    fig, ax = build_figure(clean=clean)
    if not clean:
        title_suffix = f"{ACTIVE_LAYOUT_DESC}"
        add_title(
            ax,
            f"Wormhole B0 — SF-MLA 物理运行图 ({title_suffix})",
            f"10x12 NOC0 physical tile layout | {ACTIVE_CORES} active cores | emphasize left IO/DRAM spine, center DRAM spine, and two Tensix arrays",
        )
    draw_floorplan_bands(ax, clean=clean)
    draw_headers(ax, show_logical_headers=False, clean=clean)
    draw_tiles(ax, style="floorplan", clean=clean)
    draw_sblocks(ax, style="floorplan", clean=clean)
    draw_flows(ax, style="floorplan", clean=clean)
    draw_tile_legend(ax, -2.4, -1.15, clean=clean)
    draw_flow_legend(ax, 4.8, -1.15, clean=clean)
    draw_floorplan_notes(ax, clean=clean)
    if not clean:
        draw_footer(ax, "数据流: ① K/V 在物理 DRAM 接口 tile 上进入 → ② sender 在 block 内多播 K → ③ Q 从 S1 output-core L1 扇出 → ④ 各核本地计算 → ⑤ 树形归约回 S1 → ⑥ 最终 O 留在 S1 output L1")
    save(fig, stem, clean=clean)


def render_overlay(stem):
    activate_layout("4c")
    fig, ax = build_figure()
    add_title(
        ax,
        "Wormhole B0 — SF-MLA 逻辑到物理映射图 (Logical overlay style)",
        "same physical NOC0 layout, but overlay current 8x7 tt-metal logical compute grid on top of the physical worker tiles",
    )
    draw_headers(ax, show_logical_headers=True)
    draw_tiles(ax, style="overlay")
    draw_sblocks(ax, style="overlay")
    draw_flows(ax, style="overlay")
    draw_tile_legend(ax, -2.4, -1.15)
    draw_flow_legend(ax, 4.7, -1.15)
    draw_overlay_mapping_boxes(ax)
    draw_footer(ax, "数据流: 物理 tile 版保持不变；青色 Lx/Ly 只是叠加显示当前 8x7 logical compute grid 如何映射到 Wormhole 的物理 worker tiles")
    save(fig, stem)


def render_pptx_clean(filepath, layout_name="8c"):
    """Render the clean floorplan diagram as an editable PPTX file.

    Every visual element (tile, S-block outline, data-flow arrow, legend)
    becomes a separate editable PowerPoint shape so users can freely adjust
    colors, labels, and layout in any presentation tool.
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.oxml.ns import qn
    from lxml import etree

    activate_layout(layout_name)

    prs = Presentation()
    prs.slide_width = Inches(20)
    prs.slide_height = Inches(15)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    SC = 0.72
    XO, YO, YM = 3.0, 0.8, 14.5
    NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"

    def ix(mx): return Inches(XO + mx * SC)
    def iy(my): return Inches(YO + (YM - my) * SC)
    def iw(w):  return Inches(abs(w) * SC)
    def ih(h):  return Inches(abs(h) * SC)

    def _rgb(hx):
        h = hx.lstrip("#")[:6]
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _rect(l, t, w, h, fc=None, ec=None, lw=Pt(1), dash=False):
        s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
        if fc:
            s.fill.solid()
            s.fill.fore_color.rgb = _rgb(fc)
        else:
            s.fill.background()
        if ec:
            s.line.color.rgb = _rgb(ec)
            s.line.width = lw
            if dash:
                sp_pr = s._element.find(qn("p:spPr"))
                ln = sp_pr.find(qn("a:ln"))
                if ln is not None:
                    prstDash = etree.SubElement(ln, f"{{{NS_A}}}prstDash")
                    prstDash.set("val", "dash")
        else:
            s.line.width = Pt(0)
        return s

    def _txt(text, l, t, w, h, fs=8, fc="#000000", bold=False, al=PP_ALIGN.CENTER):
        tb = slide.shapes.add_textbox(l, t, w, h)
        tf = tb.text_frame
        tf.word_wrap = False
        body_pr = tb._element.find(qn("p:txBody")).find(qn("a:bodyPr"))
        body_pr.set("anchor", "ctr")
        body_pr.set("lIns", "0")
        body_pr.set("rIns", "0")
        body_pr.set("tIns", "0")
        body_pr.set("bIns", "0")
        p = tf.paragraphs[0]
        p.alignment = al
        r = p.add_run()
        r.text = text
        r.font.size = Pt(fs)
        r.font.color.rgb = _rgb(fc)
        r.font.bold = bold
        r.font.name = "Consolas"
        return tb

    def _arrow(x1, y1, x2, y2, color, w=Pt(2)):
        from pptx.enum.shapes import MSO_CONNECTOR_TYPE
        cn = slide.shapes.add_connector(MSO_CONNECTOR_TYPE.STRAIGHT, x1, y1, x2, y2)
        cn.line.color.rgb = _rgb(color)
        cn.line.width = w
        sp_pr = cn._element.find(qn("p:spPr"))
        if sp_pr is not None:
            ln = sp_pr.find(qn("a:ln"))
            if ln is not None:
                te = etree.SubElement(ln, f"{{{NS_A}}}tailEnd")
                te.set("type", "triangle")
                te.set("w", "med")
                te.set("len", "med")
        return cn

    # === 1. Floorplan bands ===
    dfc, dec = "#ede9fe", "#c4b5fd"
    tfc, tec = "#eff6ff", "#bfdbfe"

    bh = gy(0) - gy(11) + TILE_H + 0.32
    bt = gy(11) - 0.16 + bh
    for spine_x in [gx(0) - 0.18, gx(5) - 0.18]:
        _rect(ix(spine_x), iy(bt), iw(TILE_W + 0.36), ih(bh), dfc, dec, Pt(1.2), True)

    th = gy(1) - gy(11) + TILE_H + 0.20
    tt = gy(11) - 0.10 + th
    for tx0, tw0 in [(gx(1) - 0.15, gx(4) + TILE_W - gx(1) + 0.30),
                     (gx(6) - 0.15, gx(9) + TILE_W - gx(6) + 0.30)]:
        _rect(ix(tx0), iy(tt), iw(tw0), ih(th), tfc, tec, Pt(1))

    lbl_y = gy(-1) + 0.10
    _txt("Left IO / DRAM spine", ix(gcx(0) - 1.5), iy(lbl_y + 0.15), iw(3), ih(0.3), 7, "#6d28d9")
    _txt("Center DRAM spine",    ix(gcx(5) - 1.5), iy(lbl_y + 0.15), iw(3), ih(0.3), 7, "#6d28d9")
    _txt("Left Tensix array",    ix((gx(1) + gx(4) + TILE_W) / 2 - 1.5), iy(lbl_y + 0.15), iw(3), ih(0.3), 7, "#1d4ed8")
    _txt("Right Tensix array",   ix((gx(6) + gx(9) + TILE_W) / 2 - 1.5), iy(lbl_y + 0.15), iw(3), ih(0.3), 7, "#1d4ed8")

    # === 2. Tiles ===
    for y_idx in range(GRID_Y):
        for x_idx in range(GRID_X):
            coord = (x_idx, y_idx)
            tile_type = TYPE_BY_COORD.get(coord, "worker")
            fc, ec, lw = LIGHT_FILL[tile_type], LIGHT_EDGE[tile_type], Pt(1)

            if coord in SELECTED_DRAM_TILES:
                c = sbc(SELECTED_DRAM_TILES[coord]["color"])
                fc, ec, lw = "#f5f5f5", c, Pt(2)
            if coord in SELECTED_WORKER_TILES:
                sb_info, _ = SELECTED_WORKER_TILES[coord]
                c = sbc(sb_info["color"])
                fc, ec, lw = "#ffffff", c, Pt(2)

            _rect(ix(gx(x_idx)), iy(gy(y_idx) + TILE_H), iw(TILE_W), ih(TILE_H), fc, ec, lw)

            cx, cy = gcx(x_idx), gcy(y_idx)
            if coord in SELECTED_DRAM_TILES:
                sb_d = SELECTED_DRAM_TILES[coord]
                c_d = sbc(sb_d["color"])
                _txt(f"DV{sb_d['dram_view']}", ix(cx - 0.4), iy(cy + 0.22), iw(0.8), ih(0.24), 6, c_d, True)
                _txt(f"Lane {sb_d['name'][1:]}", ix(cx - 0.4), iy(cy - 0.02), iw(0.8), ih(0.22), 5, c_d)
            elif coord in SELECTED_WORKER_TILES:
                sb_w, w_idx = SELECTED_WORKER_TILES[coord]
                is_wide = len(sb_w["cores"]) > 4
                cfs = 4 if is_wide else 5
                lfs = 5 if is_wide else 6
                _txt(f"({x_idx},{y_idx})", ix(cx - 0.4), iy(cy + 0.35), iw(0.8), ih(0.20), cfs, "#6b7280")
                if w_idx == 0:
                    _txt("Lsrc", ix(cx - 0.4), iy(cy + 0.10), iw(0.8), ih(0.22), lfs, "#b91c1c", True)
                else:
                    _txt(f"w{w_idx}", ix(cx - 0.4), iy(cy + 0.10), iw(0.8), ih(0.22), lfs, "#4b5563", True)
                if sb_w["name"] == "S1":
                    _txt(f"OUT[{w_idx}]", ix(cx - 0.4), iy(cy - 0.18), iw(0.8), ih(0.18), 4, "#1e40af", True)
            else:
                _txt(TYPE_LABEL[tile_type], ix(cx - 0.2), iy(cy + 0.10), iw(0.4), ih(0.22), 6, LIGHT_EDGE[tile_type])

    # === 3. S-block outlines ===
    for sb in SBLOCKS:
        xs_sb = [c[0] for c in sb["cores"]]
        ys_sb = [c[1] for c in sb["cores"]]
        mn_x, mx_x = min(xs_sb), max(xs_sb)
        mn_y, mx_y = min(ys_sb), max(ys_sb)
        pad = 0.11
        sb_w = (mx_x - mn_x + 1) * (TILE_W + GAP) - GAP + 2 * pad
        sb_h = (mx_y - mn_y + 1) * (TILE_H + GAP) - GAP + 2 * pad
        c_sb = sbc(sb["color"])
        _rect(ix(gx(mn_x) - pad), iy(gy(mx_y) - pad + sb_h), iw(sb_w), ih(sb_h), None, c_sb, Pt(2), True)

        cx_sb = (gx(mn_x) + gx(mx_x) + TILE_W) / 2
        cy_sb = gy(mn_y) + TILE_H + 0.34
        _txt(f"Lane {sb['name'][1:]}  (DV{sb['dram_view']})",
             ix(cx_sb - 2), iy(cy_sb + 0.15), iw(4), ih(0.3), 8, c_sb, True)

    # === 4. Data-flow arrows ===
    for sb in SBLOCKS:
        dv = DRAM_VIEWS_NOC0[sb["dram_view"]]
        sndr = sb["cores"][0]
        _arrow(ix(gcx(dv[0])), iy(gcy(dv[1])),
               ix(gcx(sndr[0])), iy(gcy(sndr[1])), "#b91c1c", Pt(2.5))

    for sb in SBLOCKS:
        s0 = sb["cores"][0]
        for wk in sb["cores"][1:]:
            _arrow(ix(gcx(s0[0])), iy(gcy(s0[1])),
                   ix(gcx(wk[0])), iy(gcy(wk[1])), "#92400e", Pt(1.8))

    s1c = block_center(BLOCK_BY_NAME["S1"])
    for sb in SBLOCKS[1:]:
        dc = block_center(sb)
        _arrow(ix(s1c[0] + 0.18), iy(s1c[1] - 0.10),
               ix(dc[0] - 0.18), iy(dc[1] + 0.08), "#1e40af", Pt(1.5))

    # === 5. Tile legend ===
    lx0, ly0 = -2.2, -1.15
    tile_leg = [
        ("Tensix",              LIGHT_FILL["worker"], LIGHT_EDGE["worker"]),
        ("Lane group core",     "#ffffff",            "#2471a3"),
        ("DRAM endpoint",       LIGHT_FILL["dram"],   LIGHT_EDGE["dram"]),
        ("Bound DRAM bank",     "#ffffff",            "#c0392b"),
        ("Ethernet",            LIGHT_FILL["eth"],    LIGHT_EDGE["eth"]),
        ("PCIe / Router / ARC", LIGHT_FILL["router"], LIGHT_EDGE["router"]),
    ]
    for i, (lab, fc_l, ec_l) in enumerate(tile_leg):
        yy = ly0 - i * 0.45
        _rect(ix(lx0), iy(yy + 0.15), iw(0.42), ih(0.30), fc_l, ec_l, Pt(1.1))
        _txt(lab, ix(lx0 + 0.55), iy(yy + 0.12), iw(3), ih(0.25), 7, "#1f2937", al=PP_ALIGN.LEFT)

    # === 6. Flow legend ===
    fx0 = 4.8
    flow_leg = [
        ("Track-wise Q distribution",  "#1e40af"),
        ("K chunk: DRAM bank \u2192 Lsrc", "#b91c1c"),
        ("Lane-wise K broadcast",      "#92400e"),
    ]
    for i, (lab, c_f) in enumerate(flow_leg):
        yy = ly0 - i * 0.40
        _arrow(ix(fx0), iy(yy), ix(fx0 + 0.78), iy(yy), c_f, Pt(3))
        _txt(lab, ix(fx0 + 0.95), iy(yy + 0.10), iw(5), ih(0.25), 7, c_f, al=PP_ALIGN.LEFT)

    prs.save(filepath)
    print(f"Saved editable PPTX: {filepath}")


def main():
    docs_dir = Path(__file__).resolve().parent.parent
    images_dir = docs_dir / "assets" / "images"
    images_dir.mkdir(exist_ok=True)
    # Main figure now points to the more presentation-friendly floorplan style.
    render_floorplan(str(images_dir / "wh_sf_mla_runtime_diagram"), layout_name="4c")
    render_floorplan(str(images_dir / "wh_sf_mla_runtime_diagram_floorplan"), layout_name="4c")
    render_overlay(str(images_dir / "wh_sf_mla_runtime_diagram_logical_overlay"))
    render_floorplan(str(images_dir / "wh_sf_mla_runtime_diagram_8core"), layout_name="8c")
    render_floorplan(str(images_dir / "wh_sf_mla_runtime_diagram_8core_clean"), layout_name="8c", clean=True)
    render_pptx_clean(str(images_dir / "wh_sf_mla_runtime_diagram_8core_clean.pptx"), layout_name="8c")
    activate_layout("4c")
    print("Saved 4-core floorplan, 4-core overlay, 8-core floorplan, 8-core clean, and editable PPTX SF-MLA diagrams")


if __name__ == "__main__":
    main()
