#!/usr/bin/env python3
"""Generate small PNG diagrams for moe.ipynb (run from repo root)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parent


def save(fig: plt.Figure, name: str, *, pad_inches: float = 0.12) -> None:
    path = OUT / name
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=pad_inches,
    )
    plt.close(fig)


def fig_routing_topk() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    # Tokens
    ty = 1.9
    for i, x in enumerate([0.9, 1.45, 2.0]):
        ax.add_patch(plt.Circle((x, ty), 0.16, color="#27ae60", zorder=3))
    ax.text(1.45, 2.55, "Token batch", ha="center", fontsize=10, color="#2c3e50")

    # Router
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (3.35, 1.55),
            1.35,
            0.85,
            boxstyle="round,pad=0.04",
            facecolor="#2980b9",
            edgecolor="#1f618d",
            linewidth=1.2,
        )
    )
    ax.text(4.02, 1.98, "Router\nsoftmax + top-k", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Experts
    ex_x = [6.0, 7.05, 8.1, 9.15]
    labels = ["Expert 0", "Expert 1", "Expert 2", "Expert 3"]
    colors = ["#8e44ad", "#9b59b6", "#a569bd", "#bb8fce"]
    for x, lab, col in zip(ex_x, labels, colors, strict=True):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - 0.42, 0.45),
                0.84,
                1.05,
                boxstyle="round,pad=0.03",
                facecolor=col,
                edgecolor="#5b2c6f",
                linewidth=1,
                alpha=0.92,
            )
        )
        ax.text(x, 0.98, "FFN", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x, 0.55, lab.replace(" ", "\n"), ha="center", va="center", fontsize=7, color="white")

    # Arrows token -> router
    ax.add_patch(FancyArrowPatch((2.15, ty), (3.35, 1.98), arrowstyle="-|>", mutation_scale=12, color="#34495e", lw=1.4))

    # Arrows router -> two experts (top-2)
    for x in (ex_x[1], ex_x[3]):
        ax.add_patch(
            FancyArrowPatch(
                (4.7, 1.98),
                (x, 1.5),
                arrowstyle="-|>",
                mutation_scale=11,
                color="#c0392b",
                lw=1.5,
                connectionstyle="arc3,rad=0.12",
            )
        )
    ax.text(6.8, 1.35, "top-k (e.g. k=2)", fontsize=9, color="#c0392b", fontweight="bold")

    ax.text(5.0, 2.85, "Sparse activation: only k experts compute per token", ha="center", fontsize=10, color="#2c3e50")

    save(fig, "moe_routing_topk.png")


def fig_parallel_axes() -> None:
    """TP / PP / EP schematic — NVIDIA greens; room below for foot panel + padding."""
    NV = "#76B900"
    NV_DARK = "#5A8700"
    NV_MID = "#8BC727"
    NV_LIGHT = "#D4EDAA"
    NV_TINT = "#EEF8E0"
    INK = "#1A1A1A"
    INK_MUTED = "#3D3D3D"

    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 11)
    # Extra space below y=0 so the foot panel's rounded bottom is not clipped by tight bbox
    ax.set_ylim(-0.65, 8.75)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    box_edge = dict(facecolor=NV_TINT, edgecolor=NV, linewidth=1.1, alpha=0.99)

    def _callout(x0: float, y0: float, w: float, h: float, title: str, bullets: str) -> None:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.07",
                zorder=4,
                **box_edge,
            )
        )
        pad_x = 0.1
        top_y = y0 + h - 0.1
        ax.text(
            x0 + pad_x,
            top_y,
            title,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=INK,
            zorder=6,
        )
        # Second block anchored clearly below title so nothing clips the box bottom
        ax.text(
            x0 + pad_x,
            top_y - 0.28,
            bullets,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="normal",
            color=INK,
            linespacing=1.28,
            zorder=6,
        )

    # Short axis names at arrow tips
    tip_kw = dict(fontsize=14, fontweight="bold", color=NV_DARK, zorder=7)

    ax.text(
        5.5,
        7.85,
        "Three orthogonal parallel axes (conceptual)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=NV_DARK,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=NV_DARK, linewidth=1.0),
    )

    origin = (1.45, 2.85)
    tp_end = (7.35, 2.85)
    pp_end = (1.45, 6.05)
    ep_end = (6.55, 5.78)

    arr = dict(arrowstyle="-|>", mutation_scale=17, lw=2.5)
    ax.annotate("", xy=tp_end, xytext=origin, arrowprops={**arr, "color": NV_DARK, "zorder": 2})
    ax.annotate("", xy=pp_end, xytext=origin, arrowprops={**arr, "color": NV, "zorder": 2})
    ax.annotate("", xy=ep_end, xytext=origin, arrowprops={**arr, "color": NV_MID, "zorder": 2})

    ax.plot(*origin, "o", color=NV_DARK, markersize=9, zorder=4)
    # Tucked under / beside the corner — not centered on the diagram
    ax.text(
        origin[0] - 0.42,
        origin[1] - 0.18,
        "Rank grid\norigin",
        fontsize=8.5,
        color=INK_MUTED,
        ha="right",
        va="top",
        fontweight="normal",
        zorder=5,
    )

    # Bold axis-tip labels (TP / PP / EP) — separate from the pale callout boxes
    ax.text(tp_end[0] + 0.2, tp_end[1] - 0.18, "TP", va="bottom", **tip_kw)
    ax.text(pp_end[0] - 0.58, pp_end[1] + 0.12, "PP", va="bottom", ha="left", **tip_kw)
    ax.text(ep_end[0] - 0.32, ep_end[1] + 0.06, "EP", va="bottom", **tip_kw)

    # TP callout: nudge lower to clear the axis tip / EP region
    _callout(
        5.95,
        3.06,
        2.85,
        1.12,
        "Tensor parallel (TP)",
        "• Split matrices across ranks\n• All-reduce activations / layer",
    )
    _callout(
        0.12,
        3.88,
        2.55,
        1.05,
        "Pipeline parallel (PP)",
        "• Stages along depth\n• Bubble + microbatch schedule",
    )
    # EP callout: nudge left so it clears neighboring labels / geometry
    _callout(
        6.72,
        6.05,
        2.78,
        1.05,
        "Expert parallel (EP)",
        "• Shard experts across ranks\n• Allgather / alltoall tokens",
    )

    # Narrower foot panel; extra height so bullets stay inside rounded rect
    foot_w = 7.85
    foot_x = (11.0 - foot_w) / 2
    foot_h = 1.22
    foot_y = -0.44
    foot = mpatches.FancyBboxPatch(
        (foot_x, foot_y),
        foot_w,
        foot_h,
        boxstyle="round,pad=0.08",
        facecolor=NV_LIGHT,
        edgecolor=NV_DARK,
        linewidth=1.25,
        zorder=3,
    )
    ax.add_patch(foot)
    top_inner = foot_y + foot_h - 0.1
    ax.text(
        foot_x + foot_w / 2,
        top_inner,
        "MoE rule of thumb",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    ax.text(
        foot_x + foot_w / 2,
        top_inner - 0.26,
        "• Grow EP toward the number of experts first (spread expert weights).\n"
        "• Add TP when one expert (or adjacent stack) still exceeds single-GPU memory.\n"
        "• Add PP when full depth still does not fit after EP / TP.",
        ha="center",
        va="top",
        fontsize=7.95,
        fontweight="normal",
        color=INK,
        linespacing=1.34,
        zorder=5,
    )

    save(fig, "moe_parallel_axes.png", pad_inches=0.45)


def fig_moe_layer_schematic() -> None:
    """Single MoE layer: input → router (top-2) → gated experts → sum → output. NVIDIA greens."""
    NV = "#76B900"
    NV_DARK = "#5A8700"
    NV_MID = "#8BC727"
    NV_LIGHT = "#D4EDAA"
    NV_TINT = "#EEF8E0"
    INK = "#1A1A1A"
    MUTED = "#5C6F4A"

    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    # No suptitle — it leaves a large blank band. Use a tight data-space title instead (see end).
    fig.subplots_adjust(left=0.06, right=0.94, top=0.995, bottom=0.05)
    ax.set_xlim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Inner frame: strictly between Input (below) and Output (above).
    box_x0, box_y0 = 0.95, 1.68
    box_w, box_h = 8.1, 6.22
    box_y1 = box_y0 + box_h
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (box_x0, box_y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.05,rounding_size=0.38",
            facecolor=NV_TINT,
            edgecolor=NV,
            linewidth=1.85,
            zorder=0,
        )
    )
    ax.text(
        box_x0 + 0.12,
        box_y1 - 0.12,
        "MoE Layer",
        fontsize=12,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
        zorder=10,
    )

    cx = 5.0
    cy_router = 2.48
    r_router = 0.58
    ex_y = 4.12
    ex_w, ex_h = 1.45, 0.92
    ex_xs = (2.62, 5.0, 7.38)
    r_mul = 0.34
    x_mul = ((ex_xs[0], 5.88), (ex_xs[2], 5.88))
    c_plus = (5.0, 7.08)
    r_plus = 0.34

    def expert_box(xc: float, yc: float) -> tuple[float, float, float, float]:
        return (xc - ex_w / 2, yc - ex_h / 2, ex_w, ex_h)

    def rim_from_router(tx: float, ty: float, *, pad: float = 0.95) -> tuple[float, float]:
        """Point on the router circle toward (tx, ty), for clean arrow starts."""
        dx, dy = tx - cx, ty - cy_router
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return cx, cy_router + r_router * pad
        s = r_router * pad / d
        return cx + dx * s, cy_router + dy * s

    # Experts
    for xc in ex_xs:
        x0, y0, w, h = expert_box(xc, ex_y)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=NV_LIGHT,
                edgecolor=NV,
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(
            xc,
            ex_y - 0.22,
            "Expert",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
            zorder=3,
        )

    # Router circle (see set_box_aspect at end for true circles on export)
    ax.add_patch(
        plt.Circle((cx, cy_router), r_router, facecolor=NV_LIGHT, edgecolor=NV_DARK, linewidth=1.85, zorder=2)
    )
    # Label to the right of the disk so upward routing arrows stay visible.
    ax.text(
        cx + r_router + 0.38,
        cy_router,
        "Router",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
        zorder=4,
    )

    # Mini bar chart inside router — centered on (cx, cy_router), same floor for all bars.
    bw = 0.2
    bar_gap = 0.07
    heights = (0.36, 0.52, 0.32)
    h_max = max(heights)
    cluster_w = 3 * bw + 2 * bar_gap
    bx0 = cx - cluster_w / 2
    by0 = cy_router - h_max / 2
    bar_cols = (NV_MID, NV_DARK, "#9BCB3D")
    for j, (h, col) in enumerate(zip(heights, bar_cols, strict=True)):
        ax.add_patch(
            mpatches.Rectangle(
                (bx0 + j * (bw + bar_gap), by0),
                bw,
                h,
                facecolor=col,
                edgecolor=NV_DARK,
                linewidth=0.55,
                zorder=3,
            )
        )

    # Input (outside box)
    ax.text(cx, 0.36, "Input", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.add_patch(
        FancyArrowPatch(
            (cx, 0.72),
            (cx, cy_router - r_router - 0.03),
            arrowstyle="-|>",
            mutation_scale=13,
            color=NV_DARK,
            lw=1.55,
            zorder=1,
        )
    )

    # Router → experts (top-2: left + right solid, middle dashed); start on rim toward each expert.
    def _router_to_expert(xt: float, yt: float, *, dashed: bool) -> None:
        sx, sy = rim_from_router(xt, yt)
        ax.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (xt, yt - ex_h / 2 - 0.02),
                arrowstyle="-|>",
                mutation_scale=10,
                color=MUTED if dashed else NV_DARK,
                lw=1.45 if not dashed else 1.15,
                linestyle="--" if dashed else "solid",
                zorder=1,
            )
        )

    for xt, dashed in zip(ex_xs, (False, True, False), strict=True):
        _router_to_expert(xt, ex_y, dashed=dashed)

    def tiny_gate_icon(gx: float, gy: float, highlight: int, *, gw: float = 0.1) -> None:
        """Three micro-bars inside expert box; `highlight` in {0,1,2}."""
        gap = 0.03
        for k in range(3):
            h = 0.11 + 0.08 * (k == 1)
            col = NV if k == highlight else "#C5D9B0"
            ax.add_patch(
                mpatches.Rectangle(
                    (gx + k * (gw + gap), gy),
                    gw,
                    h,
                    facecolor=col,
                    edgecolor=NV_DARK,
                    linewidth=0.35,
                    zorder=4,
                )
            )

    def gate_inside_expert(xc: float, highlight: int) -> None:
        x0, y0, w, h = expert_box(xc, ex_y)
        gw, gap = 0.1, 0.03
        gate_w = 3 * gw + 2 * gap
        max_h = 0.19
        gx = x0 + w - gate_w - 0.12
        gy = y0 + h - max_h - 0.12
        tiny_gate_icon(gx, gy, highlight, gw=gw)

    gate_inside_expert(ex_xs[0], 0)
    gate_inside_expert(ex_xs[2], 2)

    # Expert → multiply nodes
    for xc, (xm, ym) in zip((ex_xs[0], ex_xs[2]), x_mul, strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (xc, ex_y + ex_h / 2 + 0.02),
                (xm, ym - r_mul - 0.02),
                arrowstyle="-|>",
                mutation_scale=9,
                color=NV_DARK,
                lw=1.25,
                zorder=1,
            )
        )

    # Router-side weights into multiply (curved), starting on rim toward each multiply node.
    for xm, ym in x_mul:
        sx, sy = rim_from_router(xm, ym)
        ax.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (xm, ym - r_mul - 0.02),
                arrowstyle="-|>",
                mutation_scale=8,
                color=NV_MID,
                lw=1.1,
                connectionstyle="arc3,rad=0.18" if xm < cx else "arc3,rad=-0.18",
                zorder=1,
            )
        )

    for xm, ym in x_mul:
        ax.add_patch(plt.Circle((xm, ym), r_mul, facecolor="white", edgecolor=NV, linewidth=1.65, zorder=3))
        ax.text(xm, ym, "×", ha="center", va="center", fontsize=15, fontweight="bold", color=INK, zorder=4)

    for xm, ym in x_mul:
        ax.add_patch(
            FancyArrowPatch(
                (xm, ym + r_mul + 0.02),
                (c_plus[0], c_plus[1] - r_plus - 0.02),
                arrowstyle="-|>",
                mutation_scale=9,
                color=NV_DARK,
                lw=1.25,
                zorder=1,
            )
        )

    ax.add_patch(plt.Circle(c_plus, r_plus, facecolor="white", edgecolor=NV, linewidth=1.65, zorder=3))
    ax.text(c_plus[0], c_plus[1], "+", ha="center", va="center", fontsize=17, fontweight="bold", color=INK, zorder=4)

    # Output: label and arrow tip sit above the MoE frame.
    y_arrow_tip = box_y1 + 0.52
    ax.add_patch(
        FancyArrowPatch(
            (c_plus[0], c_plus[1] + r_plus + 0.02),
            (c_plus[0], y_arrow_tip),
            arrowstyle="-|>",
            mutation_scale=11,
            color=NV_DARK,
            lw=1.5,
            zorder=1,
        )
    )
    ax.text(c_plus[0], y_arrow_tip + 0.26, "Output", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)

    # Title stacked just above Output (avoids fig.suptitle's extra vertical band).
    y_out_label = y_arrow_tip + 0.26
    y_title = y_out_label + 0.6
    ax.text(
        cx,
        y_title,
        "Top-2 routing (example)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
        zorder=25,
    )
    ax.set_ylim(0, y_title + 0.38)

    # One data unit same length on screen in x and y (true circles); box aspect from data limits.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_box_aspect((ymax - ymin) / (xmax - xmin))
    save(fig, "moe_layer_schematic.png", pad_inches=0.14)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_routing_topk()
    fig_parallel_axes()
    fig_moe_layer_schematic()
    for p in sorted(OUT.glob("moe_*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
