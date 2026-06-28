"""Visual diagnostics for trilateration solutions."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from trilateration.coordinates import ecef_to_enu_matrix, geodetic_to_ecef
from trilateration.solver import Anchor, TrilaterationResult


def plot_solution(
    anchors: list[Anchor],
    result: TrilaterationResult,
    output_path: Path,
    *,
    title: str = "Altitude-constrained 3D trilateration",
) -> None:
    """Write a PNG plot of anchors, estimated target, and range circles."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise RuntimeError('Plotting requires matplotlib. Install with: python -m pip install -e ".[dev]"') from exc

    target_ecef = geodetic_to_ecef(result.lat_deg, result.lon_deg, result.alt_m)
    rot = ecef_to_enu_matrix(result.lat_deg, result.lon_deg)

    anchor_enu = np.vstack(
        [rot @ (geodetic_to_ecef(anchor.lat_deg, anchor.lon_deg, anchor.alt_m) - target_ecef) for anchor in anchors]
    )

    fig = plt.figure(figsize=(12, 8), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.1])
    ax = fig.add_subplot(grid[0, 0])
    alt_ax = fig.add_subplot(grid[0, 1])

    ax.scatter(anchor_enu[:, 0], anchor_enu[:, 1], s=95, color="#2563eb", label="GPS-good anchors", zorder=3)
    ax.scatter([0.0], [0.0], s=130, marker="x", linewidths=3, color="#dc2626", label="Estimated target", zorder=4)

    for idx, (anchor, point, residual) in enumerate(zip(anchors, anchor_enu, result.residuals_m, strict=True)):
        label = anchor.node_id or f"anchor_{idx}"
        vertical_delta_m = point[2]
        horizontal_radius_sq = anchor.distance_m**2 - vertical_delta_m**2
        horizontal_radius_m = math.sqrt(max(0.0, horizontal_radius_sq))

        circle = Circle(
            (point[0], point[1]),
            horizontal_radius_m,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            edgecolor="#64748b",
            alpha=0.85,
        )
        ax.add_patch(circle)
        ax.plot([point[0], 0.0], [point[1], 0.0], color="#94a3b8", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"{label}\nrange {anchor.distance_m:.1f} m\nalt dz {vertical_delta_m:+.1f} m\nres {residual.residual_m:+.2f} m",
            xy=(point[0], point[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
        )

        if horizontal_radius_sq < 0.0:
            ax.text(
                point[0],
                point[1],
                "range < altitude delta",
                fontsize=8,
                color="#b91c1c",
                ha="center",
                va="top",
            )

    summary = (
        f"target: {result.lat_deg:.8f}, {result.lon_deg:.8f}, {result.alt_m:.2f} m\n"
        f"RMS residual: {result.rms_error_m:.3f} m\n"
        f"geometry condition: {result.geometry_condition:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#94a3b8", "alpha": 0.95},
    )

    ax.set_title(title)
    ax.set_xlabel("East from estimated target, m")
    ax.set_ylabel("North from estimated target, m")
    ax.axhline(0.0, color="#e2e8f0", linewidth=1.0)
    ax.axvline(0.0, color="#e2e8f0", linewidth=1.0)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    _set_plot_limits(ax, anchor_enu, anchors)
    _plot_altitudes(alt_ax, anchors, result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _set_plot_limits(ax, anchor_enu: np.ndarray, anchors: list[Anchor]) -> None:
    radii = []
    for anchor, point in zip(anchors, anchor_enu, strict=True):
        radii.append(math.sqrt(max(0.0, anchor.distance_m**2 - point[2] ** 2)))

    east_values = [0.0]
    north_values = [0.0]
    for point, radius in zip(anchor_enu, radii, strict=True):
        east_values.extend([float(point[0] - radius), float(point[0] + radius)])
        north_values.extend([float(point[1] - radius), float(point[1] + radius)])

    east_min, east_max = min(east_values), max(east_values)
    north_min, north_max = min(north_values), max(north_values)
    span = max(east_max - east_min, north_max - north_min, 1.0)
    margin = span * 0.08
    ax.set_xlim(east_min - margin, east_max + margin)
    ax.set_ylim(north_min - margin, north_max + margin)


def _plot_altitudes(ax, anchors: list[Anchor], result: TrilaterationResult) -> None:
    labels = [anchor.node_id or f"A{idx}" for idx, anchor in enumerate(anchors)]
    anchor_alts = [anchor.alt_m for anchor in anchors]
    x = np.arange(len(anchors), dtype=float)

    ax.scatter(x, anchor_alts, s=90, color="#2563eb", label="Anchors", zorder=3)
    ax.axhline(result.alt_m, color="#dc2626", linewidth=2.0, linestyle="-", label="Target altitude")

    for idx, (label, alt_m) in enumerate(zip(labels, anchor_alts, strict=True)):
        ax.plot([idx, idx], [result.alt_m, alt_m], color="#94a3b8", linewidth=1.2)
        ax.annotate(
            f"{label}\n{alt_m:.1f} m\n{alt_m - result.alt_m:+.1f} m",
            xy=(idx, alt_m),
            xytext=(0, 9 if alt_m >= result.alt_m else -42),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
        )

    ax.text(
        0.5,
        result.alt_m,
        f"target\n{result.alt_m:.1f} m",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#dc2626",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#fecaca", "alpha": 0.94},
    )
    ax.set_title("Altitude")
    ax.set_ylabel("Altitude, m")
    ax.set_xticks(x, labels)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_xlim(-0.6, len(anchors) - 0.4)

    all_alts = anchor_alts + [result.alt_m]
    span = max(max(all_alts) - min(all_alts), 1.0)
    margin = max(span * 0.35, 5.0)
    ax.set_ylim(min(all_alts) - margin, max(all_alts) + margin)
