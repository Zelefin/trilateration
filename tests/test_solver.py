from __future__ import annotations

import math

import numpy as np
import pytest

from trilateration import Anchor, TrilaterationError, geodetic_to_ecef, solve_position


TARGET = (50.4529, 30.5268, 183.5)
ANCHOR_COORDS = [
    ("A", 50.4501, 30.5234, 180.0),
    ("B", 50.4565, 30.5201, 190.0),
    ("C", 50.4510, 30.5340, 175.0),
]


def _range_m(anchor: tuple[str, float, float, float], target: tuple[float, float, float]) -> float:
    _, lat, lon, alt = anchor
    return float(np.linalg.norm(geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(*target)))


def _anchors(target: tuple[float, float, float] = TARGET) -> list[Anchor]:
    return [
        Anchor(lat_deg=lat, lon_deg=lon, alt_m=alt, distance_m=_range_m(anchor, target), node_id=node_id)
        for anchor, (node_id, lat, lon, alt) in zip(ANCHOR_COORDS, ANCHOR_COORDS, strict=True)
    ]


def test_exact_synthetic_solution_recovers_target() -> None:
    result = solve_position(_anchors(), target_alt_m=TARGET[2])

    assert result.converged
    assert math.isclose(result.lat_deg, TARGET[0], abs_tol=1e-9)
    assert math.isclose(result.lon_deg, TARGET[1], abs_tol=1e-9)
    assert math.isclose(result.alt_m, TARGET[2], abs_tol=0.0)
    assert result.rms_error_m < 1e-6


def test_noisy_ranges_remain_close() -> None:
    noisy_offsets = [1.2, -0.8, 0.5]
    anchors = [
        Anchor(
            lat_deg=anchor.lat_deg,
            lon_deg=anchor.lon_deg,
            alt_m=anchor.alt_m,
            distance_m=anchor.distance_m + offset,
            node_id=anchor.node_id,
        )
        for anchor, offset in zip(_anchors(), noisy_offsets, strict=True)
    ]

    result = solve_position(anchors, target_alt_m=TARGET[2])

    assert result.converged
    assert abs(result.lat_deg - TARGET[0]) < 3e-5
    assert abs(result.lon_deg - TARGET[1]) < 3e-5
    assert result.rms_error_m < 1.5


def test_wrong_altitude_changes_horizontal_solution() -> None:
    correct = solve_position(_anchors(), target_alt_m=TARGET[2])
    wrong = solve_position(_anchors(), target_alt_m=TARGET[2] + 50.0)

    horizontal_shift_deg = math.hypot(wrong.lat_deg - correct.lat_deg, wrong.lon_deg - correct.lon_deg)

    assert horizontal_shift_deg > 1e-7
    assert wrong.rms_error_m > correct.rms_error_m


@pytest.mark.parametrize(
    "anchors",
    [
        [],
        _anchors()[:2],
        _anchors() + [Anchor(50.0, 30.0, 100.0, 100.0)],
    ],
)
def test_rejects_wrong_anchor_count(anchors: list[Anchor]) -> None:
    with pytest.raises(TrilaterationError, match="exactly 3"):
        solve_position(anchors, target_alt_m=TARGET[2])


def test_rejects_invalid_measurements() -> None:
    anchors = _anchors()
    anchors[0] = Anchor(lat_deg=95.0, lon_deg=30.0, alt_m=180.0, distance_m=100.0)
    with pytest.raises(TrilaterationError, match="lat_deg"):
        solve_position(anchors, target_alt_m=TARGET[2])

    anchors = _anchors()
    anchors[0] = Anchor(lat_deg=50.0, lon_deg=30.0, alt_m=180.0, distance_m=-1.0)
    with pytest.raises(TrilaterationError, match="distance_m"):
        solve_position(anchors, target_alt_m=TARGET[2])


def test_rejects_poorly_conditioned_anchor_geometry() -> None:
    target = (50.0005, 30.0, 100.0)
    coords = [
        ("A", 50.0000, 30.0, 100.0),
        ("B", 50.0010, 30.0, 100.0),
        ("C", 50.0020, 30.0, 100.0),
    ]
    anchors = [
        Anchor(lat, lon, alt, float(np.linalg.norm(geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(*target))), node_id)
        for node_id, lat, lon, alt in coords
    ]

    with pytest.raises(TrilaterationError, match="poorly conditioned"):
        solve_position(anchors, target_alt_m=target[2])


def test_rejects_high_residual_when_limit_is_set() -> None:
    anchors = _anchors()
    anchors[0] = Anchor(
        lat_deg=anchors[0].lat_deg,
        lon_deg=anchors[0].lon_deg,
        alt_m=anchors[0].alt_m,
        distance_m=anchors[0].distance_m + 500.0,
        node_id=anchors[0].node_id,
    )

    with pytest.raises(TrilaterationError, match="RMS residual"):
        solve_position(anchors, target_alt_m=TARGET[2], max_rms_residual_m=1.0)
