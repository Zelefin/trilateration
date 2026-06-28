"""Altitude-constrained trilateration solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from trilateration.coordinates import (
    ecef_to_enu_matrix,
    ecef_to_geodetic,
    enu_to_ecef,
    geodetic_to_ecef,
)


class TrilaterationError(ValueError):
    """Raised when inputs cannot produce a useful trilateration solution."""


@dataclass(frozen=True)
class Anchor:
    lat_deg: float
    lon_deg: float
    alt_m: float
    distance_m: float
    node_id: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "Anchor":
        return cls(
            lat_deg=float(data["lat_deg"]),
            lon_deg=float(data["lon_deg"]),
            alt_m=float(data["alt_m"]),
            distance_m=float(data["distance_m"]),
            node_id=None if data.get("node_id") is None else str(data["node_id"]),
        )


@dataclass(frozen=True)
class RangeResidual:
    anchor: str
    expected_distance_m: float
    observed_distance_m: float
    residual_m: float


@dataclass(frozen=True)
class TrilaterationResult:
    lat_deg: float
    lon_deg: float
    alt_m: float
    residuals_m: list[RangeResidual]
    rms_error_m: float
    max_abs_error_m: float
    converged: bool
    iterations: int
    geometry_condition: float
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "alt_m": self.alt_m,
            "residuals_m": [asdict(residual) for residual in self.residuals_m],
            "rms_error_m": self.rms_error_m,
            "max_abs_error_m": self.max_abs_error_m,
            "converged": self.converged,
            "iterations": self.iterations,
            "geometry_condition": self.geometry_condition,
            "message": self.message,
        }


def solve_position(
    anchors: list[Anchor],
    target_alt_m: float,
    *,
    initial_guess: tuple[float, float] | None = None,
    max_rms_residual_m: float | None = None,
) -> TrilaterationResult:
    """Solve target latitude and longitude from three anchors and target altitude.

    The optimizer minimizes the difference between measured TWR 3D ranges and
    the ECEF distance from each anchor to the candidate target point on the
    fixed-altitude WGS84 surface.
    """

    _validate_inputs(anchors, target_alt_m, initial_guess, max_rms_residual_m)

    anchor_ecef = np.vstack([geodetic_to_ecef(anchor.lat_deg, anchor.lon_deg, anchor.alt_m) for anchor in anchors])
    observed = np.array([anchor.distance_m for anchor in anchors], dtype=np.float64)
    geometry_condition = _geometry_condition(anchors)
    if not math.isfinite(geometry_condition) or geometry_condition > 1.0e8:
        raise TrilaterationError(f"Anchor geometry is poorly conditioned: {geometry_condition:g}")

    if initial_guess is None:
        initial_guess = _initial_guess_from_local_solution(anchors, target_alt_m)

    def residual_fn(params: NDArray[np.float64]) -> NDArray[np.float64]:
        lat_deg = math.degrees(float(params[0]))
        lon_deg = math.degrees(float(params[1]))
        target = geodetic_to_ecef(lat_deg, lon_deg, target_alt_m)
        predicted = np.linalg.norm(anchor_ecef - target, axis=1)
        return predicted - observed

    x0 = np.radians(np.array(initial_guess, dtype=np.float64))
    result = least_squares(
        residual_fn,
        x0=x0,
        bounds=([-math.pi / 2.0, -math.pi], [math.pi / 2.0, math.pi]),
        x_scale=np.array([1.0e-5, 1.0e-5]),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=200,
    )

    lat_deg = math.degrees(float(result.x[0]))
    lon_deg = _normalize_lon_deg(math.degrees(float(result.x[1])))
    final_residual_values = residual_fn(result.x)
    rms_error_m = float(np.sqrt(np.mean(np.square(final_residual_values))))
    max_abs_error_m = float(np.max(np.abs(final_residual_values)))

    if max_rms_residual_m is not None and rms_error_m > max_rms_residual_m:
        raise TrilaterationError(
            f"Best solution RMS residual {rms_error_m:.3f} m exceeds limit {max_rms_residual_m:.3f} m"
        )

    residuals = [
        RangeResidual(
            anchor=anchor.node_id or f"anchor_{idx}",
            expected_distance_m=float(anchor.distance_m + residual),
            observed_distance_m=float(anchor.distance_m),
            residual_m=float(residual),
        )
        for idx, (anchor, residual) in enumerate(zip(anchors, final_residual_values, strict=True))
    ]

    return TrilaterationResult(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=float(target_alt_m),
        residuals_m=residuals,
        rms_error_m=rms_error_m,
        max_abs_error_m=max_abs_error_m,
        converged=bool(result.success),
        iterations=int(result.nfev),
        geometry_condition=geometry_condition,
        message=str(result.message),
    )


def _validate_inputs(
    anchors: list[Anchor],
    target_alt_m: float,
    initial_guess: tuple[float, float] | None,
    max_rms_residual_m: float | None,
) -> None:
    if len(anchors) != 3:
        raise TrilaterationError(f"Expected exactly 3 anchors, got {len(anchors)}")
    if not math.isfinite(target_alt_m):
        raise TrilaterationError("target_alt_m must be finite")
    if max_rms_residual_m is not None and (not math.isfinite(max_rms_residual_m) or max_rms_residual_m < 0.0):
        raise TrilaterationError("max_rms_residual_m must be a non-negative finite value")
    if initial_guess is not None:
        _validate_lat_lon(initial_guess[0], initial_guess[1], "initial_guess")

    for idx, anchor in enumerate(anchors):
        prefix = f"anchors[{idx}]"
        _validate_lat_lon(anchor.lat_deg, anchor.lon_deg, prefix)
        if not math.isfinite(anchor.alt_m):
            raise TrilaterationError(f"{prefix}.alt_m must be finite")
        if not math.isfinite(anchor.distance_m) or anchor.distance_m <= 0.0:
            raise TrilaterationError(f"{prefix}.distance_m must be positive and finite")


def _validate_lat_lon(lat_deg: float, lon_deg: float, prefix: str) -> None:
    if not math.isfinite(lat_deg) or not -90.0 <= lat_deg <= 90.0:
        raise TrilaterationError(f"{prefix}.lat_deg must be in [-90, 90]")
    if not math.isfinite(lon_deg) or not -180.0 <= lon_deg <= 180.0:
        raise TrilaterationError(f"{prefix}.lon_deg must be in [-180, 180]")


def _geometry_condition(anchors: list[Anchor]) -> float:
    ref_lat = sum(anchor.lat_deg for anchor in anchors) / len(anchors)
    ref_lon = sum(anchor.lon_deg for anchor in anchors) / len(anchors)
    ref_alt = sum(anchor.alt_m for anchor in anchors) / len(anchors)
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    rot = ecef_to_enu_matrix(ref_lat, ref_lon)
    points = np.vstack(
        [
            rot @ (geodetic_to_ecef(anchor.lat_deg, anchor.lon_deg, anchor.alt_m) - ref_ecef)
            for anchor in anchors
        ]
    )
    horizontal = points[:, :2] - np.mean(points[:, :2], axis=0)
    singular_values = np.linalg.svd(horizontal, compute_uv=False)
    if len(singular_values) < 2 or singular_values[-1] < 1.0e-6:
        return math.inf
    return float(singular_values[0] / singular_values[-1])


def _initial_guess_from_local_solution(anchors: list[Anchor], target_alt_m: float) -> tuple[float, float]:
    ref_lat = sum(anchor.lat_deg for anchor in anchors) / len(anchors)
    ref_lon = sum(anchor.lon_deg for anchor in anchors) / len(anchors)
    ref_alt = float(target_alt_m)
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    rot = ecef_to_enu_matrix(ref_lat, ref_lon)

    anchor_enu = np.vstack(
        [
            rot @ (geodetic_to_ecef(anchor.lat_deg, anchor.lon_deg, anchor.alt_m) - ref_ecef)
            for anchor in anchors
        ]
    )

    first = anchor_enu[0]
    first_range_sq = anchors[0].distance_m**2 - first[2] ** 2
    rows: list[list[float]] = []
    rhs: list[float] = []
    for anchor, point in zip(anchors[1:], anchor_enu[1:], strict=True):
        range_sq = anchor.distance_m**2 - point[2] ** 2
        rows.append([2.0 * (point[0] - first[0]), 2.0 * (point[1] - first[1])])
        rhs.append(
            first_range_sq
            - range_sq
            + point[0] ** 2
            + point[1] ** 2
            - first[0] ** 2
            - first[1] ** 2
        )

    try:
        local_xy, *_ = np.linalg.lstsq(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64), rcond=None)
        candidate_ecef = enu_to_ecef(np.array([local_xy[0], local_xy[1], 0.0]), ref_lat, ref_lon, ref_alt)
        lat_deg, lon_deg, _ = ecef_to_geodetic(candidate_ecef)
        return lat_deg, lon_deg
    except np.linalg.LinAlgError:
        return ref_lat, ref_lon


def _normalize_lon_deg(lon_deg: float) -> float:
    return ((lon_deg + 180.0) % 360.0) - 180.0
