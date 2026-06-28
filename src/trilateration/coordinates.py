"""WGS84 coordinate conversion helpers."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A_M**2 - WGS84_B_M**2) / WGS84_B_M**2


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> NDArray[np.float64]:
    """Convert WGS84 geodetic coordinates to ECEF XYZ meters."""

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(ecef_m: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert ECEF XYZ meters to WGS84 geodetic coordinates.

    Uses Bowring's closed-form initial solution, which is accurate enough for
    the short-distance navigation use case here.
    """

    x, y, z = map(float, ecef_m)
    p = math.hypot(x, y)
    if p == 0.0:
        lat_deg = 90.0 if z >= 0.0 else -90.0
        return lat_deg, 0.0, abs(z) - WGS84_B_M

    theta = math.atan2(z * WGS84_A_M, p * WGS84_B_M)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    lat = math.atan2(
        z + WGS84_EP2 * WGS84_B_M * sin_theta**3,
        p - WGS84_E2 * WGS84_A_M * cos_theta**3,
    )
    lon = math.atan2(y, x)

    sin_lat = math.sin(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt_m = p / math.cos(lat) - n

    return math.degrees(lat), _normalize_lon_deg(math.degrees(lon)), alt_m


def ecef_to_enu_matrix(ref_lat_deg: float, ref_lon_deg: float) -> NDArray[np.float64]:
    """Return the rotation matrix that maps ECEF deltas to local ENU."""

    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def enu_to_ecef(
    enu_m: NDArray[np.float64],
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> NDArray[np.float64]:
    """Convert a local ENU point to ECEF XYZ meters."""

    ref_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    rot = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    return ref_ecef + rot.T @ enu_m


def _normalize_lon_deg(lon_deg: float) -> float:
    return ((lon_deg + 180.0) % 360.0) - 180.0
