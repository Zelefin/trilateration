from __future__ import annotations

import math

import numpy as np

from trilateration.coordinates import ecef_to_geodetic, geodetic_to_ecef


def test_geodetic_ecef_round_trip() -> None:
    original = (50.4529, 30.5268, 183.5)

    ecef = geodetic_to_ecef(*original)
    recovered = ecef_to_geodetic(ecef)

    assert math.isclose(recovered[0], original[0], abs_tol=1e-9)
    assert math.isclose(recovered[1], original[1], abs_tol=1e-9)
    assert math.isclose(recovered[2], original[2], abs_tol=1e-5)
    assert np.isfinite(ecef).all()
