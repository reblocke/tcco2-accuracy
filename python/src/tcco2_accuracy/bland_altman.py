"""Bland-Altman limits of agreement utilities.

Uses Conway's sign convention where d = PaCO2 - TcCO2.
"""

from __future__ import annotations

import math
from typing import Tuple


def total_sd(within_study_sd: float, between_study_variance: float) -> float:
    """Return pooled SD for Bland-Altman LoA.

    Uses SD_total = sqrt(sigma^2 + tau^2).
    """

    return math.sqrt(within_study_sd**2 + between_study_variance)


def loa_bounds(
    delta: float,
    within_study_sd: float,
    between_study_variance: float,
    multiplier: float = 2.0,
) -> Tuple[float, float]:
    """Compute lower/upper limits of agreement.

    LoA = delta ± multiplier * SD_total.
    """

    pooled_sd = total_sd(within_study_sd, between_study_variance)
    spread = multiplier * pooled_sd
    return delta - spread, delta + spread
