"""Optional command scaling for smoother execution of saved reaching policies."""

import numpy as np


def slow_near_target(action, distance, *, radius=0.08, minimum_scale=0.2):
    """Reduce command magnitude near the goal while preserving its direction.

    This is execution-time postprocessing, not a change to the saved policy.
    A radius of zero disables the filter for unmodified-policy comparisons.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Slowdown radius must be finite and nonnegative")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Target distance must be finite and nonnegative")
    if not 0 < minimum_scale <= 1:
        raise ValueError("Minimum command scale must be in (0, 1]")
    scale = 1.0 if radius == 0 else float(np.clip(distance / radius, minimum_scale, 1.0))
    return np.asarray(action) * scale
