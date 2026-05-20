"""Global RNG seeding. Call ``set_global_seed()`` once at process start."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 20260519


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
