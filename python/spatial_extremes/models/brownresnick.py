from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spatial_extremes._rust import (
    brownresnick_simulate,
    brownresnick_extremal_coefficient,
    brownresnick_bivariate_cdf,
    brownresnick_bivariate_logpdf,
)


def _rng_to_seed(rng):
    if rng is None:
        return np.random.default_rng().integers(0, 2**63, dtype=np.uint64)
    return rng.integers(0, 2**63, dtype=np.uint64)


@dataclass
class BrownResnickModel:
    range_: float
    alpha: float = 1.0
    variogram_fn: Callable = None

    n_free_params = 2
    free_param_names = ("log_range", "logit_alpha")

    def __post_init__(self):
        if self.range_ <= 0:
            raise ValueError("range_ must be positive")
        if not (0 < self.alpha <= 2):
            raise ValueError("alpha must be in (0, 2]")
        if self.variogram_fn is None:
            self.variogram_fn = lambda h, r, a: (np.linalg.norm(h) / r) ** a
        self._rust_params = [float(self.range_), float(self.alpha)]

    @classmethod
    def from_free_params(cls, free_params, **kwargs) -> "BrownResnickModel":
        log_range, logit_alpha = free_params
        range_ = np.exp(log_range)
        alpha = 2.0 / (1.0 + np.exp(-logit_alpha))
        return cls(range_=range_, alpha=alpha, **kwargs)

    def _gamma(self, h: np.ndarray) -> float:
        h = np.asarray(h, dtype=float)
        return float(self.variogram_fn(h, self.range_, self.alpha))

    def extremal_coefficient(self, h: np.ndarray) -> float:
        return brownresnick_extremal_coefficient(np.asarray(h, dtype=float).flatten()[:2], *self._rust_params)

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        return brownresnick_bivariate_cdf(z1, z2, np.asarray(h, dtype=float).flatten()[:2], *self._rust_params)

    def bivariate_logpdf(self, z1, z2, h: np.ndarray):
        return brownresnick_bivariate_logpdf(
            np.asarray(z1, dtype=float),
            np.asarray(z2, dtype=float),
            np.asarray(h, dtype=float).flatten()[:2],
            *self._rust_params,
        )

    def simulate(self, coords, n_reps=1, max_points=20_000, batch_size=256, rng=None):
        return brownresnick_simulate(
            np.asarray(coords, dtype=float),
            n_reps,
            self.range_,
            self.alpha,
            max_points,
            batch_size,
            int(_rng_to_seed(rng)),
        )
