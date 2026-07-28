from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spatial_extremes._rust import (
    schlather_simulate,
    schlather_extremal_coefficient,
    schlather_bivariate_cdf,
    schlather_bivariate_logpdf,
)


def _rng_to_seed(rng):
    if rng is None:
        return np.random.default_rng().integers(0, 2**63, dtype=np.uint64)
    return rng.integers(0, 2**63, dtype=np.uint64)


@dataclass
class SchlatherModel:
    range_: float
    smoothness: float = 1.0
    correlation_fn: Callable = None

    n_free_params = 2
    free_param_names = ("log_range", "logit_smoothness")

    def __post_init__(self):
        if self.range_ <= 0:
            raise ValueError("range_ must be positive")
        if not (0 < self.smoothness <= 2):
            raise ValueError("smoothness must be in (0, 2]")
        if self.correlation_fn is None:
            self.correlation_fn = lambda h, r, s: np.exp(-((np.linalg.norm(h) / r) ** s))
        self._rust_params = [float(self.range_), float(self.smoothness)]

    @classmethod
    def from_free_params(cls, free_params, **kwargs) -> "SchlatherModel":
        log_range, logit_smoothness = free_params
        range_ = np.exp(log_range)
        smoothness = 2.0 / (1.0 + np.exp(-logit_smoothness))
        return cls(range_=range_, smoothness=smoothness, **kwargs)

    def _rho(self, h: np.ndarray) -> float:
        h = np.asarray(h, dtype=float)
        rho = float(self.correlation_fn(h, self.range_, self.smoothness))
        return np.clip(rho, -0.999999, 0.999999)

    def extremal_coefficient(self, h: np.ndarray) -> float:
        return schlather_extremal_coefficient(np.asarray(h, dtype=float).flatten()[:2], *self._rust_params)

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        return schlather_bivariate_cdf(z1, z2, np.asarray(h, dtype=float).flatten()[:2], *self._rust_params)

    def bivariate_logpdf(self, z1, z2, h: np.ndarray):
        return schlather_bivariate_logpdf(
            np.asarray(z1, dtype=float),
            np.asarray(z2, dtype=float),
            np.asarray(h, dtype=float).flatten()[:2],
            *self._rust_params,
        )

    def simulate(self, coords, n_reps=1, max_points=20_000, rng=None):
        return schlather_simulate(
            np.asarray(coords, dtype=float),
            n_reps,
            self.range_,
            self.smoothness,
            max_points,
            int(_rng_to_seed(rng)),
        )
