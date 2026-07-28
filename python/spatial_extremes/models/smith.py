from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_extremes._rust import (
    smith_simulate,
    smith_extremal_coefficient,
    smith_bivariate_cdf,
    smith_bivariate_logpdf,
)

try:
    import geopandas as gpd
except ImportError:
    gpd = None


def _rng_to_seed(rng):
    if rng is None:
        return np.random.default_rng().integers(0, 2**63, dtype=np.uint64)
    return rng.integers(0, 2**63, dtype=np.uint64)


@dataclass
class SmithModel:
    cov: np.ndarray
    buffer_factor: float = 4.0

    n_free_params = 3
    free_param_names = ("log_l11", "l21", "log_l22")

    def __post_init__(self):
        self.cov = np.asarray(self.cov, dtype=float)
        if self.cov.shape != (2, 2):
            raise ValueError("cov must be a 2x2 matrix")
        det = np.linalg.det(self.cov)
        if det <= 0:
            raise ValueError("cov must be positive definite")
        self._rust_params = [
            float(self.cov[0, 0]), float(self.cov[0, 1]),
            float(self.cov[1, 0]), float(self.cov[1, 1]),
        ]

    @classmethod
    def from_free_params(cls, free_params, **kwargs) -> "SmithModel":
        log_l11, l21, log_l22 = free_params
        l11, l22 = np.exp(log_l11), np.exp(log_l22)
        L = np.array([[l11, 0.0], [l21, l22]])
        cov = L @ L.T
        return cls(cov=cov, **kwargs)

    def extremal_coefficient(self, h: np.ndarray) -> float:
        return smith_extremal_coefficient(np.asarray(h, dtype=float).flatten()[:2], self._rust_params)

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        return smith_bivariate_cdf(z1, z2, np.asarray(h, dtype=float).flatten()[:2], self._rust_params)

    def bivariate_logpdf(self, z1, z2, h: np.ndarray):
        return smith_bivariate_logpdf(
            np.asarray(z1, dtype=float),
            np.asarray(z2, dtype=float),
            np.asarray(h, dtype=float).flatten()[:2],
            self._rust_params,
        )

    def simulate(self, coords, n_reps=1, max_points=200_000, rng=None):
        return smith_simulate(
            np.asarray(coords, dtype=float),
            n_reps,
            self._rust_params,
            self.buffer_factor,
            max_points,
            int(_rng_to_seed(rng)),
        )

    def simulate_geo(self, gdf, n_reps=1, value_prefix="sim", rng=None):
        if gpd is None:
            raise ImportError("geopandas is required for simulate_geo()")
        if gdf.crs is not None and gdf.crs.is_geographic:
            raise ValueError("gdf is in a geographic CRS. Reproject to a projected CRS first.")
        coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
        sims = self.simulate(coords, n_reps=n_reps, rng=rng)
        sim_cols = pd.DataFrame(
            sims.T, columns=[f"{value_prefix}_{r}" for r in range(n_reps)], index=gdf.index
        )
        return gpd.GeoDataFrame(pd.concat([gdf, sim_cols], axis=1), crs=gdf.crs)
