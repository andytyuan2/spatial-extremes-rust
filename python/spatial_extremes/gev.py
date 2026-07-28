from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from spatial_extremes._rust import gev_neg_log_lik, gev_numerical_se

try:
    import polars as pl
except ImportError:
    pl = None


@dataclass
class GEVResult:
    loc: float
    scale: float
    shape: float
    loc_se: float
    scale_se: float
    shape_se: float
    nllh: float
    converged: bool
    n_obs: int

    def as_dict(self) -> dict:
        return {
            "loc": self.loc, "scale": self.scale, "shape": self.shape,
            "loc_se": self.loc_se, "scale_se": self.scale_se, "shape_se": self.shape_se,
            "nllh": self.nllh, "converged": self.converged, "n_obs": self.n_obs,
        }

    def cdf(self, x):
        return stats.genextreme.cdf(x, c=-self.shape, loc=self.loc, scale=self.scale)

    def to_unit_frechet(self, x):
        u = self.cdf(np.asarray(x, dtype=float))
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return -1.0 / np.log(u)


def fit_gev(x, x0: tuple[float, float, float] | None = None) -> GEVResult:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        raise ValueError(f"Need at least ~10 observations for a stable GEV fit, got {n}")

    if x0 is None:
        loc0 = np.mean(x) - 0.5772 * np.std(x) * np.sqrt(6) / np.pi
        scale0 = np.std(x) * np.sqrt(6) / np.pi
        shape0 = 0.0
        x0 = (loc0, np.log(max(scale0, 1e-3)), shape0)
    else:
        x0 = (x0[0], np.log(max(x0[1], 1e-6)), x0[2])

    result = optimize.minimize(
        lambda p: gev_neg_log_lik(p, x),
        x0=np.array(x0),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )

    mu, log_sigma, xi = result.x
    sigma = np.exp(log_sigma)

    se = gev_numerical_se(mu, sigma, xi, x, 1e-4)

    return GEVResult(
        loc=mu, scale=sigma, shape=xi,
        loc_se=se[0], scale_se=se[1], shape_se=se[2],
        nllh=result.fun, converged=bool(result.success), n_obs=n,
    )


def fit_gev_by_site(df, site_cols: list[str] | None = None) -> pd.DataFrame:
    if pl is not None and isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df

    cols = site_cols if site_cols is not None else list(pdf.columns)
    rows = []
    for col in cols:
        res = fit_gev(pdf[col].to_numpy())
        row = {"site": col, **res.as_dict()}
        rows.append(row)
    return pd.DataFrame(rows)
