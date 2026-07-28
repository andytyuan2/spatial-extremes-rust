from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import optimize

from spatial_extremes._rust import pairwise_likelihood_rust

try:
    import polars as pl
except ImportError:
    pl = None


def _to_numpy(df):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_numpy()
    return df.to_numpy()


@dataclass
class PairwiseFitResult:
    model: object
    free_params: np.ndarray
    nllh: float
    converged: bool
    n_pairs_used: int
    n_replicates: int
    optimizer_result: object


def fit_pairwise_likelihood(
    model_class,
    frechet_df,
    coords: np.ndarray,
    x0_free: np.ndarray | None = None,
    max_pairs: int | None = None,
    method: str = "Nelder-Mead",
    rng: np.random.Generator | None = None,
) -> PairwiseFitResult:
    values = _to_numpy(frechet_df)
    n_obs, D = values.shape
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] != D:
        raise ValueError(f"coords has {coords.shape[0]} rows but frechet_df has {D} columns")

    all_pairs = list(combinations(range(D), 2))
    if max_pairs is not None and max_pairs < len(all_pairs):
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    pairs_arr = np.array(pairs, dtype=np.int64)

    if x0_free is None:
        x0_free = np.zeros(model_class.n_free_params)

    def neg_composite_llh(free_params):
        try:
            model = model_class.from_free_params(free_params)
        except (ValueError, np.linalg.LinAlgError):
            return 1e12
        try:
            params = model._rust_params
            return pairwise_likelihood_rust(
                values, coords, pairs_arr, model_class.__name__, params
            )
        except Exception:
            return 1e12

    result = optimize.minimize(
        neg_composite_llh,
        x0=np.asarray(x0_free, dtype=float),
        method=method,
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6}
        if method == "Nelder-Mead" else {"maxiter": 5000},
    )

    fitted_model = model_class.from_free_params(result.x)
    return PairwiseFitResult(
        model=fitted_model, free_params=result.x, nllh=result.fun,
        converged=bool(result.success), n_pairs_used=len(pairs),
        n_replicates=n_obs, optimizer_result=result,
    )
