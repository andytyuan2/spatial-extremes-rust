from __future__ import annotations

import numpy as np
import pandas as pd

from spatial_extremes._rust import fmadogram_rust

try:
    import polars as pl
except ImportError:
    pl = None


def _to_numpy(df):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_numpy(), df.columns
    return df.to_numpy(), list(df.columns)


def fmadogram(frechet_df, coords: np.ndarray) -> pd.DataFrame:
    values, cols = _to_numpy(frechet_df)
    coords = np.asarray(coords, dtype=float)
    n, D = values.shape
    if coords.shape[0] != D:
        raise ValueError(f"coords has {coords.shape[0]} rows but frechet_df has {D} columns")

    distances, fmadograms, thetas = fmadogram_rust(values, coords)

    rows = []
    idx = 0
    for i in range(D):
        for j in range(i + 1, D):
            rows.append({
                "site_i": cols[i], "site_j": cols[j],
                "distance": distances[idx],
                "fmadogram": fmadograms[idx],
                "extremal_coefficient": thetas[idx],
            })
            idx += 1
    return pd.DataFrame(rows)


def empirical_extremal_coefficient(frechet_df, coords: np.ndarray) -> pd.DataFrame:
    result = fmadogram(frechet_df, coords)
    return result[["site_i", "site_j", "distance", "extremal_coefficient"]]


def binned_extremal_coefficient(frechet_df, coords: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    ec = empirical_extremal_coefficient(frechet_df, coords)
    ec["dist_bin"] = pd.cut(ec["distance"], bins=n_bins)
    summary = (
        ec.groupby("dist_bin", observed=True)
        .agg(mean_distance=("distance", "mean"),
             mean_extremal_coefficient=("extremal_coefficient", "mean"),
             n_pairs=("extremal_coefficient", "size"))
        .reset_index(drop=True)
    )
    return summary
