use ndarray::ArrayView2;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use nalgebra::DMatrix;
use crate::models::shared::{de_haan_bivariate_cdf, de_haan_bivariate_logpdf, de_haan_extremal_coefficient};

fn power_variogram(dist: f64, range: f64, alpha: f64) -> f64 {
    (dist / range).powf(alpha)
}

#[pyfunction]
pub fn brownresnick_extremal_coefficient(_py: Python<'_>, h: [f64; 2], range: f64, alpha: f64) -> PyResult<f64> {
    let dist = (h[0] * h[0] + h[1] * h[1]).sqrt();
    let gamma = power_variogram(dist, range, alpha);
    let a = (2.0 * gamma).sqrt();
    Ok(de_haan_extremal_coefficient(a))
}

#[pyfunction]
pub fn brownresnick_bivariate_cdf(_py: Python<'_>, z1: f64, z2: f64, h: [f64; 2], range: f64, alpha: f64) -> PyResult<f64> {
    let dist = (h[0] * h[0] + h[1] * h[1]).sqrt();
    let gamma = power_variogram(dist, range, alpha);
    let a = (2.0 * gamma).sqrt();
    Ok(de_haan_bivariate_cdf(z1, z2, a))
}

#[pyfunction]
pub fn brownresnick_bivariate_logpdf<'py>(
    py: Python<'py>,
    z1: PyReadonlyArray1<f64>,
    z2: PyReadonlyArray1<f64>,
    h: [f64; 2],
    range: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dist = (h[0] * h[0] + h[1] * h[1]).sqrt();
    let gamma = power_variogram(dist, range, alpha);
    let a = (2.0 * gamma).sqrt();
    let res = de_haan_bivariate_logpdf(z1.as_array(), z2.as_array(), a);
    Ok(PyArray1::from_owned_array_bound(py, res))
}

#[pyfunction]
pub fn brownresnick_simulate<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    n_reps: usize,
    range: f64,
    alpha: f64,
    max_points: usize,
    batch_size: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    py.allow_threads(|| {
        let coords = coords.as_array();
        let d = coords.nrows();

        let origin = ndarray::Array1::from_vec(vec![
            coords.column(0).sum() / d as f64,
            coords.column(1).sum() / d as f64,
        ]);

        let gamma_to_origin: Vec<f64> = (0..d)
            .map(|i| {
                let dx = coords[[i, 0]] - origin[0];
                let dy = coords[[i, 1]] - origin[1];
                power_variogram((dx * dx + dy * dy).sqrt(), range, alpha)
            })
            .collect();

        let mut gamma_pairwise = ndarray::Array2::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let dx = coords[[i, 0]] - coords[[j, 0]];
                let dy = coords[[i, 1]] - coords[[j, 1]];
                gamma_pairwise[[i, j]] = power_variogram((dx * dx + dy * dy).sqrt(), range, alpha);
            }
        }

        let mut cov = ndarray::Array2::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                cov[[i, j]] = gamma_to_origin[i] + gamma_to_origin[j] - gamma_pairwise[[i, j]];
            }
        }
        for i in 0..d {
            cov[[i, i]] += 1e-8;
        }

        let cov_mat = DMatrix::from_row_slice(d, d, cov.as_slice().unwrap());
        let chol = cov_mat.cholesky().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("covariance not positive definite"))?;

        let var: Vec<f64> = (0..d).map(|i| cov[[i, i]].max(0.0)).collect();
        let w_bound: Vec<f64> = var.iter()
            .enumerate()
            .map(|(i, v)| (6.0 * v.max(0.0).sqrt() - gamma_to_origin[i]).exp())
            .collect();

        let mut out = ndarray::Array2::zeros((n_reps, d));
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let exp_dist = rand_distr::Exp::new(1.0).unwrap();

        for r_idx in 0..n_reps {
            let mut running_max = ndarray::Array1::zeros(d);
            let mut t_cum = 0.0;
            let mut n_used = 0;

            while n_used < max_points {
                let b = batch_size.min(max_points - n_used);
                let mut cum = Vec::with_capacity(b);
                for _ in 0..b {
                    t_cum += exp_dist.sample(&mut rng);
                    cum.push(t_cum);
                }
                n_used += b;

                let xi: Vec<f64> = cum.iter().map(|t| 1.0 / t).collect();

                let eps_mat = {
                    let mut flat = Vec::with_capacity(d * b);
                    for _ in 0..d {
                        for _ in 0..b {
                            flat.push(rng.sample(rand_distr::StandardNormal));
                        }
                    }
                    DMatrix::from_row_slice(d, b, &flat)
                };
                let eps = chol.l() * eps_mat;

                for batch_idx in 0..b {
                    for i in 0..d {
                        let w = (eps[(i, batch_idx)] - gamma_to_origin[i]).exp();
                        let contrib = xi[batch_idx] * w;
                        if contrib > running_max[i] {
                            running_max[i] = contrib;
                        }
                    }
                }

                if w_bound.iter().enumerate().all(|(i, wb)| xi.last().unwrap() * wb <= running_max[i]) {
                    break;
                }
            }
            out.row_mut(r_idx).assign(&running_max);
        }

        Ok(PyArray2::from_owned_array_bound(py, out))
    })
}
