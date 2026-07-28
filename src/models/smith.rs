use ndarray::ArrayView2;
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use crate::models::shared::{de_haan_bivariate_cdf, de_haan_bivariate_logpdf, de_haan_extremal_coefficient};
use nalgebra::Matrix2;

#[pyfunction]
pub fn smith_extremal_coefficient(_py: Python<'_>, h: [f64; 2], cov_flat: [f64; 4]) -> PyResult<f64> {
    let cov = Matrix2::new(cov_flat[0], cov_flat[1], cov_flat[2], cov_flat[3]);
    let inv_cov = cov.try_inverse().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cov not invertible"))?;
    let h_vec = nalgebra::Vector2::new(h[0], h[1]);
    let a = (h_vec.transpose() * inv_cov * h_vec)[(0, 0)].sqrt();
    Ok(de_haan_extremal_coefficient(a))
}

#[pyfunction]
pub fn smith_bivariate_cdf(_py: Python<'_>, z1: f64, z2: f64, h: [f64; 2], cov_flat: [f64; 4]) -> PyResult<f64> {
    let cov = Matrix2::new(cov_flat[0], cov_flat[1], cov_flat[2], cov_flat[3]);
    let inv_cov = cov.try_inverse().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cov not invertible"))?;
    let h_vec = nalgebra::Vector2::new(h[0], h[1]);
    let a = (h_vec.transpose() * inv_cov * h_vec)[(0, 0)].sqrt();
    Ok(de_haan_bivariate_cdf(z1, z2, a))
}

#[pyfunction]
pub fn smith_bivariate_logpdf<'py>(
    py: Python<'py>,
    z1: PyReadonlyArray1<f64>,
    z2: PyReadonlyArray1<f64>,
    h: [f64; 2],
    cov_flat: [f64; 4],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cov = Matrix2::new(cov_flat[0], cov_flat[1], cov_flat[2], cov_flat[3]);
    let inv_cov = cov.try_inverse().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cov not invertible"))?;
    let h_vec = nalgebra::Vector2::new(h[0], h[1]);
    let a = (h_vec.transpose() * inv_cov * h_vec)[(0, 0)].sqrt();
    let res = de_haan_bivariate_logpdf(z1.as_array(), z2.as_array(), a);
    Ok(PyArray1::from_owned_array_bound(py, res))
}

#[pyfunction]
pub fn smith_simulate<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    n_reps: usize,
    cov_flat: [f64; 4],
    buffer_factor: f64,
    max_points: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    py.allow_threads(|| {
        let coords = coords.as_array();
        let cov = Matrix2::new(cov_flat[0], cov_flat[1], cov_flat[2], cov_flat[3]);
        let inv_cov = cov.try_inverse().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cov not invertible"))?;
        let det_cov = cov.determinant();
        let norm_const = 1.0 / (2.0 * std::f64::consts::PI * det_cov.sqrt());
        let kernel_max = norm_const;

        let eigvals = cov.symmetric_eigenvalues();
        let max_eigval = eigvals.max();
        let buffer = buffer_factor * max_eigval.sqrt();

        let d = coords.nrows();
        let mut lo = [f64::INFINITY, f64::INFINITY];
        let mut hi = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        for i in 0..d {
            lo[0] = lo[0].min(coords[[i, 0]]);
            lo[1] = lo[1].min(coords[[i, 1]]);
            hi[0] = hi[0].max(coords[[i, 0]]);
            hi[1] = hi[1].max(coords[[i, 1]]);
        }
        lo[0] -= buffer; lo[1] -= buffer;
        hi[0] += buffer; hi[1] += buffer;
        let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);

        let mut out = ndarray::Array2::zeros((n_reps, d));
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let exp_dist = rand_distr::Exp::new(1.0).unwrap();

        for r in 0..n_reps {
            let mut running_max = ndarray::Array1::zeros(d);
            let mut t_cum = 0.0;
            let mut n_used = 0;

            while n_used < max_points {
                t_cum += exp_dist.sample(&mut rng);
                let xi = area / t_cum;
                n_used += 1;

                if xi * kernel_max <= running_max.iter().cloned().fold(f64::INFINITY, f64::min) && n_used > d {
                    break;
                }

                let cx = lo[0] + rng.gen::<f64>() * (hi[0] - lo[0]);
                let cy = lo[1] + rng.gen::<f64>() * (hi[1] - lo[1]);

                for i in 0..d {
                    let dx = coords[[i, 0]] - cx;
                    let dy = coords[[i, 1]] - cy;
                    let delta = nalgebra::Vector2::new(dx, dy);
                    let quad = (delta.transpose() * inv_cov * delta)[(0, 0)];
                    let kernel_val = norm_const * (-0.5 * quad).exp();
                    let contrib = xi * kernel_val;
                    if contrib > running_max[i] {
                        running_max[i] = contrib;
                    }
                }
            }
            out.row_mut(r).assign(&running_max);
        }

        Ok(PyArray2::from_owned_array_bound(py, out))
    })
}
