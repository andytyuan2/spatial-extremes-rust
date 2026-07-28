use ndarray::ArrayView2;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray2};
use crate::models::shared::de_haan_bivariate_logpdf;
use crate::models::schlather::schlather_bivariate_logpdf;
use nalgebra::Matrix2;

#[pyfunction]
pub fn pairwise_likelihood_rust(
    _py: Python<'_>,
    values: PyReadonlyArray2<f64>,
    coords: PyReadonlyArray2<f64>,
    pairs: PyReadonlyArray2<i64>,
    model_type: &str,
    params: Vec<f64>,
) -> PyResult<f64> {
    let values = values.as_array();
    let coords = coords.as_array();
    let pairs = pairs.as_array();

    let nllh = match model_type {
        "SmithModel" | "smith" => {
            if params.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err("SmithModel needs 4 covariance params"));
            }
            let cov = Matrix2::new(params[0], params[1], params[2], params[3]);
            let inv_cov = cov.try_inverse().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cov not invertible"))?;
            let mut total = 0.0;
            for pair in pairs.outer_iter() {
                let i = pair[0] as usize;
                let j = pair[1] as usize;
                let h = nalgebra::Vector2::new(
                    coords[[i, 0]] - coords[[j, 0]],
                    coords[[i, 1]] - coords[[j, 1]],
                );
                let a = (h.transpose() * inv_cov * h)[(0, 0)].sqrt();
                let logpdf = de_haan_bivariate_logpdf(values.column(i), values.column(j), a);
                total += logpdf.sum();
            }
            -total
        }
        "SchlatherModel" | "schlather" => {
            if params.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err("SchlatherModel needs 2 params"));
            }
            let range = params[0];
            let smoothness = params[1];
            let mut total = 0.0;
            for pair in pairs.outer_iter() {
                let i = pair[0] as usize;
                let j = pair[1] as usize;
                let dx = coords[[i, 0]] - coords[[j, 0]];
                let dy = coords[[i, 1]] - coords[[j, 1]];
                let dist = (dx * dx + dy * dy).sqrt();
                let rho = (-((dist / range).powf(smoothness))).exp().clamp(-0.999999, 0.999999);
                let logpdf = schlather_bivariate_logpdf(values.column(i), values.column(j), rho);
                total += logpdf.sum();
            }
            -total
        }
        "BrownResnickModel" | "brownresnick" => {
            if params.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err("BrownResnickModel needs 2 params"));
            }
            let range = params[0];
            let alpha = params[1];
            let mut total = 0.0;
            for pair in pairs.outer_iter() {
                let i = pair[0] as usize;
                let j = pair[1] as usize;
                let dx = coords[[i, 0]] - coords[[j, 0]];
                let dy = coords[[i, 1]] - coords[[j, 1]];
                let dist = (dx * dx + dy * dy).sqrt();
                let gamma = (dist / range).powf(alpha);
                let a = (2.0 * gamma).sqrt();
                let logpdf = de_haan_bivariate_logpdf(values.column(i), values.column(j), a);
                total += logpdf.sum();
            }
            -total
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err("unknown model_type")),
    };

    Ok(nllh)
}
