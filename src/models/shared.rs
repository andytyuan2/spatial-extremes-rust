use ndarray::{Array1, ArrayView1};
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use crate::utils::{normal_cdf, normal_pdf};

/// De Haan bivariate CDF shared by Smith and Brown–Resnick.
pub fn de_haan_bivariate_cdf(z1: f64, z2: f64, a: f64) -> f64 {
    if a <= 0.0 {
        return (-1.0 / z1.min(z2)).exp();
    }
    let w = a / 2.0 + (z2 / z1).ln() / a;
    let v = a - w;
    let v_val = normal_cdf(w) / z1 + normal_cdf(v) / z2;
    (-v_val).exp()
}

/// De Haan extremal coefficient: θ(a) = 2·Φ(a/2)
pub fn de_haan_extremal_coefficient(a: f64) -> f64 {
    2.0 * normal_cdf(a / 2.0)
}

/// Vectorised bivariate log-density on unit-Fréchet margins.
pub fn de_haan_bivariate_logpdf(z1: ArrayView1<f64>, z2: ArrayView1<f64>, a: f64) -> Array1<f64> {
    let a = a.max(1e-8);
    let ratio = &z2 / &z1;
    let w = a / 2.0 + ratio.mapv(f64::ln) / a;
    let v = a - &w;

    let phi_w = w.mapv(normal_cdf);
    let phi_v = v.mapv(normal_cdf);
    let pdf_w = w.mapv(normal_pdf);

    let v_sum = &phi_w / &z1 + &phi_v / &z2;
    let v1 = -&phi_w / (&z1 * &z1);
    let v2 = -&phi_v / (&z2 * &z2);
    let v12 = -&pdf_w / (a * &z1 * &z1 * &z2);

    let density_factor = &v1 * &v2 - &v12;
    let density_factor = density_factor.mapv(|x| if x > 0.0 { x } else { 1e-300 });

    -&v_sum + density_factor.mapv(f64::ln)
}

// PyO3 wrappers
#[pyfunction]
pub fn de_haan_bivariate_cdf_py(_py: Python<'_>, z1: f64, z2: f64, a: f64) -> PyResult<f64> {
    Ok(de_haan_bivariate_cdf(z1, z2, a))
}

#[pyfunction]
pub fn de_haan_extremal_coefficient_py(_py: Python<'_>, a: f64) -> PyResult<f64> {
    Ok(de_haan_extremal_coefficient(a))
}

#[pyfunction]
pub fn de_haan_bivariate_logpdf<'py>(
    py: Python<'py>,
    z1: PyReadonlyArray1<f64>,
    z2: PyReadonlyArray1<f64>,
    a: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let res = de_haan_bivariate_logpdf(z1.as_array(), z2.as_array(), a);
    Ok(PyArray1::from_owned_array_bound(py, res))
}
