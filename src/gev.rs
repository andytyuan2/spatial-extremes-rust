use ndarray::ArrayView1;
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

fn gev_neg_log_lik_raw(params: [f64; 3], x: ArrayView1<f64>) -> f64 {
    let mu = params[0];
    let sigma = params[1].exp();
    let xi = params[2];
    let z = (&x - mu) / sigma;

    if xi.abs() < 1e-8 {
        let t = (-z).mapv(f64::exp);
        if !t.iter().all(|v| v.is_finite()) {
            return 1e10;
        }
        return (sigma.ln() + &z + &t).sum();
    }

    let support = 1.0 + xi * &z;
    if support.iter().any(|&v| v <= 0.0) {
        return 1e10;
    }

    let t = support.mapv(|v| v.powf(-1.0 / xi));
    let nllh = (sigma.ln() + (1.0 + 1.0 / xi) * support.mapv(f64::ln) + &t).sum();
    if nllh.is_finite() { nllh } else { 1e10 }
}

#[pyfunction]
pub fn gev_neg_log_lik(_py: Python<'_>, params: [f64; 3], x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(gev_neg_log_lik_raw(params, x.as_array()))
}

#[pyfunction]
pub fn gev_numerical_se<'py>(
    py: Python<'py>,
    mu: f64,
    sigma: f64,
    xi: f64,
    x: PyReadonlyArray1<f64>,
    eps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array();
    let p = [mu, sigma, xi];
    let mut hess = [[0.0; 3]; 3];
    let f0 = gev_neg_log_lik_raw(p, x);

    for i in 0..3 {
        for j in 0..3 {
            let mut pp = p; pp[i] += eps; pp[j] += eps;
            let mut pm = p; pm[i] += eps; pm[j] -= eps;
            let mut mp = p; mp[i] -= eps; mp[j] += eps;
            let mut mm = p; mm[i] -= eps; mm[j] -= eps;

            hess[i][j] = (gev_neg_log_lik_raw(pp, x) - gev_neg_log_lik_raw(pm, x)
                - gev_neg_log_lik_raw(mp, x) + gev_neg_log_lik_raw(mm, x)) / (4.0 * eps * eps);
        }
    }

    let hess_mat = nalgebra::Matrix3::from(hess);
    let se = match hess_mat.try_inverse() {
        Some(inv) => {
            let diag = inv.diagonal();
            diag.map(|v| if v > 0.0 { v.sqrt() } else { f64::NAN }).into_owned()
        }
        None => nalgebra::Vector3::new(f64::NAN, f64::NAN, f64::NAN),
    };

    let result = ndarray::Array1::from_vec(vec![se[0], se[1], se[2]]);
    Ok(PyArray1::from_owned_array_bound(py, result))
}
