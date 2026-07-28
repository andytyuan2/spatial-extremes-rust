use ndarray::ArrayView2;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use nalgebra::DMatrix;

fn powered_exponential(dist: f64, range: f64, smoothness: f64) -> f64 {
    (-((dist / range).powf(smoothness))).exp()
}

fn schlather_bivariate_logpdf(z1: ndarray::ArrayView1<f64>, z2: ndarray::ArrayView1<f64>, rho: f64) -> ndarray::Array1<f64> {
    let rho = rho.clamp(-0.999999, 0.999999);
    let b = (&z1 * &z1 + &z2 * &z2 - 2.0 * rho * &z1 * &z2).mapv(|x| x.max(1e-10).sqrt());

    let v1 = (rho * &z1 - &z2 - &b) / (2.0 * &z1 * &z1 * &b);
    let v2 = (rho * &z2 - &z1 - &b) / (2.0 * &z2 * &z2 * &b);
    let v12 = (rho * rho - 1.0) / (2.0 * &b * &b * &b);

    let s = 1.0 - 2.0 * (rho + 1.0) * &z1 * &z2 / (&z1 + &z2).mapv(|x| x * x);
    let s = s.mapv(|x| x.max(1e-12));
    let v = 0.5 * (1.0 / &z1 + 1.0 / &z2) * (1.0 + s.mapv(|x| x.sqrt()));

    let density_factor = &v1 * &v2 - &v12;
    let density_factor = density_factor.mapv(|x| if x > 0.0 { x } else { 1e-300 });

    -&v + density_factor.mapv(f64::ln)
}

#[pyfunction]
pub fn schlather_extremal_coefficient(_py: Python<'_>, h: [f64; 2], range: f64, smoothness: f64) -> PyResult<f64> {
    let dist = (h[0] * h[0] + h[1] * h[1]).sqrt();
    let rho = powered_exponential(dist, range, smoothness).clamp(-0.999999, 0.999999);
    Ok(1.0 + ((1.0 - rho) / 2.0).sqrt())
}

#[pyfunction]
pub fn schlather_bivariate_cdf(_py: Python<'_>, z1: f64, z2: f64, h: [f64; 2], range: f64, smoothness: f64) -> PyResult<f64> {
    let dist = (h[0] * h[0] + h[1] * h[1]).sqrt();
    let rho = powered_exponential(dist, range, smoothness).clamp(-0.999999, 0.999999);
    let s = 1.0 - 2.0 * (rho + 1.0) * z1 * z2 / (z1 + z2).powi(2);
    let s = s.max(0.0);
    let v = 0.5 * (1.0 / z1 + 1.0 / z2) * (1.0 + s.sqrt());
    Ok((-v).exp())
}

#[pyfunction]
pub fn schlather_bivariate_logpdf<'py>(
    py: Python<'py>,
    z1: PyReadonlyArray1<f64>,
    z2: PyReadonlyArray1<f64>,
    h: [f64; 2],
    range: f64,
    smoothness: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dist = (h[0] * h[0] + h[1] * h[1]).sqrt();
    let rho = powered_exponential(dist, range, smoothness).clamp(-0.999999, 0.999999);
    let res = schlather_bivariate_logpdf(z1.as_array(), z2.as_array(), rho);
    Ok(PyArray1::from_owned_array_bound(py, res))
}

#[pyfunction]
pub fn schlather_simulate<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    n_reps: usize,
    range: f64,
    smoothness: f64,
    max_points: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    py.allow_threads(|| {
        let coords = coords.as_array();
        let d = coords.nrows();

        let mut r = ndarray::Array2::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let dx = coords[[i, 0]] - coords[[j, 0]];
                let dy = coords[[i, 1]] - coords[[j, 1]];
                let dist = (dx * dx + dy * dy).sqrt();
                r[[i, j]] = powered_exponential(dist, range, smoothness);
            }
        }
        for i in 0..d {
            r[[i, i]] += 1e-10;
        }

        let r_mat = DMatrix::from_row_slice(d, d, r.as_slice().unwrap());
        let chol = r_mat.cholesky().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("correlation matrix not positive definite"))?;

        let w_bound = (2.0 * std::f64::consts::PI).sqrt() * 10.0;
        let mut out = ndarray::Array2::zeros((n_reps, d));
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let exp_dist = rand_distr::Exp::new(1.0).unwrap();

        for r_idx in 0..n_reps {
            let mut running_max = ndarray::Array1::zeros(d);
            let mut t_cum = 0.0;
            let mut n_used = 0;

            while n_used < max_points {
                t_cum += exp_dist.sample(&mut rng);
                let xi = 1.0 / t_cum;
                n_used += 1;

                if xi * w_bound <= running_max.iter().cloned().fold(f64::INFINITY, f64::min) && n_used > d {
                    break;
                }

                let z_std: Vec<f64> = (0..d).map(|_| rng.sample(rand_distr::StandardNormal)).collect();
                let z_vec = DMatrix::from_column_slice(d, 1, &z_std);
                let z = chol.l() * z_vec;
                let w: Vec<f64> = z.iter().map(|x| (2.0 * std::f64::consts::PI).sqrt() * x.max(0.0)).collect();

                for i in 0..d {
                    let contrib = xi * w[i];
                    if contrib > running_max[i] {
                        running_max[i] = contrib;
                    }
                }
            }
            out.row_mut(r_idx).assign(&running_max);
        }

        Ok(PyArray2::from_owned_array_bound(py, out))
    })
}
