use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray2};

#[pyfunction]
pub fn fmadogram_rust<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    coords: PyReadonlyArray2<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let values = values.as_array();
    let coords = coords.as_array();
    let n = values.nrows();
    let d = values.ncols();

    let u = values.mapv(|z| (-1.0 / z).exp());

    let n_pairs = d * (d - 1) / 2;
    let mut distances = Vec::with_capacity(n_pairs);
    let mut fmadograms = Vec::with_capacity(n_pairs);
    let mut thetas = Vec::with_capacity(n_pairs);

    for i in 0..d {
        for j in (i + 1)..d {
            let nu = 0.5 * u.column(i).iter().zip(u.column(j).iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>() / n as f64;
            let theta = if nu < 0.5 { (1.0 + 2.0 * nu) / (1.0 - 2.0 * nu) } else { 2.0 };
            let dist = ((coords[[i, 0]] - coords[[j, 0]]).powi(2)
                + (coords[[i, 1]] - coords[[j, 1]]).powi(2)).sqrt();

            distances.push(dist);
            fmadograms.push(nu);
            thetas.push(theta);
        }
    }

    Ok((
        PyArray1::from_vec_bound(py, distances),
        PyArray1::from_vec_bound(py, fmadograms),
        PyArray1::from_vec_bound(py, thetas),
    ))
}
