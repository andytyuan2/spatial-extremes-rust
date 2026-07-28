use pyo3::prelude::*;

mod utils;
mod gev;
mod models;
mod dependence;
mod fitting;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // GEV
    m.add_function(wrap_pyfunction!(gev::gev_neg_log_lik, m)?)?;
    m.add_function(wrap_pyfunction!(gev::gev_numerical_se, m)?)?;

    // Models – shared de Haan family
    m.add_function(wrap_pyfunction!(models::shared::de_haan_bivariate_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(models::shared::de_haan_bivariate_logpdf, m)?)?;
    m.add_function(wrap_pyfunction!(models::shared::de_haan_extremal_coefficient, m)?)?;

    // Models – Smith
    m.add_function(wrap_pyfunction!(models::smith::smith_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(models::smith::smith_extremal_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(models::smith::smith_bivariate_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(models::smith::smith_bivariate_logpdf, m)?)?;

    // Models – Schlather
    m.add_function(wrap_pyfunction!(models::schlather::schlather_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(models::schlather::schlather_extremal_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(models::schlather::schlather_bivariate_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(models::schlather::schlather_bivariate_logpdf, m)?)?;

    // Models – Brown–Resnick
    m.add_function(wrap_pyfunction!(models::brownresnick::brownresnick_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(models::brownresnick::brownresnick_extremal_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(models::brownresnick::brownresnick_bivariate_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(models::brownresnick::brownresnick_bivariate_logpdf, m)?)?;

    // Dependence diagnostics
    m.add_function(wrap_pyfunction!(dependence::fmadogram_rust, m)?)?;

    // Pairwise likelihood
    m.add_function(wrap_pyfunction!(fitting::pairwise_likelihood_rust, m)?)?;

    Ok(())
}
