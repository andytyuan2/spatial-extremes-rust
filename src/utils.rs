use std::f64::consts::{PI, SQRT_2};

/// Standard normal CDF: Φ(x)
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / SQRT_2))
}

/// Standard normal PDF: φ(x)
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}
