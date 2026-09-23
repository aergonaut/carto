//! Numeric helpers that reproduce NumPy/Python semantics exactly.
//!
//! The original implementation is written against NumPy, and several of its
//! primitives behave differently from their Rust standard library namesakes
//! (e.g. `np.sign(0) == 0` whereas `0.0f64.signum() == 1.0`). Output parity
//! with the original depends on matching these bit-for-bit.

use std::f64::consts::PI;

unsafe extern "C" {
    // Rust's `f64::asinh` is implemented in Rust rather than delegating to the
    // platform libm, so call libm directly to match `np.arcsinh`/`math.asinh`.
    #[link_name = "asinh"]
    fn libm_asinh(x: f64) -> f64;
}

/// Sine, never fused with a cosine of the same argument.
///
/// LLVM combines `x.sin()` and `x.cos()` into a single `sincos` call, whose
/// sine differs from libm's `sin` in the last bit for some inputs on macOS.
/// NumPy always evaluates them separately, so for identical results these
/// must be opaque calls.
#[inline(never)]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine, never fused with a sine of the same argument (see [`sin`]).
#[inline(never)]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// `np.arcsinh` / `math.asinh`.
pub fn asinh(x: f64) -> f64 {
    // SAFETY: `asinh` is a pure function from the C math library.
    unsafe { libm_asinh(x) }
}

/// `np.sign`: returns 0 for ±0 and propagates NaN.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// `np.sinc`: the normalized sinc function `sin(pi x) / (pi x)`.
pub fn sinc(x: f64) -> f64 {
    let y = PI * if x == 0.0 { 1.0e-20 } else { x };
    y.sin() / y
}

/// `np.remainder` / Python's `%` for floats: the result takes the sign of the divisor.
pub fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m == 0.0 {
        0.0f64.copysign(b)
    } else if (b < 0.0) != (m < 0.0) {
        m + b
    } else {
        m
    }
}

/// The larger of two values, or NaN if either is NaN, like `np.amax`. Use
/// with a `-inf` identity to reduce a sequence.
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Pixel-center coordinates spanning `(start, stop)`, as computed by
/// `np.linspace(start, stop, n, endpoint=False) + offset`.
pub fn linspace_open(start: f64, stop: f64, n: usize, offset: f64) -> Vec<f64> {
    let step = (stop - start) / n as f64;
    (0..n).map(|i| (i as f64 * step + start) + offset).collect()
}

/// Horizontal pixel-center coordinates of an image `n` pixels wide, from -1 to 1.
pub fn pixel_centers_x(n: usize) -> Vec<f64> {
    linspace_open(-1.0, 1.0, n, 1.0 / n as f64)
}

/// Vertical pixel-center coordinates of an image `n` pixels tall, from 1 to -1.
pub fn pixel_centers_y(n: usize) -> Vec<f64> {
    linspace_open(1.0, -1.0, n, -1.0 / n as f64)
}

/// `np.arange(start, stop, step)` for floats.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if !(len > 0.0) {
        return Vec::new();
    }
    let delta = (start + step) - start;
    (0..len as usize)
        .map(|i| start + i as f64 * delta)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_matches_numpy() {
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
        assert_eq!(sign(-3.0), -1.0);
        assert!(sign(f64::NAN).is_nan());
    }

    #[test]
    fn py_mod_matches_python() {
        assert_eq!(py_mod(-1.0, 3.0), 2.0);
        assert_eq!(py_mod(1.0, -3.0), -2.0);
        assert_eq!(py_mod(-0.0, 3.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(py_mod(190.0, 360.0), 190.0);
    }

    #[test]
    fn nan_max_propagates() {
        assert!(nan_max(1.0, f64::NAN).is_nan());
        assert!(nan_max(f64::NAN, 1.0).is_nan());
        assert_eq!(nan_max(1.0, 3.0), 3.0);
    }

    #[test]
    fn sinc_at_zero_is_one() {
        assert_eq!(sinc(0.0), 1.0);
    }

    #[test]
    fn arange_counts() {
        assert_eq!(arange(-180.0, 180.0, 30.0).len(), 12);
        assert_eq!(arange(-90.0, 90.0, 30.0)[0], -90.0);
    }
}
