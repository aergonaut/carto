//! Interpolating pixel values.
//!
//! Backward projection samples the input image (a regular grid) at arbitrary
//! points; forward projection resamples scattered input points onto the
//! output grid. Both follow SciPy's `interpn` and `griddata` semantics.

pub mod scattered;
mod spline;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::options::InterpType;
use crate::raster::{Raster, Sample};

/// An image whose pixel centers lie at `xs` (ascending) and `ys` (descending,
/// top row first).
pub struct GridSource<'a, T> {
    pub xs: &'a [f64],
    pub ys: &'a [f64],
    pub raster: &'a Raster<T>,
}

impl<T: Sample> GridSource<'_, T> {
    /// The value at column `ix` and row `iy`, with rows counted from the
    /// bottom (i.e. in ascending y order).
    fn value(&self, iy: usize, ix: usize, ch: usize) -> T {
        self.raster.pixel(self.raster.height - 1 - iy, ix)[ch]
    }

    /// The y coordinates in ascending order.
    fn ys_ascending(&self) -> Vec<f64> {
        self.ys.iter().rev().copied().collect()
    }
}

/// Finds `i` with `grid[i] <= v < grid[i + 1]` in an ascending grid,
/// extrapolating beyond the ends, and the position of `v` within that
/// interval (`0..1` inside it). Returns `None` for NaN.
///
/// This matches SciPy's `find_interval_ascending` and `find_indices`.
pub(crate) fn find_interval(grid: &[f64], v: f64) -> Option<(usize, f64)> {
    let n = grid.len();
    if v.is_nan() {
        return None;
    }
    let i = if v < grid[0] {
        0
    } else if v >= grid[n - 1] {
        n - 2
    } else {
        grid.partition_point(|&g| g <= v) - 1
    };
    Some((i, (v - grid[i]) / (grid[i + 1] - grid[i])))
}

/// Samples `src` at each point `(px[i], py[i])`, writing pixel `i` of `out`.
pub fn sample_grid<T: Sample>(
    src: &GridSource<T>,
    px: &[f64],
    py: &[f64],
    method: InterpType,
    out: &mut Raster<T>,
) -> Result<()> {
    if src.xs.len() < 2 || src.ys.len() < 2 {
        bail!("input image must be at least 2 pixels in each dimension");
    }
    match method {
        InterpType::Nearest => {
            let ys = src.ys_ascending();
            sample_each(src, &ys, px, py, out, |src, (iy, fy), (ix, fx), ch| {
                let iy = if fy <= 0.5 { iy } else { iy + 1 };
                let ix = if fx <= 0.5 { ix } else { ix + 1 };
                src.value(iy, ix, ch).to_f64()
            });
            Ok(())
        }
        InterpType::Linear => {
            let ys = src.ys_ascending();
            sample_each(src, &ys, px, py, out, |src, (i0, y0), (i1, y1), ch| {
                let v = |a, b| src.value(a, b, ch).to_f64();
                // Accumulated with fused multiply-adds, as SciPy's compiled
                // kernel is, so that truncation to integers matches exactly.
                let r = (v(i0, i1) * (1.0 - y0)) * (1.0 - y1);
                let r = (v(i0, i1 + 1) * (1.0 - y0)).mul_add(y1, r);
                let r = (v(i0 + 1, i1) * y0).mul_add(1.0 - y1, r);
                (v(i0 + 1, i1 + 1) * y0).mul_add(y1, r)
            });
            Ok(())
        }
        InterpType::Slinear => spline::sample(src, px, py, spline::Kind::BSpline(1), out),
        InterpType::Cubic => spline::sample(src, px, py, spline::Kind::BSpline(3), out),
        InterpType::Quintic => spline::sample(src, px, py, spline::Kind::BSpline(5), out),
        InterpType::Pchip => spline::sample(src, px, py, spline::Kind::Pchip, out),
        InterpType::None => {
            bail!("interpolation type \"none\" does not sample through interpolation")
        }
    }
}

/// Evaluates `f` for every point and channel, with NaN points yielding NaN.
fn sample_each<T, F>(
    src: &GridSource<T>,
    ys: &[f64],
    px: &[f64],
    py: &[f64],
    out: &mut Raster<T>,
    f: F,
) where
    T: Sample,
    F: Fn(&GridSource<T>, (usize, f64), (usize, f64), usize) -> f64 + Sync,
{
    let c = out.channels;
    out.data
        .par_chunks_mut(c)
        .enumerate()
        .for_each(|(i, pixel)| {
            let found = find_interval(ys, py[i]).zip(find_interval(src.xs, px[i]));
            for (ch, v) in pixel.iter_mut().enumerate() {
                *v = T::from_f64(match found {
                    Some((y, x)) => f(src, y, x, ch),
                    None => f64::NAN,
                });
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_search_matches_scipy() {
        let g = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(find_interval(&g, 1.0), Some((1, 0.0)));
        assert_eq!(find_interval(&g, 1.5), Some((1, 0.5)));
        assert_eq!(find_interval(&g, 3.0), Some((2, 1.0)));
        assert_eq!(find_interval(&g, -1.0), Some((0, -1.0)));
        assert_eq!(find_interval(&g, 4.0), Some((2, 2.0)));
        assert_eq!(find_interval(&g, f64::NAN), None);
    }
}
