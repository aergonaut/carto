//! Iterative inversion of projections without closed-form inverses.

use rayon::prelude::*;

use super::ProjCtx;
use crate::math::nan_max;

/// Inverts `pos` by Newton iteration, following Bildirici 2016
/// (<https://doi.org/10.1080/15230406.2016.1200492>).
///
/// `lon`/`lat` hold the initial guess and are refined in place. Convergence is
/// judged over the `visible` positions only (or all positions if `None`),
/// since some regions outside the visible map never settle.
pub fn invert<P>(
    x: &[f64],
    y: &[f64],
    lon: &mut [f64],
    lat: &mut [f64],
    pos: P,
    visible: Option<&[bool]>,
    ctx: &ProjCtx,
) where
    P: Fn(f64, f64) -> (f64, f64) + Sync,
{
    println!("  (using iterative method, may take a bit)");
    let t = ctx.tolerance;
    let mut steps = vec![(0.0, 0.0); x.len()];
    let mut i = 1;
    loop {
        steps.par_iter_mut().enumerate().for_each(|(k, step)| {
            let (lo, la) = (lon[k], lat[k]);
            let (x1, y1) = pos(lo, la);
            let (xa, ya) = pos(lo, la + t);
            let (xb, yb) = pos(lo, la - t);
            let (xc, yc) = pos(lo + t, la);
            let (xd, yd) = pos(lo - t, la);
            let dxla = (xa - xb) / (2.0 * t);
            let dyla = (ya - yb) / (2.0 * t);
            let dxlo = (xc - xd) / (2.0 * t);
            let dylo = (yc - yd) / (2.0 * t);
            let div = dxla * dylo - dyla * dxlo;
            let difx = x1 - x[k];
            let dify = y1 - y[k];
            *step = if div != 0.0 {
                (
                    (dify * dxla - difx * dyla) / div,
                    (difx * dylo - dify * dxlo) / div,
                )
            } else {
                (0.0, 0.0)
            };
        });
        let max_step = |f: fn(&(f64, f64)) -> f64| {
            steps
                .par_iter()
                .enumerate()
                .map(|(k, s)| {
                    if visible.is_none_or(|v| v[k]) {
                        f(s).abs()
                    } else {
                        0.0
                    }
                })
                .reduce(|| f64::NEG_INFINITY, nan_max)
        };
        if max_step(|s| s.0) < t && max_step(|s| s.1) < t {
            break;
        }
        lon.par_iter_mut()
            .zip(lat.par_iter_mut())
            .zip(steps.par_iter())
            .for_each(|((lo, la), (dlo, dla))| {
                *lo -= dlo;
                *la -= dla;
            });
        i += 1;
        if i > ctx.max_iter {
            println!(
                "  Reached maximum of {} iterations without converging, outputting result",
                ctx.max_iter
            );
            break;
        }
    }
}

/// Solves `f(th, lat) = 0` for an auxiliary angle by Newton iteration, stopping
/// once `err` (maximized over all positions) falls to within tolerance.
pub fn solve_auxiliary<S, E>(lat: &[f64], th: &mut [f64], step: S, err: E, ctx: &ProjCtx)
where
    S: Fn(f64, f64) -> f64 + Sync,
    E: Fn(f64, f64) -> f64 + Sync,
{
    println!("  (using iterative method, may take a bit)");
    let mut i = 1;
    let mut max_err = 1.0;
    while max_err > ctx.tolerance {
        th.par_iter_mut()
            .zip(lat.par_iter())
            .for_each(|(th, &lat)| *th = step(*th, lat));
        max_err = th
            .par_iter()
            .zip(lat.par_iter())
            .map(|(&th, &lat)| err(th, lat).abs())
            .reduce(|| f64::NEG_INFINITY, nan_max);
        i += 1;
        if i > ctx.max_iter {
            println!(
                "  Reached maximum of {} iterations without converging, outputting result",
                ctx.max_iter
            );
            println!("   max remaining error: {}", fmt_py_float(max_err));
            break;
        }
    }
}

/// Formats a float roughly like Python's `str(float)`.
fn fmt_py_float(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else {
        format!("{v:?}")
    }
}
