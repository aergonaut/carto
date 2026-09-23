//! Drawing graticules (lines of constant longitude and latitude).

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::math::{py_mod, sign};

/// Finds the pixels along each longitude and latitude line (in degrees),
/// keeping lines continuous by marking the pixel on whichever side of the
/// line is closer wherever the line passes between pixels.
///
/// Like the original, neighbors wrap around the image edges and lines are
/// widened in a plus shape rather than a square.
pub fn mask(
    lon: &[f64],
    lat: &[f64],
    rows: usize,
    cols: usize,
    lons: &[f64],
    lats: &[f64],
    width: u32,
) -> Vec<bool> {
    let lon1: Vec<f64> = lon
        .par_iter()
        .map(|l| py_mod(l + PI, 2.0 * PI) - PI)
        .collect();
    let lat1: Vec<f64> = lat
        .par_iter()
        .map(|l| py_mod(l + PI / 2.0, PI) - PI / 2.0)
        .collect();
    let mut grats = vec![false; rows * cols];
    for (lines, coord) in [(lons, &lon1), (lats, &lat1)] {
        for g in lines {
            let g = g.to_radians();
            let diff: Vec<f64> = if g == -PI {
                // The antimeridian: measure from the western side.
                coord
                    .par_iter()
                    .map(|&l| if l > 0.0 { l - 2.0 * PI } else { l } - g)
                    .collect()
            } else {
                coord.par_iter().map(|&l| l - g).collect()
            };
            mark_crossings(&diff, rows, cols, &mut grats);
        }
    }
    if width > 1 {
        widen(&grats, rows, cols, width as usize)
    } else {
        grats
    }
}

/// Marks pixels where `diff` is zero or changes sign between neighbors.
fn mark_crossings(diff: &[f64], rows: usize, cols: usize, grats: &mut [bool]) {
    let n = rows * cols;
    for axis in 0..2 {
        // Index of the previous (rolled by +1) and next neighbor along the axis.
        let prev = |i: usize| match axis {
            0 => (i + n - cols) % n,
            _ => i - i % cols + (i % cols + cols - 1) % cols,
        };
        let next = |i: usize| match axis {
            0 => (i + cols) % n,
            _ => i - i % cols + (i + 1) % cols,
        };
        // A crossing between a pixel and its previous neighbor, where the line
        // lies closer to the previous neighbor.
        let crossing = |i: usize| -> (bool, bool) {
            let j = prev(i);
            let mut over = sign(diff[i]) + sign(diff[j]);
            // Avoid spurious crossings where longitude wraps around.
            if (diff[i] - diff[j]).abs() > PI / 2.0 {
                over = 2.0;
            }
            let crossed = over == 0.0;
            let prev_closer = crossed && diff[i].abs() > diff[j].abs();
            (crossed && !prev_closer, prev_closer)
        };
        grats.par_iter_mut().enumerate().for_each(|(i, g)| {
            *g |= sign(diff[i]) == 0.0 || crossing(i).0 || crossing(next(i)).1;
        });
    }
}

fn widen(grats: &[bool], rows: usize, cols: usize, width: usize) -> Vec<bool> {
    let mut out = grats.to_vec();
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let (r, c) = (i / cols, i % cols);
        for w in 1..width {
            *o |= grats[((r + rows - w % rows) % rows) * cols + c]
                || grats[((r + w) % rows) * cols + c]
                || grats[r * cols + (c + cols - w % cols) % cols]
                || grats[r * cols + (c + w) % cols];
        }
    });
    out
}
