//! Cylindrical projections.

use std::f64::consts::{PI, SQRT_2};

use super::{ProjCtx, map2};
use crate::math::{asinh, cos, sign, sin};

pub fn equi_coords(x: f64, y: f64) -> (f64, f64) {
    (PI * x, PI * y / 2.0)
}

pub fn equi_pos(lon: f64, lat: f64) -> (f64, f64) {
    (lon / PI, 2.0 * lat / PI)
}

/// Default Mercator truncation latitude, in degrees, at which the map is square.
pub fn merc_default_ref() -> f64 {
    (2.0 * PI.exp().atan() - PI / 2.0).to_degrees()
}

/// `ln(tan(pi/4 + ref/2))`: the Mercator y-extent at the truncation latitude.
fn merc_extent(ctx: &ProjCtx) -> f64 {
    let r = ctx.reference_rad(&[merc_default_ref()])[0];
    (PI / 4.0 + r / 2.0).tan().ln()
}

pub fn merc_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let extent = merc_extent(ctx);
    map2(x, y, |x, y| {
        let y1 = y * extent / PI;
        (
            PI * x,
            sign(y1) * (2.0 * (y1.abs() * PI).exp().atan() - PI / 2.0),
        )
    })
}

pub fn merc_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let extent = merc_extent(ctx);
    map2(lon, lat, |lon, lat| {
        let y = sign(lat) * (PI / 4.0 + lat.abs() / 2.0).tan().ln() / PI;
        (lon / PI, y / extent * PI)
    })
}

pub fn merc_vis(x: f64, y: f64) -> bool {
    // The original computes (y * x) / x, which is NaN (and thus visible) at x == 0.
    !(((y * x) / x).abs() > 1.0)
}

pub fn merc_ratio(ctx: &ProjCtx) -> f64 {
    PI / merc_extent(ctx)
}

pub const GALL_RATIO: f64 = (PI / SQRT_2) / (1.0 + SQRT_2 / 2.0);

pub fn gall_coords(x: f64, y: f64) -> (f64, f64) {
    (x * PI, 2.0 * y.atan())
}

pub fn gall_pos(lon: f64, lat: f64) -> (f64, f64) {
    (lon / PI, (lat / 2.0).tan())
}

fn miller_extent() -> f64 {
    asinh((PI * 2.0 / 5.0).tan())
}

pub fn miller_ratio() -> f64 {
    PI / (5.0 / 4.0 * miller_extent())
}

pub fn miller_coords(x: f64, y: f64) -> (f64, f64) {
    let y1 = y * miller_extent();
    (x * PI, 5.0 / 4.0 * y1.sinh().atan())
}

pub fn miller_pos(lon: f64, lat: f64) -> (f64, f64) {
    let y1 = asinh((lat * 4.0 / 5.0).tan());
    (lon / PI, y1 / miller_extent())
}

pub fn cyleq_coords(x: f64, y: f64) -> (f64, f64) {
    (x * PI, y.asin())
}

pub fn cyleq_pos(lon: f64, lat: f64) -> (f64, f64) {
    (lon / PI, sin(lat))
}

pub fn cyleq_ratio(ctx: &ProjCtx) -> f64 {
    let r = ctx.reference_rad(&[37.5])[0];
    PI * cos(r).powf(2.0)
}
