//! Miscellaneous and special-use projections.

use std::f64::consts::{PI, SQRT_2};

use super::azimuthal::{
    bihem_lon_out, bihem_split, layout_coords, layout_pos, layout_vis, ortho_coords1, ortho_pos1,
    stereo_coords1, stereo_pos1,
};
use super::iterative::invert;
use super::pseudo::hammer_coords;
use super::{AzimType, ProjCtx, circ_vis, map2, mask2, pill_vis};
use crate::math::{cos, sign, sin};

/// Nicolosi globular forward projection for one hemisphere, spanning `[-1, 1]`
/// (or a wider range for the experimental global layout).
fn nicolosi_pos1(lon: f64, lat: f64, global: bool) -> (f64, f64) {
    let lat0 = lat == 0.0;
    let lon0 = lon == 0.0;
    let latmax = lat.abs() == PI / 2.0;
    let lonmax = lon.abs() == PI / 2.0;
    let b = PI / (2.0 * lon) - 2.0 * lon / PI;
    let c = 2.0 * lat / PI;
    let sinla = sin(lat);
    let d = (1.0 - c * c) / (sinla - c);
    let b2 = b * b;
    let d2 = d * d;
    let b2d2 = 1.0 + b2 / d2;
    let d2b2 = 1.0 + d2 / b2;
    let m = (b * sinla / d - b / 2.0) / b2d2;
    let n = (d2 * sinla / b2 + d / 2.0) / d2b2;
    let cl = cos(lat);
    let mut x = m + sign(lon) * (m * m + cl * cl / b2d2).sqrt();
    let mut y = n + sign(lon)
        * sign(-lat * b)
        * (n * n - ((d2b2 - 1.0) * (sinla * sinla) + d * sinla - 1.0) / d2b2).sqrt();
    // Special cases where the general formula divides by zero.
    if lon0 && latmax {
        x = 0.0;
    }
    if lat0 {
        x = lon / PI;
    }
    if lonmax {
        x = lon * cl / PI;
    }
    if lon0 {
        y = lat * 2.0 / PI;
    }
    if latmax {
        y = sign(lat);
    }
    if lat0 {
        y = 0.0;
    }
    if lonmax {
        y = sinla;
    }
    if global {
        (x / (0.5 + SQRT_2), y / SQRT_2)
    } else {
        (x / 2.0, y / 2.0)
    }
}

fn nicolosi_pos_pointwise(ctx: &ProjCtx) -> impl Fn(f64, f64) -> (f64, f64) + Sync + use<> {
    let hem = ctx.azim_type(true);
    let global = ctx.azim_type(false) == AzimType::Global;
    move |lon, lat| {
        layout_pos(lon, lat, hem, 0.5, |lon, lat| {
            nicolosi_pos1(lon, lat, global)
        })
    }
}

pub fn nicolosi_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    map2(lon, lat, nicolosi_pos_pointwise(ctx))
}

pub fn nicolosi_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let ctx = ctx.with_azim(ctx.azim_type(true));
    let hem = ctx.azim_type(false);
    let (mut lon, mut lat) = map2(x, y, |x, y| {
        layout_coords(x, y, hem, 0.5, |x, y| {
            super::azimuthal::from_polar((x * x + y * y).sqrt(), x.atan2(-y))
        })
    });
    let vis = nicolosi_vis(x, y, &ctx);
    invert(
        x,
        y,
        &mut lon,
        &mut lat,
        nicolosi_pos_pointwise(&ctx),
        Some(&vis),
        &ctx,
    );
    (lon, lat)
}

pub fn nicolosi_vis(x: &[f64], y: &[f64], ctx: &ProjCtx) -> Vec<bool> {
    let hem = ctx.azim_type(true);
    mask2(x, y, |x, y| layout_vis(x, y, hem))
}

pub fn nicolosi_ratio(ctx: &ProjCtx) -> f64 {
    if ctx.azim_type(true) == AzimType::Bihem {
        2.0
    } else {
        1.0
    }
}

/// The oval's `x` for a meridian, within the central circle.
fn ortelius_inner_x(abslon: f64, y: f64) -> f64 {
    let f = (PI.powf(2.0) / (4.0 * abslon) + abslon) / 2.0;
    let h = y * PI / 2.0;
    abslon - f + (f * f - h * h).sqrt()
}

fn ortelius_pos1(lon: f64, lat: f64) -> (f64, f64) {
    let y = lat * 2.0 / PI;
    let x = ortelius_inner_x(lon.abs(), y);
    (2.0 * sign(lon) * x / PI, y)
}

fn ortelius_pos2(lon: f64, lat: f64) -> (f64, f64) {
    let y = lat * 2.0 / PI;
    let abslon = lon.abs();
    let x = if abslon > PI / 2.0 {
        (PI.powf(2.0) / 4.0 - lat * lat).sqrt() + abslon - PI / 2.0
    } else {
        ortelius_inner_x(abslon, y)
    };
    (if lon > 0.0 { x / PI } else { -x / PI }, y)
}

fn ortelius_coords1(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (mut lon, mut lat) = map2(x, y, hammer_coords);
    let vis = mask2(x, y, circ_vis);
    invert(x, y, &mut lon, &mut lat, ortelius_pos1, Some(&vis), ctx);
    let lat = y.iter().map(|y| y * PI / 2.0).collect();
    (lon, lat)
}

fn ortelius_coords2(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (mut lon, mut lat) = map2(x, y, hammer_coords);
    let vis = mask2(x, y, circ_vis);
    invert(x, y, &mut lon, &mut lat, ortelius_pos2, Some(&vis), ctx);
    // Latitude is exact from y; longitude outside the central circle has a closed form.
    x.iter()
        .zip(y)
        .zip(&lon)
        .map(|((&x, &y), &lon1)| {
            let lat = y * PI / 2.0;
            let lon2 = x.abs() * PI + PI / 2.0 - (PI.powf(2.0) / 4.0 - lat * lat).sqrt();
            let lon2 = if x > 0.0 { lon2 } else { -lon2 };
            let outside = ((x * 2.0) * (x * 2.0) + y * y).sqrt() > 1.0;
            (if outside { lon2 } else { lon1 }, lat)
        })
        .unzip()
}

pub fn ortelius_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    match ctx.azim_type(false) {
        AzimType::Hem => ortelius_coords1(x, y, ctx),
        AzimType::Bihem => {
            let xs: Vec<f64> = x.iter().map(|&x| bihem_split(x)).collect();
            let (lon, lat) = ortelius_coords1(&xs, y, ctx);
            let lon = x
                .iter()
                .zip(lon)
                .map(|(&x, lon)| bihem_lon_out(x, lon))
                .collect();
            (lon, lat)
        }
        _ => ortelius_coords2(x, y, ctx),
    }
}

pub fn ortelius_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(false);
    if hem.is_hemispheric() {
        map2(lon, lat, |lon, lat| {
            layout_pos(lon, lat, hem, 1.0, ortelius_pos1)
        })
    } else {
        map2(lon, lat, ortelius_pos2)
    }
}

pub fn ortelius_vis(x: &[f64], y: &[f64], ctx: &ProjCtx) -> Vec<bool> {
    let hem = ctx.azim_type(false);
    if hem.is_hemispheric() {
        mask2(x, y, |x, y| layout_vis(x, y, hem))
    } else {
        mask2(x, y, pill_vis)
    }
}

pub fn ortelius_ratio(ctx: &ProjCtx) -> f64 {
    if ctx.azim_type(false) == AzimType::Hem {
        1.0
    } else {
        2.0
    }
}

pub fn psstereo_coords(x: f64, y: f64) -> (f64, f64) {
    let (lon, lat) = stereo_coords1(x * 0.5, y * 0.5);
    (lon * 2.0, lat)
}

pub fn psstereo_pos(lon: f64, lat: f64) -> (f64, f64) {
    let (x, y) = stereo_pos1(lon / 2.0, lat);
    (x / 0.5, y / 0.5)
}

pub fn psortho_coords(x: f64, y: f64) -> (f64, f64) {
    let (lon, lat) = ortho_coords1(x * 0.5, y * 0.5);
    (lon * 2.0, lat)
}

pub fn psortho_pos(lon: f64, lat: f64) -> (f64, f64) {
    let (x, y) = ortho_pos1(lon / 2.0, lat);
    (x / 0.5, y / 0.5)
}
