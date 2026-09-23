//! Azimuthal projections, and the hemisphere layouts shared with other
//! projections that support them.

use std::f64::consts::{PI, SQRT_2};

use super::{AzimType, ProjCtx, circ_vis, map2, mask2};
use crate::math::{cos, sin};

/// Converts polar coordinates around the equator (radius in half-turns) to lon/lat.
pub fn from_polar(rh: f64, th: f64) -> (f64, f64) {
    let rh = rh * PI;
    let lat = (-sin(rh) * cos(th)).asin();
    let lon = (sin(th) * sin(rh)).atan2(cos(rh));
    (lon, lat)
}

/// Converts lon/lat to polar coordinates around the equator (radius in half-turns).
pub fn to_polar(lon: f64, lat: f64) -> (f64, f64) {
    let rh = (cos(lat) * cos(lon)).acos() / PI;
    let th = (sin(lon) * cos(lat)).atan2(sin(lat));
    (rh, th)
}

/// Maps a position on one hemisphere of a bihemisphere map to that hemisphere's own frame.
pub fn bihem_split(x: f64) -> f64 {
    if x > 0.0 {
        2.0 * x - 1.0
    } else {
        2.0 * x + 1.0
    }
}

/// Shifts longitudes from a hemisphere's own frame back to the bihemisphere map.
pub fn bihem_lon_out(x: f64, lon: f64) -> f64 {
    if x > 0.0 {
        lon + PI / 2.0
    } else {
        lon - PI / 2.0
    }
}

/// Shifts longitudes into the frame of the hemisphere they fall on.
pub fn bihem_lon_in(lon: f64) -> f64 {
    (if lon > 0.0 { lon } else { lon + PI }) - PI / 2.0
}

/// Places a position from a hemisphere's own frame onto the bihemisphere map.
pub fn bihem_x_out(lon: f64, x: f64) -> f64 {
    let x = x / 2.0;
    if lon > 0.0 { x + 0.5 } else { x - 0.5 }
}

/// Applies an azimuthal layout to a pointwise inverse function `f`, whose
/// hemisphere spans `[-s, s]`.
pub fn layout_coords(
    x: f64,
    y: f64,
    hem: AzimType,
    s: f64,
    f: impl Fn(f64, f64) -> (f64, f64),
) -> (f64, f64) {
    match hem {
        AzimType::Global | AzimType::GlobalIf => f(x, y),
        AzimType::Hem => f(x * s, y * s),
        AzimType::Bihem => {
            let (lon, lat) = f(bihem_split(x) * s, y * s);
            (bihem_lon_out(x, lon), lat)
        }
    }
}

/// Applies an azimuthal layout to a pointwise forward function `f`.
pub fn layout_pos(
    lon: f64,
    lat: f64,
    hem: AzimType,
    s: f64,
    f: impl Fn(f64, f64) -> (f64, f64),
) -> (f64, f64) {
    match hem {
        AzimType::Global | AzimType::GlobalIf => f(lon, lat),
        AzimType::Hem => {
            let (x, y) = f(lon, lat);
            (x / s, y / s)
        }
        AzimType::Bihem => {
            let (x, y) = f(bihem_lon_in(lon), lat);
            (bihem_x_out(lon, x / s), y / s)
        }
    }
}

pub fn layout_vis(x: f64, y: f64, hem: AzimType) -> bool {
    match hem {
        AzimType::Bihem => circ_vis(1.0 - (x * 2.0).abs(), y),
        _ => circ_vis(x, y),
    }
}

pub fn azim_vis(x: &[f64], y: &[f64], ctx: &ProjCtx, no_glob: bool) -> Vec<bool> {
    let hem = ctx.azim_type(no_glob);
    mask2(x, y, |x, y| layout_vis(x, y, hem))
}

pub fn azim_ratio(ctx: &ProjCtx, no_glob: bool) -> f64 {
    if ctx.azim_type(no_glob) == AzimType::Bihem {
        2.0
    } else {
        1.0
    }
}

fn azimeq_coords1(x: f64, y: f64) -> (f64, f64) {
    from_polar((x * x + y * y).sqrt(), x.atan2(-y))
}

fn azimeq_pos1(lon: f64, lat: f64) -> (f64, f64) {
    let (rh, th) = to_polar(lon, lat);
    (rh * sin(th), rh * cos(th))
}

pub fn azimeq_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(false);
    map2(x, y, |x, y| layout_coords(x, y, hem, 0.5, azimeq_coords1))
}

pub fn azimeq_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(false);
    map2(lon, lat, |lon, lat| {
        layout_pos(lon, lat, hem, 0.5, azimeq_pos1)
    })
}

fn lambert_coords1(x: f64, y: f64) -> (f64, f64) {
    let rh = (x * x + y * y).sqrt() % 1.0;
    let th = x.atan2(-y);
    from_polar(2.0 * rh.asin() / PI, th)
}

fn lambert_pos1(lon: f64, lat: f64) -> (f64, f64) {
    let (rh, th) = to_polar(lon, lat);
    let rh1 = sin(rh * PI / 2.0);
    (rh1 * sin(th), rh1 * cos(th))
}

pub fn lambert_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(false);
    map2(x, y, |x, y| {
        layout_coords(x, y, hem, SQRT_2 / 2.0, lambert_coords1)
    })
}

pub fn lambert_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(false);
    map2(lon, lat, |lon, lat| {
        layout_pos(lon, lat, hem, SQRT_2 / 2.0, lambert_pos1)
    })
}

pub fn stereo_coords1(x: f64, y: f64) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    let th = x.atan2(-y);
    let rh = ((2.0 * r).atan() - PI / 4.0) * 2.0 / PI + 0.5;
    from_polar(rh, th)
}

pub fn stereo_pos1(lon: f64, lat: f64) -> (f64, f64) {
    let (rh, th) = to_polar(lon, lat);
    let r = (PI / 4.0 + PI * (rh - 0.5) / 2.0).tan() / 2.0;
    (r * sin(th), r * cos(th))
}

/// Stereographic maps can only be global when truncated; the scale then
/// depends on the maximum angular distance from the center (the reference).
fn stereo_layout(ctx: &ProjCtx) -> (AzimType, f64) {
    if ctx.azim_type(false) == AzimType::Global {
        let r = ctx.reference_rad(&[90.0])[0];
        (AzimType::Hem, (r / 2.0).tan() / 2.0)
    } else {
        (ctx.azim_type(true), 0.5)
    }
}

pub fn stereo_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (hem, s) = stereo_layout(ctx);
    map2(x, y, |x, y| layout_coords(x, y, hem, s, stereo_coords1))
}

pub fn stereo_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (hem, s) = stereo_layout(ctx);
    map2(lon, lat, |lon, lat| {
        layout_pos(lon, lat, hem, s, stereo_pos1)
    })
}

pub fn ortho_coords1(x: f64, y: f64) -> (f64, f64) {
    let (x2, y2) = (2.0 * x, 2.0 * y);
    let rh = (x2 * x2 + y2 * y2).sqrt();
    let c = (rh % 1.0).asin();
    let lat = (2.0 * y).asin();
    let lon = (2.0 * x * sin(c) / (rh * cos(c))).atan();
    (lon, lat)
}

pub fn ortho_pos1(lon: f64, lat: f64) -> (f64, f64) {
    // Points on the far hemisphere are pushed off the map area.
    let x = if lon.abs() > PI / 2.0 {
        1.0
    } else {
        cos(lat) * sin(lon) / 2.0
    };
    (x, sin(lat) / 2.0)
}

pub fn ortho_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(true);
    map2(x, y, |x, y| layout_coords(x, y, hem, 0.5, ortho_coords1))
}

pub fn ortho_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let hem = ctx.azim_type(true);
    map2(lon, lat, |lon, lat| {
        layout_pos(lon, lat, hem, 0.5, ortho_pos1)
    })
}
