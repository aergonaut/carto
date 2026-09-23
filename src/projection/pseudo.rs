//! Pseudocylindrical and pseudoazimuthal projections.

use std::f64::consts::{PI, SQRT_2};

use super::iterative::{invert, solve_auxiliary};
use super::{ProjCtx, map2, mask2};
use crate::math::{cos, sign, sin, sinc};

pub fn sin_coords(x: f64, y: f64) -> (f64, f64) {
    let lat = PI * y / 2.0;
    (PI * x / cos(lat), lat)
}

pub fn sin_pos(lon: f64, lat: f64) -> (f64, f64) {
    (cos(lat) * lon / PI, 2.0 * lat / PI)
}

pub fn sin_vis(x: f64, y: f64) -> bool {
    !(x.abs() > cos(PI * y / 2.0))
}

pub fn moll_coords(x: f64, y: f64) -> (f64, f64) {
    let th = y.asin();
    let lat = ((2.0 * th + sin(2.0 * th)) / PI).asin();
    (PI * x / cos(th), lat)
}

pub fn moll_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let mut th = lat.to_vec();
    solve_auxiliary(
        lat,
        &mut th,
        |th, lat| th - (2.0 * th + sin(2.0 * th) - PI * sin(lat)) / (2.0 + 2.0 * cos(2.0 * th)),
        |th, lat| (2.0 * th + sin(2.0 * th)) / (PI * sin(lat)) - 1.0,
        ctx,
    );
    map2(lon, &th, |lon, th| (lon * cos(th) / PI, sin(th)))
}

pub fn hammer_coords(x: f64, y: f64) -> (f64, f64) {
    let x1 = x * SQRT_2 * 2.0;
    let y1 = y * SQRT_2;
    let z = (1.0 - (x1 / 4.0) * (x1 / 4.0) - (y1 / 2.0) * (y1 / 2.0)).sqrt();
    let lon = 2.0 * (z * x1 / (2.0 * (2.0 * (z * z) - 1.0))).atan();
    (lon, (z * y1).asin())
}

pub fn hammer_pos(lon: f64, lat: f64) -> (f64, f64) {
    let d = (1.0 + cos(lat) * cos(lon / 2.0)).sqrt();
    (cos(lat) * sin(lon / 2.0) / d, sin(lat) / d)
}

pub fn eckert4_coords(x: f64, y: f64) -> (f64, f64) {
    let r = 1.0 / (2.0 * (PI / (4.0 + PI)).sqrt());
    let th = (y * (4.0 + PI).sqrt() / (2.0 * PI.sqrt() * r)).asin();
    let (s, c) = (sin(th), cos(th));
    let lat = ((th + s * c + 2.0 * s) / (2.0 + PI / 2.0)).asin();
    let lon = 2.0 * x * (4.0 * PI + PI.powf(2.0)).sqrt() / (2.0 * r * (1.0 + c));
    (lon, lat)
}

pub fn eckert4_pos(lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let k = 2.0 + PI / 2.0;
    let mut th: Vec<f64> = lat.iter().map(|l| l / 2.0).collect();
    solve_auxiliary(
        lat,
        &mut th,
        |th, lat| {
            let (s, c) = (sin(th), cos(th));
            th - (th + s * c + 2.0 * s - k * sin(lat)) / (2.0 * c * (1.0 + c))
        },
        |th, lat| {
            let (s, c) = (sin(th), cos(th));
            (th + s * c + 2.0 * s) - k * sin(lat)
        },
        ctx,
    );
    map2(lon, &th, |lon, th| {
        (lon * (1.0 + cos(th)) / (2.0 * PI), sin(th))
    })
}

const EQEAR_C: [f64; 4] = [1.340264, -0.081106, 0.000893, 0.003796];

fn equal_earth_denom() -> f64 {
    let c = EQEAR_C;
    let th = (3f64.sqrt() / 2.0).asin();
    c[3] * th.powf(9.0) + c[2] * th.powf(7.0) + c[1] * th.powf(3.0) + c[0] * th
}

pub fn equal_earth_ratio() -> f64 {
    let num = 2.0 * 3f64.sqrt() * PI / 3.0 * EQEAR_C[0];
    0.5 * num / equal_earth_denom()
}

pub fn equal_earth_pos(lon: f64, lat: f64) -> (f64, f64) {
    let c = EQEAR_C;
    let th = (sin(lat) * 3f64.sqrt() / 2.0).asin();
    let mut x = lon * cos(th);
    x /= 9.0 * c[3] * th.powf(8.0) + 7.0 * c[2] * th.powf(6.0) + 3.0 * c[1] * (th * th) + c[0];
    let y = c[3] * th.powf(9.0) + c[2] * th.powf(7.0) + c[1] * th.powf(3.0) + c[0] * th;
    (x / (PI / c[0]), y / equal_earth_denom())
}

pub fn equal_earth_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (mut lon, mut lat) = map2(x, y, wagner_coords);
    invert(x, y, &mut lon, &mut lat, equal_earth_pos, None, ctx);
    (lon, lat)
}

pub const WINKEL_RATIO: f64 = (1.0 + PI / 2.0) / (PI / 2.0);

pub fn winkel_pos(lon: f64, lat: f64) -> (f64, f64) {
    let al = (cos(lat) * cos(lon / 2.0)).acos() / PI;
    let x = (lon / PI + cos(lat) * sin(lon / 2.0) / sinc(al)) / (1.0 + PI / 2.0);
    let y = (lat + sin(lat) / sinc(al)) / PI;
    (x, y)
}

pub fn winkel_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (mut lon, mut lat) = map2(x, y, aitoff_coords);
    let vis = mask2(x, y, winkel_vis);
    invert(x, y, &mut lon, &mut lat, winkel_pos, Some(&vis), ctx);
    (lon, lat)
}

pub fn winkel_vis(x: f64, y: f64) -> bool {
    let x1 = if x.abs() > 1.0 { 1.0 } else { x };
    let lat1 = (2.0 * (x1.abs() * (1.0 + PI / 2.0) - 1.0) / PI).acos();
    let ymax = (lat1 + PI / 2.0 * sin(lat1)) / PI;
    !(y.abs() > ymax)
}

pub const ROBINSON_RATIO: f64 = PI * 0.8487 / 1.3523;

const ROB_A: [f64; 19] = [
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350,
    0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322,
];
const ROB_B: [f64; 19] = [
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769,
    0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000,
];
const ROB_CP: [f64; 19] = [
    0.40711579454,
    -0.00875326537,
    -0.01069796348,
    -0.01167039606,
    -0.00680782592,
    -0.01847822803,
    -0.02090931959,
    -0.01847842619,
    -0.02090971277,
    -0.01410147990,
    -0.02236858853,
    -0.01701955610,
    -0.01215649454,
    -0.01069792545,
    -0.02090967766,
    -0.03160740722,
    0.01361549135,
    0.04425022432,
    0.60843116534,
];
const ROB_CQ: [f64; 19] = [
    0.91083562255,
    -0.00000589975,
    0.00000564852,
    -0.00000557909,
    0.00000555879,
    -0.00000001291,
    -0.00000546138,
    -0.00154708482,
    -0.00387351841,
    -0.00619324913,
    -0.00930492848,
    -0.01239340212,
    -0.01549814705,
    -0.01937169560,
    -0.02401844414,
    -0.03331171624,
    -0.07051393824,
    -0.09917388904,
    0.24527101656,
];
const ROB_CM: [f64; 19] = [
    0.47371661113,
    -0.00911028522,
    -0.01113479305,
    -0.01214704697,
    -0.00708577740,
    -0.01923282436,
    -0.02176345915,
    -0.01957843209,
    -0.02288586729,
    -0.01676092031,
    -0.02731224791,
    -0.02386224240,
    -0.02119239013,
    -0.02327513775,
    -0.04193330922,
    -0.07123235442,
    -0.06423048161,
    -0.10536278437,
    1.00598851957,
];
const ROB_CN: [f64; 19] = [
    1.07729625255,
    -0.00012324928,
    -0.00032923415,
    -0.00056627609,
    -0.00045168290,
    -0.00141388769,
    -0.00211521349,
    -0.00083658786,
    0.00073523299,
    0.00349045186,
    0.00502041018,
    0.00860101415,
    0.01281238969,
    0.01794606372,
    0.02090220870,
    0.02831504310,
    0.11177176318,
    0.28108668066,
    -0.45126573496,
];

/// Multiquadric-interpolated Robinson x-extent for a scaled `|y|`.
fn robinson_a_st(b_st: f64) -> f64 {
    let mut a_st = 0.0;
    for j in 0..19 {
        a_st += ROB_CM[j] * (ROB_B[j] * 1.3523 - b_st).abs();
    }
    a_st
}

/// Multiquadric interpolation from Ipbuker 2013 (<https://doi.org/10.1559/1523040041649425>).
///
/// The paper has some apparent errors and ambiguities: the sums in equations
/// 22 and 23 should run from j=0 to j=18, the 5-degree steps should be in
/// radians with latitude also in radians, and latitude should be taken as an
/// absolute value (nested within the abs) with the sign of the resulting y
/// restored afterwards; likewise for the inverse projection with abs(y).
pub fn robinson_coords(x: f64, y: f64) -> (f64, f64) {
    let b_st = y.abs() * 1.3523;
    let a_st = robinson_a_st(b_st);
    let lon = (x * (0.8487 * PI)) / a_st;
    let mut lat = 0.0;
    for j in 0..19 {
        let da = ROB_A[j] * 0.8487 - a_st;
        let db = ROB_B[j] * 1.3523 - b_st;
        lat += ROB_CN[j] * (da * da + db * db).sqrt();
    }
    (lon, lat * sign(y))
}

pub fn robinson_pos(lon: f64, lat: f64) -> (f64, f64) {
    let k = 5f64.to_radians();
    let alat = lat.abs();
    let (mut x, mut y) = (0.0, 0.0);
    for j in 0..19 {
        let d = (k * j as f64 - alat).abs();
        x += ROB_CP[j] * d;
        y += ROB_CQ[j] * d;
    }
    (lon * x / (0.8487 * PI), sign(lat) * y / 1.3523)
}

pub fn robinson_vis(x: f64, y: f64) -> bool {
    !(x.abs() > robinson_a_st(y.abs() * 1.3523) / 0.8487)
}

pub fn wagner_coords(x: f64, y: f64) -> (f64, f64) {
    let lat = y * PI / 2.0;
    let ph = (lat * 3f64.sqrt() / PI).asin();
    (x * PI / cos(ph), lat)
}

pub fn wagner_pos(lon: f64, lat: f64) -> (f64, f64) {
    let l = lat / PI;
    (lon / PI * (1.0 - 3.0 * (l * l)).sqrt(), lat * 2.0 / PI)
}

pub fn wagner_vis(x: f64, y: f64) -> bool {
    let h = y / 2.0;
    !(x.abs() > (1.0 - 3.0 * (h * h)).sqrt())
}

const NATEAR_CA: [f64; 5] = [0.870700, -0.131979, -0.013791, 0.003971, -0.001529];
const NATEAR_CB: [f64; 5] = [1.007226, 0.015085, -0.044475, 0.028874, -0.005916];

fn natural_earth_denom() -> f64 {
    let c = NATEAR_CB;
    let h = PI / 2.0;
    c[0] + c[1] * h.powf(2.0) + c[2] * h.powf(6.0) + c[3] * h.powf(8.0) + c[4] * h.powf(10.0)
}

pub fn natural_earth_ratio() -> f64 {
    2.0 * NATEAR_CA[0] / natural_earth_denom()
}

pub fn natural_earth_pos(lon: f64, lat: f64) -> (f64, f64) {
    let (a, b) = (NATEAR_CA, NATEAR_CB);
    let lat2 = lat * lat;
    let lat10 = lat.powf(10.0);
    let x =
        lon * (a[0] + a[1] * lat2 + a[2] * lat.powf(4.0) + a[3] * lat10 + a[4] * lat.powf(12.0));
    let y = lat * (b[0] + b[1] * lat2 + b[2] * lat.powf(6.0) + b[3] * lat.powf(8.0) + b[4] * lat10);
    (x / (a[0] * PI), y / (natural_earth_denom() * PI / 2.0))
}

pub fn natural_earth_coords(x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
    let (mut lon, mut lat) = map2(x, y, wagner_coords);
    invert(x, y, &mut lon, &mut lat, natural_earth_pos, None, ctx);
    (lon, lat)
}

pub fn aitoff_coords(x: f64, y: f64) -> (f64, f64) {
    let (hx, hy) = (x / 2.0, y / 2.0);
    let rh = (hx * hx + hy * hy).sqrt();
    let th = hx.atan2(-y / 2.0);
    let (lon, lat) = super::azimuthal::from_polar(rh, th);
    (lon * 2.0, lat)
}

pub fn aitoff_pos(lon: f64, lat: f64) -> (f64, f64) {
    let al = PI / 2.0 * sinc((cos(lat) * cos(lon / 2.0)).acos() / PI);
    (cos(lat) * sin(lon / 2.0) / al, sin(lat) / al)
}

/// Wagner VII (Hammer-Wagner) is Hammer applied to a compressed sphere:
/// longitude is scaled by 2/3 and latitude mapped through `sin 65° · sin(lat)`.
const WAGNER7_SIN: f64 = 0.906_307_787_036_65; // sin 65°

/// Hammer's normalized `y` at the map's corners (pole, ±180°), which becomes
/// Wagner VII's `y = 1`. The pole lines bow outward, so this is their highest point.
fn wagner7_ymax() -> f64 {
    hammer_pos(PI / 1.5, WAGNER7_SIN.asin()).1
}

pub fn wagner7_ratio() -> f64 {
    // Snyder's x and y scale factors, 2.66723 and 1.24104, make the map equal-area.
    2.66723 / (1.24104 * SQRT_2 * wagner7_ymax())
}

/// Longitude and the sine of latitude, which exceeds 1 beyond the pole lines.
fn wagner7_lon_sinlat(x: f64, y: f64) -> (f64, f64) {
    let (lon, lat) = hammer_coords(x / SQRT_2, y * wagner7_ymax());
    (lon * 1.5, sin(lat) / WAGNER7_SIN)
}

pub fn wagner7_coords(x: f64, y: f64) -> (f64, f64) {
    let (lon, sinlat) = wagner7_lon_sinlat(x, y);
    // Clamped so points on the pole lines don't round to NaN; `wagner7_vis`
    // excludes the points beyond them.
    (lon, sinlat.clamp(-1.0, 1.0).asin())
}

pub fn wagner7_pos(lon: f64, lat: f64) -> (f64, f64) {
    let (x, y) = hammer_pos(lon / 1.5, (WAGNER7_SIN * sin(lat)).asin());
    (x * SQRT_2, y / wagner7_ymax())
}

pub fn wagner7_vis(x: f64, y: f64) -> bool {
    // Outside the underlying Hammer ellipse both values are NaN.
    let (lon, sinlat) = wagner7_lon_sinlat(x, y);
    lon.abs() <= PI && sinlat.abs() <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wagner VII as published (Snyder 1993; PROJ `wag7`).
    fn wagner7_reference(lon: f64, lat: f64) -> (f64, f64) {
        let s = 0.906_307_787_036_65 * lat.sin();
        let c0 = (1.0 - s * s).sqrt();
        let c1 = (2.0 / (1.0 + c0 * (lon / 3.0).cos())).sqrt();
        (2.66723 * c0 * c1 * (lon / 3.0).sin(), 1.24104 * s * c1)
    }

    #[test]
    fn wagner7_matches_reference() {
        let (xmax, _) = wagner7_reference(PI, 0.0);
        let (_, ymax) = wagner7_reference(PI, PI / 2.0);
        assert!((wagner7_ratio() - xmax / ymax).abs() < 1e-12);
        for i in -6..=6 {
            for j in -3..=3 {
                let (lon, lat) = (i as f64 * PI / 6.0, j as f64 * PI / 6.0);
                let (x, y) = wagner7_pos(lon, lat);
                let (rx, ry) = wagner7_reference(lon, lat);
                assert!((x - rx / xmax).abs() < 1e-5 && (y - ry / ymax).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn wagner7_round_trips() {
        for i in -12..=12 {
            for j in -6..=6 {
                let (lon, lat) = (i as f64 * PI / 12.0, j as f64 * PI / 12.0);
                let (x, y) = wagner7_pos(lon, lat);
                let (lon2, lat2) = wagner7_coords(x, y);
                assert!((lon - lon2).abs() < 1e-9 || lat.abs() == PI / 2.0);
                // `asin` near the poles loses about half the digits.
                assert!((lat - lat2).abs() < 1e-7);
            }
        }
        // The pole line dips toward the center: its corners reach y = 1, its middle doesn't.
        let (xc, yc) = wagner7_pos(PI * 0.99, PI / 2.0 * 0.99);
        assert!(wagner7_vis(0.99, 0.0) && wagner7_vis(xc, yc));
        assert!(wagner7_vis(0.0, 0.9 * wagner7_pos(0.0, PI / 2.0).1));
        assert!(!wagner7_vis(0.0, 0.99) && !wagner7_vis(1.01, 0.0) && !wagner7_vis(0.0, 1.01));
    }
}
