//! Rotating geographic coordinates between map aspects (by Mads de Silva in the original).

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::math::{cos, sin};

/// A map's orientation: central longitude, central latitude, and clockwise
/// rotation from north, all in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aspect {
    pub lon: f64,
    pub lat: f64,
    pub rot: f64,
}

type Matrix = [[f64; 3]; 3];

impl Aspect {
    pub fn from_degrees([lon, lat, rot]: [f64; 3]) -> Self {
        Aspect {
            lon: lon.to_radians(),
            lat: lat.to_radians(),
            rot: rot.to_radians(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.lon == 0.0 && self.lat == 0.0 && self.rot == 0.0
    }

    /// Rotation matrix: a clockwise rotation about the y axis followed by an
    /// anticlockwise rotation about the z axis.
    fn matrix(&self) -> Matrix {
        let (sl, cl) = (sin(self.lon), cos(self.lon));
        let (sp, cp) = (sin(self.lat), cos(self.lat));
        let (sr, cr) = (sin(self.rot), cos(self.rot));
        [
            [cl * cp, -sl * cr + sr * sp * cl, -sp * cl * cr - sr * sl],
            [sl * cp, cl * cr + sr * sp * sl, -sp * sl * cr + sr * cl],
            [sp, -sr * cp, cp * cr],
        ]
    }

    /// Rotates coordinates relative to this aspect into standard orientation.
    pub fn rotate_from(&self, lon: &mut [f64], lat: &mut [f64]) {
        if self.lat != 0.0 || self.rot != 0.0 {
            apply(&self.matrix(), lon, lat);
        } else if self.lon != 0.0 {
            // A pure longitude shift is a simple frameshift.
            let a = self.lon;
            lon.par_iter_mut().for_each(|l| {
                *l += a;
                if a > 0.0 && *l > PI {
                    *l -= 2.0 * PI;
                } else if a <= 0.0 && *l < -PI {
                    *l += 2.0 * PI;
                }
            });
        }
    }

    /// Rotates coordinates in standard orientation into this aspect.
    pub fn rotate_to(&self, lon: &mut [f64], lat: &mut [f64]) {
        if self.lat != 0.0 || self.rot != 0.0 {
            let m = self.matrix();
            let inverse = std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]));
            apply(&inverse, lon, lat);
        } else if self.lon != 0.0 {
            let a = self.lon;
            lon.par_iter_mut().for_each(|l| {
                *l -= a;
                if a < 0.0 && *l > PI {
                    *l -= 2.0 * PI;
                } else if a >= 0.0 && *l < -PI {
                    *l += 2.0 * PI;
                }
            });
        }
    }
}

fn apply(m: &Matrix, lon: &mut [f64], lat: &mut [f64]) {
    lon.par_iter_mut()
        .zip(lat.par_iter_mut())
        .for_each(|(lon, lat)| {
            let p = [cos(*lon) * cos(*lat), sin(*lon) * cos(*lat), sin(*lat)];
            // Accumulate with fused multiply-adds, as the BLAS matrix product in
            // the original does (Accelerate on macOS), so results match exactly.
            let q: [f64; 3] = std::array::from_fn(|i| {
                m[i][2].mul_add(p[2], m[i][1].mul_add(p[1], m[i][0] * p[0]))
            });
            *lat = q[2].asin();
            *lon = q[1].atan2(q[0]);
        });
}
