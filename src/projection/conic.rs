//! Conic projections.
//!
//! Conic maps are laid out as a single global fan, a single (northern)
//! hemisphere, or a bihemisphere with the southern hemisphere mirrored below.

use std::f64::consts::PI;

use super::{AzimType, ProjCtx, coords_vis, map2};
use crate::math::{cos, sign, sin};

const DEFAULT_PARALLELS: [f64; 2] = [15.0, 45.0];

/// Which parts of the map lie outside the fan, split out so that the far
/// (outer) edge can be treated separately when padding.
fn fan_vis(close: bool, far: bool, sides: bool) -> (bool, bool) {
    (!(close || far || sides), far)
}

/// Visibility from known lon/lat: hemispheric layouts treat the southern
/// hemisphere as off the map by stretching latitudes.
fn lonlat_vis(
    y: &[f64],
    lon: &[f64],
    lat: &[f64],
    hem: AzimType,
    min_lat: Option<f64>,
) -> (Vec<bool>, Vec<bool>) {
    let lat1: Vec<f64> = lat
        .iter()
        .zip(y)
        .map(|(&lat, &y)| match hem {
            AzimType::Hem => lat * 2.0 - PI / 2.0,
            AzimType::Bihem => lat * sign(y) * 2.0 - PI / 2.0,
            _ => match min_lat {
                Some(min) => {
                    if lat > min {
                        lat
                    } else {
                        -2.0
                    }
                }
                None => lat,
            },
        })
        .collect();
    let far = lat1.iter().map(|&l| l < -PI / 2.0).collect();
    (coords_vis(lon, &lat1), far)
}

/// Equidistant conic.
pub struct Equidistant {
    hem: AzimType,
    n: f64,
    g: f64,
    max_x: f64,
    max_y: f64,
}

impl Equidistant {
    pub fn new(ctx: &ProjCtx) -> Self {
        let r = ctx.reference_rad(&DEFAULT_PARALLELS);
        let n = if r[0] == r[1] {
            sin(r[0])
        } else {
            (cos(r[0]) - cos(r[1])) / (r[1] - r[0])
        };
        let g = cos(r[0]) / n + r[0];
        let hem = ctx.azim_type(false);
        let (max_x, max_y) = if n > 0.5 {
            if hem.is_hemispheric() {
                (g, g - g * cos(n * PI))
            } else {
                (g + PI / 2.0, g - (g + PI / 2.0) * cos(n * PI))
            }
        } else {
            let max_x = if hem.is_hemispheric() {
                g * sin(n * PI)
            } else {
                (g + PI / 2.0) * sin(n * PI)
            };
            (max_x, g - (g - PI / 2.0) * cos(n * PI))
        };
        Equidistant {
            hem,
            n,
            g,
            max_x,
            max_y,
        }
    }

    fn unit_y(&self, y: f64) -> f64 {
        match self.hem {
            AzimType::Hem => (y + 1.0) * self.max_y / 2.0,
            AzimType::Bihem => (y * self.max_y).abs(),
            _ => y * (self.max_y / 2.0 + PI / 4.0) + (self.max_y / 2.0 - PI / 4.0),
        }
    }

    pub fn coords(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        map2(x, y, |x, y| {
            let x1 = x * self.max_x;
            let y1 = self.unit_y(y);
            let rho = sign(self.n) * (x1 * x1 + (self.g - y1) * (self.g - y1)).sqrt();
            let mut lat = self.g - rho;
            let lon = x1.atan2(self.g - y1) / self.n;
            if self.hem == AzimType::Bihem {
                lat *= sign(y);
            }
            (lon, lat)
        })
    }

    pub fn pos(&self, lon: &[f64], lat: &[f64]) -> (Vec<f64>, Vec<f64>) {
        map2(lon, lat, |lon, lat| {
            let rho = self.g
                - if self.hem == AzimType::Bihem {
                    lat.abs()
                } else {
                    lat
                };
            let th = self.n * lon;
            let x = rho * sin(th) / self.max_x;
            let y = self.g - rho * cos(th);
            let y = match self.hem {
                AzimType::Hem => (y / self.max_y) * 2.0 - 1.0,
                AzimType::Bihem => y / self.max_y * sign(lat),
                _ => (y - (self.max_y / 2.0 - PI / 4.0)) / (self.max_y / 2.0 + PI / 4.0),
            };
            (x, y)
        })
    }

    pub fn vis_far(
        &self,
        x: &[f64],
        y: &[f64],
        lonlat: Option<(&[f64], &[f64])>,
        _ctx: &ProjCtx,
    ) -> (Vec<bool>, Option<Vec<bool>>) {
        if let Some((lon, lat)) = lonlat {
            let (vis, far) = lonlat_vis(y, lon, lat, self.hem, None);
            return (vis, Some(far));
        }
        let (g, n) = (self.g, self.n);
        let far_r = if self.hem.is_hemispheric() {
            g.powf(2.0)
        } else {
            (g + PI / 2.0).powf(2.0)
        };
        let close_r = (g - PI / 2.0).powf(2.0);
        let (vis, far): (Vec<bool>, Vec<bool>) = x
            .iter()
            .zip(y)
            .map(|(&x, &y)| {
                let x1 = x * self.max_x;
                let y1 = self.unit_y(y);
                let rho_s = x1 * x1 + (g - y1) * (g - y1);
                fan_vis(
                    rho_s < close_r,
                    rho_s > far_r,
                    x1.abs().atan2(g - y1) > PI * n,
                )
            })
            .unzip();
        (vis, Some(far))
    }

    pub fn ratio(&self) -> f64 {
        match self.hem {
            AzimType::Hem => self.max_x * 2.0 / self.max_y,
            AzimType::Bihem => self.max_x / self.max_y,
            _ => self.max_x * 2.0 / (self.max_y + PI / 2.0),
        }
    }
}

/// Maps y into the fan's frame, shared by the Albers and Lambert conics.
fn fan_unit_y(y: f64, hem: AzimType, max_y: f64, min_y: f64) -> f64 {
    if hem == AzimType::Bihem {
        y.abs() * max_y
    } else {
        y * (max_y - min_y) / 2.0 + (max_y + min_y) / 2.0
    }
}

/// Maps y out of the fan's frame, shared by the Albers and Lambert conics.
fn fan_map_y(y: f64, lat: f64, hem: AzimType, max_y: f64, min_y: f64) -> f64 {
    if hem == AzimType::Bihem {
        sign(lat) * y / max_y
    } else {
        (y - (max_y + min_y) / 2.0) / ((max_y - min_y) / 2.0)
    }
}

fn fan_ratio(hem: AzimType, max_x: f64, max_y: f64, min_y: f64) -> f64 {
    if hem == AzimType::Bihem {
        max_x / max_y
    } else {
        max_x * 2.0 / (max_y - min_y)
    }
}

/// Albers equal-area conic.
pub struct Albers {
    hem: AzimType,
    n: f64,
    c: f64,
    rho0: f64,
    max_x: f64,
    max_y: f64,
    min_y: f64,
}

impl Albers {
    pub fn new(ctx: &ProjCtx) -> Self {
        let r = ctx.reference_rad(&DEFAULT_PARALLELS);
        let n = if r[0] == r[1] {
            sin(r[0])
        } else {
            (sin(r[0]) + sin(r[1])) / 2.0
        };
        let c = cos(r[0]).powf(2.0) + 2.0 * n * sin(r[0]);
        let rho0 = c.sqrt() / n;
        let hem = ctx.azim_type(false);
        let (rhomax, min_y) = if hem.is_hemispheric() {
            (rho0, 0.0)
        } else {
            let rhomax = (c + 2.0 * n).sqrt() / n;
            (rhomax, rho0 - rhomax)
        };
        let (max_x, max_y) = if n > 0.5 {
            (rhomax, rho0 - rhomax * cos(n * PI))
        } else {
            (
                rhomax * sin(n * PI),
                rho0 - ((c - 2.0 * n).sqrt() / n) * cos(n * PI),
            )
        };
        Albers {
            hem,
            n,
            c,
            rho0,
            max_x,
            max_y,
            min_y,
        }
    }

    pub fn coords(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n2 = self.n.powf(2.0);
        map2(x, y, |x, y| {
            let x1 = x * self.max_x;
            let y1 = fan_unit_y(y, self.hem, self.max_y, self.min_y);
            let dy = self.rho0 - y1;
            let rho = (x1 * x1 + dy * dy).sqrt();
            let lon = x1.atan2(dy) / self.n;
            let sinla = (self.c - (rho * rho * n2)) / (2.0 * self.n);
            // Clip to avoid arcsin errors, pushing out-of-range values past the
            // poles so the visible area clips properly.
            let mut lat = if sinla > 1.0 {
                0.01 + PI / 2.0
            } else if sinla < -1.0 {
                -0.01 - PI / 2.0
            } else {
                sinla.asin()
            };
            if self.hem == AzimType::Bihem {
                lat *= sign(y);
            }
            (lon, lat)
        })
    }

    pub fn pos(&self, lon: &[f64], lat: &[f64]) -> (Vec<f64>, Vec<f64>) {
        map2(lon, lat, |lon, lat| {
            let l = if self.hem == AzimType::Bihem {
                lat.abs()
            } else {
                lat
            };
            let rho = (self.c - 2.0 * self.n * sin(l)).sqrt() / self.n;
            let th = self.n * lon;
            let x = rho * sin(th) / self.max_x;
            let y = self.rho0 - rho * cos(th);
            (x, fan_map_y(y, lat, self.hem, self.max_y, self.min_y))
        })
    }

    pub fn vis_far(
        &self,
        x: &[f64],
        y: &[f64],
        lonlat: Option<(&[f64], &[f64])>,
        _ctx: &ProjCtx,
    ) -> (Vec<bool>, Option<Vec<bool>>) {
        if let Some((lon, lat)) = lonlat {
            let (vis, far) = lonlat_vis(y, lon, lat, self.hem, None);
            return (vis, Some(far));
        }
        let (n, c) = (self.n, self.c);
        let n2 = n.powf(2.0);
        let close_r = (c - 2.0 * n) / n2;
        let far_r = if self.hem.is_hemispheric() {
            c / n2
        } else {
            (c + 2.0 * n) / n2
        };
        let (vis, far): (Vec<bool>, Vec<bool>) = x
            .iter()
            .zip(y)
            .map(|(&x, &y)| {
                let x1 = x * self.max_x;
                let dy = self.rho0 - fan_unit_y(y, self.hem, self.max_y, self.min_y);
                let rho_s = x1 * x1 + dy * dy;
                fan_vis(rho_s < close_r, rho_s > far_r, x1.abs().atan2(dy) > n * PI)
            })
            .unzip();
        (vis, Some(far))
    }

    pub fn ratio(&self) -> f64 {
        fan_ratio(self.hem, self.max_x, self.max_y, self.min_y)
    }
}

/// Lambert conformal conic. Global maps are truncated at a minimum latitude.
pub struct Lambert {
    hem: AzimType,
    /// The reference values exactly as given, whose first entry the original
    /// uses as the truncation latitude when testing visibility.
    raw_ref0: f64,
    n: f64,
    f: f64,
    rhomax: f64,
    max_x: f64,
    max_y: f64,
    min_y: f64,
}

impl Lambert {
    pub fn new(ctx: &ProjCtx) -> Self {
        let raw = ctx.reference_rad(&[0.0, 15.0, 45.0]);
        let r = if raw.len() < 3 {
            vec![0.0, raw[0], raw[1]]
        } else {
            raw.clone()
        };
        let t1 = (PI / 4.0 + r[1] / 2.0).tan();
        let t2 = (PI / 4.0 + r[2] / 2.0).tan();
        let n = (cos(r[1]) / cos(r[2])).ln() / (t2 / t1).ln();
        let f = cos(r[1]) * t1.powf(n) / n;
        let hem = ctx.azim_type(false);
        let (rhomax, min_y) = if hem.is_hemispheric() {
            (f, 0.0)
        } else {
            let rhomax = f / (PI / 4.0 + r[0] / 2.0).tan().powf(n);
            (rhomax, f - rhomax)
        };
        let (max_x, max_y) = if n > 0.5 {
            (rhomax, f - rhomax * cos(n * PI))
        } else {
            (rhomax * sin(n * PI), f)
        };
        Lambert {
            hem,
            raw_ref0: raw[0],
            n,
            f,
            rhomax,
            max_x,
            max_y,
            min_y,
        }
    }

    pub fn coords(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        map2(x, y, |x, y| {
            let x1 = x * self.max_x;
            let dy = self.f - fan_unit_y(y, self.hem, self.max_y, self.min_y);
            let rho = sign(self.n) * (x1 * x1 + dy * dy).sqrt();
            let lon = x1.atan2(dy) / self.n;
            let mut lat = 2.0 * (self.f / rho).powf(1.0 / self.n).atan() - PI / 2.0;
            if self.hem == AzimType::Bihem {
                lat *= sign(y);
            }
            (lon, lat)
        })
    }

    pub fn pos(&self, lon: &[f64], lat: &[f64]) -> (Vec<f64>, Vec<f64>) {
        map2(lon, lat, |lon, lat| {
            let l = if self.hem == AzimType::Bihem {
                lat.abs()
            } else {
                lat
            };
            let rho = self.f / (PI / 4.0 + l / 2.0).tan().powf(self.n);
            let th = self.n * lon;
            let x = rho * sin(th) / self.max_x;
            let y = self.f - rho * cos(th);
            (x, fan_map_y(y, lat, self.hem, self.max_y, self.min_y))
        })
    }

    pub fn vis_far(
        &self,
        x: &[f64],
        y: &[f64],
        lonlat: Option<(&[f64], &[f64])>,
        _ctx: &ProjCtx,
    ) -> (Vec<bool>, Option<Vec<bool>>) {
        if let Some((lon, lat)) = lonlat {
            let min_lat = (!self.hem.is_hemispheric()).then_some(self.raw_ref0);
            let (vis, far) = lonlat_vis(y, lon, lat, self.hem, min_lat);
            return (vis, Some(far));
        }
        let far_r = self.rhomax.powf(2.0);
        let (vis, far): (Vec<bool>, Vec<bool>) = x
            .iter()
            .zip(y)
            .map(|(&x, &y)| {
                let x1 = x * self.max_x;
                let dy = self.f - fan_unit_y(y, self.hem, self.max_y, self.min_y);
                let rho_s = x1 * x1 + dy * dy;
                fan_vis(false, rho_s > far_r, x1.abs().atan2(dy) > self.n * PI)
            })
            .unzip();
        (vis, Some(far))
    }

    pub fn ratio(&self) -> f64 {
        fan_ratio(self.hem, self.max_x, self.max_y, self.min_y)
    }
}
