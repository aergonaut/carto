//! Configuration, loaded from a TOML file.
//!
//! Option names follow the original projectionpasta config. Options that were
//! `None` in the original are simply omitted.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::projection::{AzimType, ProjCtx, Projection, Side};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub map: MapInfo,
    pub output: OutputOptions,
    pub procedure: Procedure,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// The maps to reproject and their projections.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MapInfo {
    /// Skip interactive setup and use the values below instead.
    pub skip_setup: bool,
    pub file_in: Option<PathBuf>,
    pub file_out: Option<PathBuf>,
    /// Projection, by name or by number in the projection list.
    pub proj_in: Option<Projection>,
    pub proj_out: Option<Projection>,
    /// Aspect: central longitude, central latitude, clockwise rotation from north (degrees).
    pub aspect_in: [f64; 3],
    pub aspect_out: [f64; 3],
    /// Layout of azimuthal and conic input maps.
    pub azim_type_in: AzimType,
    pub azim_type_out: AzimType,
    /// Layout for both maps; overrides the two above.
    pub azim_type: Option<AzimType>,
    /// Reference values in degrees (e.g. maximum latitude, standard parallels).
    pub ref_in: Option<RefValues>,
    pub ref_out: Option<RefValues>,
    /// Reference values for both maps; overrides the two above.
    #[serde(rename = "ref")]
    pub reference: Option<RefValues>,
}

impl MapInfo {
    pub fn params(&self) -> ProjectionParams {
        ProjectionParams {
            azim_type_in: self.azim_type_in,
            azim_type_out: self.azim_type_out,
            azim_type: self.azim_type,
            ref_in: self.ref_in.clone(),
            ref_out: self.ref_out.clone(),
            reference: self.reference.clone(),
        }
    }
}

/// Projection-specific parameters for each map.
#[derive(Clone, Debug, Default)]
pub struct ProjectionParams {
    pub azim_type_in: AzimType,
    pub azim_type_out: AzimType,
    pub azim_type: Option<AzimType>,
    pub ref_in: Option<RefValues>,
    pub ref_out: Option<RefValues>,
    pub reference: Option<RefValues>,
}

impl ProjectionParams {
    pub fn azim(&self, side: Side) -> AzimType {
        self.azim_type.unwrap_or(match side {
            Side::Input => self.azim_type_in,
            Side::Output => self.azim_type_out,
        })
    }

    pub fn ctx(&self, side: Side, procedure: &Procedure) -> ProjCtx {
        let reference = self.reference.as_ref().or(match side {
            Side::Input => self.ref_in.as_ref(),
            Side::Output => self.ref_out.as_ref(),
        });
        ProjCtx {
            side,
            azim: self.azim(side),
            reference: reference.map(|r| r.0.clone()),
            tolerance: procedure.tolerance,
            max_iter: procedure.max_iter,
        }
    }
}

/// One or more reference values, given as a number or an array.
#[derive(Clone, Debug, PartialEq)]
pub struct RefValues(pub Vec<f64>);

impl<'de> Deserialize<'de> for RefValues {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Spec {
            One(f64),
            Many(Vec<f64>),
        }
        Ok(RefValues(match Spec::deserialize(d)? {
            Spec::One(v) => vec![v],
            Spec::Many(v) => v,
        }))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputOptions {
    /// Force the output width/height ratio, overriding the projection's.
    pub force_ratio: Option<f64>,
    /// Scale the output ratio by the input map's ratio relative to its projection's.
    pub relative_ratio: bool,
    /// Output size: a width (height follows from the ratio) or `[width, height]`.
    pub force_scale: Option<Scale>,
    /// Treat the input map as truncated to a single world surface or hemisphere.
    pub truncate_in: bool,
    /// Truncate the output map to a single world surface or hemisphere.
    pub truncate_out: bool,
    /// Crop the input map to `[left, right, top, bottom]` pixel indices.
    pub crop_in: Option<[i64; 4]>,
    /// Crop the output map to `[left, right, top, bottom]` pixel indices of the full output.
    pub crop_out: Option<[i64; 4]>,
    /// The input map was cropped from a larger map:
    /// `[left, right, top, bottom, full width, full height]`.
    pub is_crop_in: Option<[i64; 6]>,
    /// Hide output outside `[min lon, max lon, min lat, max lat]` in the true aspect.
    pub crop_coords_true: Option<[f64; 4]>,
    /// As above, in the input map's own (normal aspect) coordinates.
    pub crop_coords_in: Option<[f64; 4]>,
    /// As above, in the output map's own (normal aspect) coordinates.
    pub crop_coords_out: Option<[f64; 4]>,
    /// Crop the output to the smallest box containing all visible areas.
    pub crop_vis: bool,
    /// Which coordinates to draw graticules in, if any.
    pub graticules: Option<GraticuleMode>,
    /// Longitudes to mark, or the interval between them (from the antimeridian), in degrees.
    pub grat_lon: GraticuleSpec,
    /// Latitudes to mark, or the interval between them (from the south pole), in degrees.
    pub grat_lat: GraticuleSpec,
    /// Graticule color as RGBA.
    pub grat_color: [u8; 4],
    /// Graticule line width in pixels.
    pub grat_width: u32,
}

impl Default for OutputOptions {
    fn default() -> Self {
        OutputOptions {
            force_ratio: None,
            relative_ratio: false,
            force_scale: None,
            truncate_in: true,
            truncate_out: true,
            crop_in: None,
            crop_out: None,
            is_crop_in: None,
            crop_coords_true: None,
            crop_coords_in: None,
            crop_coords_out: None,
            crop_vis: false,
            graticules: None,
            grat_lon: GraticuleSpec::Interval(30.0),
            grat_lat: GraticuleSpec::Interval(30.0),
            grat_color: [100, 100, 100, 50],
            grat_width: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Scale {
    Width(u32),
    Size([u32; 2]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraticuleMode {
    /// True coordinates, based on the output aspect.
    True,
    /// The input map's coordinates, treating it as normal aspect.
    In,
    /// The output map's coordinates, treating it as normal aspect.
    Out,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum GraticuleSpec {
    Interval(f64),
    List(Vec<f64>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Procedure {
    pub interp_type: InterpType,
    pub proj_direction: Direction,
    /// Use projection symmetry to speed calculations; appropriate only for global maps.
    pub use_sym: bool,
    /// Use extra steps to avoid seams at the edge of the original map.
    pub avoid_seam: bool,
    /// Maximum proportional error tolerated by iterative methods.
    pub tolerance: f64,
    pub max_iter: u32,
}

impl Default for Procedure {
    fn default() -> Self {
        Procedure {
            interp_type: InterpType::Nearest,
            proj_direction: Direction::Backward,
            use_sym: true,
            avoid_seam: true,
            tolerance: 1e-6,
            max_iter: 20,
        }
    }
}

/// How pixel values are interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpType {
    /// Copy the nearest input pixel directly (backward projection only).
    None,
    Nearest,
    Linear,
    /// Linear B-spline (backward projection only).
    Slinear,
    Cubic,
    /// Quintic B-spline (backward projection only).
    Quintic,
    /// Piecewise cubic Hermite (backward projection only).
    Pchip,
}

impl InterpType {
    pub fn name(self) -> &'static str {
        match self {
            InterpType::None => "none",
            InterpType::Nearest => "nearest",
            InterpType::Linear => "linear",
            InterpType::Slinear => "slinear",
            InterpType::Cubic => "cubic",
            InterpType::Quintic => "quintic",
            InterpType::Pchip => "pchip",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Interpolate on the input map for every output pixel.
    Backward,
    /// Interpolate on the output map from every input pixel.
    Forward,
    /// Backward, unless forward avoids an iterative method.
    AvoidIter,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let cfg: Config = toml::from_str(
            r#"
            [map]
            skip_setup = true
            file_in = "in.png"
            file_out = "out.png"
            proj_in = "Equirectangular"
            proj_out = 19
            aspect_out = [10, 20, 0]
            azim_type_out = "hem"
            ref_in = 45
            ref_out = [15, 45]

            [output]
            force_scale = [100, 50]
            graticules = "in"
            grat_lon = [0, 90]
            grat_color = [0, 0, 0, 77]

            [procedure]
            interp_type = "linear"
            proj_direction = "avoid_iter"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.map.proj_out, Some(Projection::Orthographic));
        assert_eq!(cfg.map.params().azim(Side::Output), AzimType::Hem);
        assert_eq!(cfg.map.ref_in, Some(RefValues(vec![45.0])));
        assert_eq!(cfg.output.force_scale, Some(Scale::Size([100, 50])));
        assert_eq!(cfg.output.grat_lon, GraticuleSpec::List(vec![0.0, 90.0]));
        assert_eq!(cfg.output.grat_lat, GraticuleSpec::Interval(30.0));
        assert_eq!(cfg.procedure.interp_type, InterpType::Linear);
        assert!(cfg.output.truncate_out);
    }
}
