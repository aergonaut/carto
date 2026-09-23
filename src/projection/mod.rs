//! Map projections.
//!
//! Every projection maps normalized map coordinates `(x, y)`, each in
//! `[-1, 1]` with `(0, 0)` at the map center, to geographic coordinates
//! `(lon, lat)` in radians (`coords`), and back (`pos`). `vis` reports which
//! map positions lie on the visible, non-rectangular part of the map.
//!
//! Functions operate on whole arrays rather than single points because the
//! iterative methods test for convergence across the entire array at once;
//! this keeps results identical to the original implementation.

mod azimuthal;
mod conic;
mod cylindrical;
mod iterative;
mod pseudo;
mod special;

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use serde::Deserialize;

use crate::symmetry::Symmetry;

/// How azimuthal and conic maps are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AzimType {
    /// A single global map.
    Global,
    /// A single hemisphere.
    Hem,
    /// A pair of opposite hemispheres side by side.
    Bihem,
    /// Global if the projection allows it, bihemisphere otherwise.
    #[default]
    GlobalIf,
}

impl AzimType {
    pub fn is_hemispheric(self) -> bool {
        matches!(self, AzimType::Hem | AzimType::Bihem)
    }
}

/// Which map a projection is being evaluated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Input,
    Output,
}

impl Side {
    pub fn name(self) -> &'static str {
        match self {
            Side::Input => "input",
            Side::Output => "output",
        }
    }
}

/// Per-map parameters a projection needs to evaluate itself.
#[derive(Clone, Debug)]
pub struct ProjCtx {
    pub side: Side,
    /// Azimuthal/conic layout, before resolving what the projection supports.
    pub azim: AzimType,
    /// Projection reference values in degrees (e.g. standard parallels).
    pub reference: Option<Vec<f64>>,
    /// Tolerance for iterative methods (also used as the derivative step).
    pub tolerance: f64,
    pub max_iter: u32,
}

impl ProjCtx {
    /// Resolves the layout; `no_glob` marks projections that cannot show the whole globe.
    pub fn azim_type(&self, no_glob: bool) -> AzimType {
        match (self.azim, no_glob) {
            (AzimType::GlobalIf, true) => AzimType::Bihem,
            (AzimType::Global, true) => {
                static WARNED: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
                if !WARNED[self.side as usize].swap(true, Ordering::Relaxed) {
                    println!(
                        "    Warning: global type selected for {} projection not possible; using bihemisphere instead",
                        self.side.name()
                    );
                }
                AzimType::Bihem
            }
            (azim, _) => azim,
        }
    }

    pub fn with_azim(&self, azim: AzimType) -> ProjCtx {
        ProjCtx {
            azim,
            ..self.clone()
        }
    }

    /// Reference values in radians, falling back to `default` (in degrees).
    pub fn reference_rad(&self, default: &[f64]) -> Vec<f64> {
        self.reference
            .as_deref()
            .unwrap_or(default)
            .iter()
            .map(|d| d.to_radians())
            .collect()
    }
}

/// How the input map's edges are padded to avoid seams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    /// Rectangular maps: copy the opposite edge.
    Rect,
    /// Non-rectangular maps: mirror across the visible edge.
    XWrap,
    /// Conic maps, which additionally have inner and outer edges.
    ConicXWrap,
}

macro_rules! projections {
    ($($variant:ident => $name:literal, $info:literal;)*) => {
        /// All supported projections, in the order of the original numbered list.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Projection { $($variant),* }

        impl Projection {
            pub const ALL: &[Projection] = &[$(Projection::$variant),*];

            pub fn name(self) -> &'static str {
                match self { $(Projection::$variant => $name),* }
            }

            /// A one-line description of the projection's shape and properties.
            pub fn info(self) -> &'static str {
                match self { $(Projection::$variant => $info),* }
            }
        }
    };
}

projections! {
    Equirectangular => "Equirectangular", "2:1 rectangle (cylindrical); equidistant (vertical axis)";
    Mercator => "Mercator", "variable rectangle (cylindrical); conformal; truncated poles";
    GallStereographic => "Gall Stereographic", "1.301:1 rectangle (cylindrical)";
    MillerCylindrical => "Miller Cylindrical", "1.364:1 rectangle (cylindrical)";
    CylindricalEqualArea => "Cylindrical Equal-Area", "variable rectangle (cylindrical); equal-area";
    Sinusoidal => "Sinusoidal", "2:1 sinusoid (pseudocylindrical); equal-area, equidistant (horizontal axis)";
    Mollweide => "Mollweide", "2:1 ellipse (pseudocylindrical); equal-area";
    Hammer => "Hammer", "2:1 ellipse (pseudoazimuthal); equal-area";
    EckertIV => "Eckert IV", "2:1 oval (pseudocylindrical); equal-area";
    EqualEarth => "Equal Earth", "1.845:1 ovalish (pseudocylindrical); equal-area";
    WinkelTripel => "Winkel Tripel", "1.637:1 ovalish (pseudoazimuthal)";
    Robinson => "Robinson", "1.972:1 ovalish (pseudocylindrical)";
    WagnerVI => "Wagner VI", "2:1 ovalish (pseudocylindrical)";
    KavrayskiyVII => "Kavrayskiy VII", "1.732:1 ovalish (pseudocylindrical); stretched version of Wagner VI";
    NaturalEarth => "Natural Earth", "1.923:1 ovalish (pseudocylindrical)";
    Aitoff => "Aitoff", "2:1 ellipse (pseudoazimuthal)";
    AzimuthalEquidistant => "Azimuthal Equidistant", "1:1 circle (azimuthal); equidistant (from center)";
    LambertAzimuthal => "Lambert Azimuthal Equal-Area", "1:1 circle (azimuthal); equal-area, preserves 3d chord distances";
    Stereographic => "Stereographic", "1:1 circle (azimuthal); conformal, preserves circles; non-global only";
    Orthographic => "Orthographic", "1:1 circle (azimuthal); appearance of 3d globe; hemispheres only";
    EquidistantConic => "Equidistant Conic", "partial circular arc (conic); equidistant (from top edge)";
    AlbersConic => "Albers Equal-Area Conic", "partial circular arc (conic); equal-area";
    LambertConic => "Lambert Conformal Conic", "partial circular arc (conic); conformal";
    NicolosiGlobular => "Nicolosi Globular", "1:1 circle (polyconic); hemispheres only";
    OrteliusOval => "Ortelius Oval", "2:1 oval (pseudocylindrical)";
    Pseudostereographic => "Pseudostereographic", "2:1 ellipse (pseudoazimuthal)";
    Pseudoorthographic => "Pseudoorthographic", "2:1 ellipse (pseudoazimuthal)";
}

/// A prompt for projection reference values, used by interactive setup.
pub struct RefPrompt {
    pub prompt: &'static str,
    /// Number of values a custom entry requires.
    pub count: usize,
    /// Named presets and their values in degrees.
    pub presets: Vec<(&'static str, f64)>,
}

impl Projection {
    /// Looks up a projection by its position in [`Projection::ALL`].
    pub fn from_index(i: usize) -> Option<Projection> {
        Self::ALL.get(i).copied()
    }

    /// The heading of the group of projections this one starts, if any.
    pub fn section(self) -> Option<&'static str> {
        use Projection::*;
        match self {
            Equirectangular => Some("Cylindrical"),
            Sinusoidal => Some("Pseudocylindrical/azimuthal Equal-Area"),
            WinkelTripel => Some("Pseudocylindrical/azimuthal Compromise"),
            AzimuthalEquidistant => Some("Azimuthal"),
            EquidistantConic => Some("Conic"),
            NicolosiGlobular => Some("Miscellaneous / Special-Use"),
            _ => None,
        }
    }

    /// Whether the projection supports azimuthal layouts (hemispheres etc.).
    pub fn is_azimuthal(self) -> bool {
        use Projection::*;
        matches!(
            self,
            AzimuthalEquidistant
                | LambertAzimuthal
                | Stereographic
                | Orthographic
                | NicolosiGlobular
                | OrteliusOval
        )
    }

    pub fn is_conic(self) -> bool {
        use Projection::*;
        matches!(self, EquidistantConic | AlbersConic | LambertConic)
    }

    /// Whether the projection cannot show the entire globe in a single map.
    pub fn is_nonglobal(self) -> bool {
        use Projection::*;
        matches!(
            self,
            Mercator | Stereographic | Orthographic | LambertConic | NicolosiGlobular
        )
    }

    /// Whether `coords` requires an iterative method.
    pub fn has_iterative_coords(self) -> bool {
        use Projection::*;
        matches!(
            self,
            EqualEarth | WinkelTripel | NaturalEarth | NicolosiGlobular | OrteliusOval
        )
    }

    /// Whether `pos` requires an iterative method.
    pub fn has_iterative_pos(self) -> bool {
        matches!(self, Projection::Mollweide | Projection::EckertIV)
    }

    pub fn wrap(self) -> Wrap {
        use Projection::*;
        match self {
            Equirectangular | Mercator | GallStereographic | MillerCylindrical
            | CylindricalEqualArea => Wrap::Rect,
            EquidistantConic | AlbersConic | LambertConic => Wrap::ConicXWrap,
            _ => Wrap::XWrap,
        }
    }

    pub fn symmetry(self) -> Symmetry {
        use Projection::*;
        match self {
            Hammer | WinkelTripel | Aitoff | AzimuthalEquidistant | LambertAzimuthal
            | Stereographic | NicolosiGlobular | Pseudostereographic | Pseudoorthographic => {
                Symmetry::Quad
            }
            EquidistantConic | AlbersConic | LambertConic => Symmetry::X,
            _ => Symmetry::QuadLat,
        }
    }

    /// Whether `vis` makes use of known geographic coordinates.
    pub fn uses_lonlat_vis(self) -> bool {
        use Projection::*;
        matches!(self, EqualEarth | WinkelTripel | Robinson | NaturalEarth) || self.is_conic()
    }

    /// Whether every position on the map is visible (rectangular maps).
    pub fn has_default_vis(self) -> bool {
        use Projection::*;
        matches!(
            self,
            Equirectangular | GallStereographic | MillerCylindrical | CylindricalEqualArea
        )
    }

    /// The map's width/height ratio.
    pub fn ratio(self, ctx: &ProjCtx) -> f64 {
        use Projection::*;
        match self {
            Mercator => cylindrical::merc_ratio(ctx),
            GallStereographic => cylindrical::GALL_RATIO,
            MillerCylindrical => cylindrical::miller_ratio(),
            CylindricalEqualArea => cylindrical::cyleq_ratio(ctx),
            EqualEarth => pseudo::equal_earth_ratio(),
            WinkelTripel => pseudo::WINKEL_RATIO,
            Robinson => pseudo::ROBINSON_RATIO,
            KavrayskiyVII => 3f64.sqrt(),
            NaturalEarth => pseudo::natural_earth_ratio(),
            AzimuthalEquidistant | LambertAzimuthal | Stereographic => {
                azimuthal::azim_ratio(ctx, false)
            }
            Orthographic => azimuthal::azim_ratio(ctx, true),
            EquidistantConic => conic::Equidistant::new(ctx).ratio(),
            AlbersConic => conic::Albers::new(ctx).ratio(),
            LambertConic => conic::Lambert::new(ctx).ratio(),
            NicolosiGlobular => special::nicolosi_ratio(ctx),
            OrteliusOval => special::ortelius_ratio(ctx),
            _ => 2.0,
        }
    }

    /// Geographic coordinates for map positions.
    pub fn coords(self, x: &[f64], y: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
        use Projection::*;
        match self {
            Equirectangular => map2(x, y, cylindrical::equi_coords),
            Mercator => cylindrical::merc_coords(x, y, ctx),
            GallStereographic => map2(x, y, cylindrical::gall_coords),
            MillerCylindrical => map2(x, y, cylindrical::miller_coords),
            CylindricalEqualArea => map2(x, y, cylindrical::cyleq_coords),
            Sinusoidal => map2(x, y, pseudo::sin_coords),
            Mollweide => map2(x, y, pseudo::moll_coords),
            Hammer => map2(x, y, pseudo::hammer_coords),
            EckertIV => map2(x, y, pseudo::eckert4_coords),
            EqualEarth => pseudo::equal_earth_coords(x, y, ctx),
            WinkelTripel => pseudo::winkel_coords(x, y, ctx),
            Robinson => map2(x, y, pseudo::robinson_coords),
            WagnerVI | KavrayskiyVII => map2(x, y, pseudo::wagner_coords),
            NaturalEarth => pseudo::natural_earth_coords(x, y, ctx),
            Aitoff => map2(x, y, pseudo::aitoff_coords),
            AzimuthalEquidistant => azimuthal::azimeq_coords(x, y, ctx),
            LambertAzimuthal => azimuthal::lambert_coords(x, y, ctx),
            Stereographic => azimuthal::stereo_coords(x, y, ctx),
            Orthographic => azimuthal::ortho_coords(x, y, ctx),
            EquidistantConic => conic::Equidistant::new(ctx).coords(x, y),
            AlbersConic => conic::Albers::new(ctx).coords(x, y),
            LambertConic => conic::Lambert::new(ctx).coords(x, y),
            NicolosiGlobular => special::nicolosi_coords(x, y, ctx),
            OrteliusOval => special::ortelius_coords(x, y, ctx),
            Pseudostereographic => map2(x, y, special::psstereo_coords),
            Pseudoorthographic => map2(x, y, special::psortho_coords),
        }
    }

    /// Map positions for geographic coordinates.
    pub fn pos(self, lon: &[f64], lat: &[f64], ctx: &ProjCtx) -> (Vec<f64>, Vec<f64>) {
        use Projection::*;
        match self {
            Equirectangular => map2(lon, lat, cylindrical::equi_pos),
            Mercator => cylindrical::merc_pos(lon, lat, ctx),
            GallStereographic => map2(lon, lat, cylindrical::gall_pos),
            MillerCylindrical => map2(lon, lat, cylindrical::miller_pos),
            CylindricalEqualArea => map2(lon, lat, cylindrical::cyleq_pos),
            Sinusoidal => map2(lon, lat, pseudo::sin_pos),
            Mollweide => pseudo::moll_pos(lon, lat, ctx),
            Hammer => map2(lon, lat, pseudo::hammer_pos),
            EckertIV => pseudo::eckert4_pos(lon, lat, ctx),
            EqualEarth => map2(lon, lat, pseudo::equal_earth_pos),
            WinkelTripel => map2(lon, lat, pseudo::winkel_pos),
            Robinson => map2(lon, lat, pseudo::robinson_pos),
            WagnerVI | KavrayskiyVII => map2(lon, lat, pseudo::wagner_pos),
            NaturalEarth => map2(lon, lat, pseudo::natural_earth_pos),
            Aitoff => map2(lon, lat, pseudo::aitoff_pos),
            AzimuthalEquidistant => azimuthal::azimeq_pos(lon, lat, ctx),
            LambertAzimuthal => azimuthal::lambert_pos(lon, lat, ctx),
            Stereographic => azimuthal::stereo_pos(lon, lat, ctx),
            Orthographic => azimuthal::ortho_pos(lon, lat, ctx),
            EquidistantConic => conic::Equidistant::new(ctx).pos(lon, lat),
            AlbersConic => conic::Albers::new(ctx).pos(lon, lat),
            LambertConic => conic::Lambert::new(ctx).pos(lon, lat),
            NicolosiGlobular => special::nicolosi_pos(lon, lat, ctx),
            OrteliusOval => special::ortelius_pos(lon, lat, ctx),
            Pseudostereographic => map2(lon, lat, special::psstereo_pos),
            Pseudoorthographic => map2(lon, lat, special::psortho_pos),
        }
    }

    /// Which map positions are visible. When geographic coordinates for the
    /// positions are known they may be supplied to refine the test.
    pub fn vis(
        self,
        x: &[f64],
        y: &[f64],
        lonlat: Option<(&[f64], &[f64])>,
        ctx: &ProjCtx,
    ) -> Vec<bool> {
        self.vis_far(x, y, lonlat, ctx).0
    }

    /// Like [`Projection::vis`], additionally returning for conic maps which
    /// positions lie beyond the map's outer edge.
    pub fn vis_far(
        self,
        x: &[f64],
        y: &[f64],
        lonlat: Option<(&[f64], &[f64])>,
        ctx: &ProjCtx,
    ) -> (Vec<bool>, Option<Vec<bool>>) {
        use Projection::*;
        let vis = match self {
            Equirectangular | GallStereographic | MillerCylindrical | CylindricalEqualArea => {
                vec![true; x.len()]
            }
            Mercator => mask2(x, y, cylindrical::merc_vis),
            Sinusoidal => mask2(x, y, pseudo::sin_vis),
            Mollweide | Hammer | Aitoff | Pseudostereographic | Pseudoorthographic => {
                mask2(x, y, circ_vis)
            }
            EckertIV => mask2(x, y, pill_vis),
            EqualEarth => match lonlat {
                Some((lon, lat)) => coords_vis(lon, lat),
                None => edge_vis(&pseudo::equal_earth_coords(x, y, ctx).0),
            },
            WinkelTripel => match lonlat {
                Some((lon, lat)) => coords_vis(lon, lat),
                None => mask2(x, y, pseudo::winkel_vis),
            },
            Robinson => match lonlat {
                Some((lon, lat)) => coords_vis(lon, lat),
                None => mask2(x, y, pseudo::robinson_vis),
            },
            WagnerVI | KavrayskiyVII => mask2(x, y, pseudo::wagner_vis),
            NaturalEarth => match lonlat {
                Some((lon, lat)) => coords_vis(lon, lat),
                None => edge_vis(&pseudo::natural_earth_coords(x, y, ctx).0),
            },
            AzimuthalEquidistant | LambertAzimuthal | Stereographic => {
                azimuthal::azim_vis(x, y, ctx, false)
            }
            Orthographic => azimuthal::azim_vis(x, y, ctx, true),
            EquidistantConic => return conic::Equidistant::new(ctx).vis_far(x, y, lonlat, ctx),
            AlbersConic => return conic::Albers::new(ctx).vis_far(x, y, lonlat, ctx),
            LambertConic => return conic::Lambert::new(ctx).vis_far(x, y, lonlat, ctx),
            NicolosiGlobular => special::nicolosi_vis(x, y, ctx),
            OrteliusOval => special::ortelius_vis(x, y, ctx),
        };
        (vis, None)
    }

    /// Prompt for the projection's reference values, if it takes any.
    pub fn ref_prompt(self) -> Option<RefPrompt> {
        use Projection::*;
        let parallels = || RefPrompt {
            prompt: "Select reference latitudes of best shape (-90-90 degrees from equator, mean must be positive)",
            count: 2,
            presets: vec![],
        };
        match self {
            Mercator => Some(RefPrompt {
                prompt: "Select maximum latitude to truncate map (0-90 degrees from equator)",
                count: 1,
                presets: vec![("85.05: forms a 1:1 square", cylindrical::merc_default_ref())],
            }),
            CylindricalEqualArea => Some(RefPrompt {
                prompt: "Select reference latitude of best shape (0-90 degrees from equator)",
                count: 1,
                presets: vec![
                    ("0: Lambert, aspect ratio 3.141:1", 0.0),
                    ("30: Behrmann, 2.356:1", 30.0),
                    (
                        "37.071: Smyth/Craster, 2:1",
                        (2.0 / std::f64::consts::PI).sqrt().acos().to_degrees(),
                    ),
                    ("37.5: Hobo-Dyer, 1.977:1", 37.5),
                    ("45: Gall-Peters, 1.571:1", 45.0),
                    ("50: Balthasart, 1.298:1", 50.0),
                    (
                        "55.654: Tobler, 1:1",
                        (1.0 / std::f64::consts::PI).sqrt().acos().to_degrees(),
                    ),
                ],
            }),
            EquidistantConic | AlbersConic | LambertConic => Some(parallels()),
            _ => None,
        }
    }

    /// Prompt for extra reference values that apply only to global layouts.
    /// These come before any values from [`Projection::ref_prompt`].
    pub fn global_ref_prompt(self) -> Option<RefPrompt> {
        match self {
            Projection::Stereographic => Some(RefPrompt {
                prompt: "Select maximum latitude range to truncate map (0-180 degrees from center)",
                count: 1,
                presets: vec![],
            }),
            Projection::LambertConic => Some(RefPrompt {
                prompt: "Select minimum latitude to truncate map (-90-90 degrees from equator)",
                count: 1,
                presets: vec![],
            }),
            _ => None,
        }
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Projection {
    type Err = String;

    /// Parses a projection name (case-insensitive) or its index in the list.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if let Ok(i) = s.parse::<usize>() {
            return Projection::from_index(i)
                .ok_or_else(|| format!("no projection with index {i}"));
        }
        Projection::ALL
            .iter()
            .copied()
            .find(|p| p.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| format!("unknown projection \"{s}\""))
    }
}

impl<'de> Deserialize<'de> for Projection {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Spec {
            Index(usize),
            Name(String),
        }
        match Spec::deserialize(d)? {
            Spec::Index(i) => Projection::from_index(i),
            Spec::Name(s) => s.parse().ok(),
        }
        .ok_or_else(|| serde::de::Error::custom("invalid projection name or index"))
    }
}

/// Applies a pointwise function across paired arrays in parallel.
pub(crate) fn map2<F>(a: &[f64], b: &[f64], f: F) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(f64, f64) -> (f64, f64) + Sync,
{
    let mut oa = vec![0.0; a.len()];
    let mut ob = vec![0.0; a.len()];
    oa.par_iter_mut()
        .zip(ob.par_iter_mut())
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|((oa, ob), (&a, &b))| (*oa, *ob) = f(a, b));
    (oa, ob)
}

/// Applies a pointwise predicate across paired arrays in parallel.
pub(crate) fn mask2<F>(a: &[f64], b: &[f64], f: F) -> Vec<bool>
where
    F: Fn(f64, f64) -> bool + Sync,
{
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&a, &b)| f(a, b))
        .collect()
}

/// Circular or elliptical maps.
pub(crate) fn circ_vis(x: f64, y: f64) -> bool {
    !((x * x + y * y).sqrt() > 1.0)
}

/// Two semicircles flanking a square (at a 2:1 aspect ratio).
pub(crate) fn pill_vis(x: f64, y: f64) -> bool {
    if x.abs() > 0.5 {
        circ_vis(1.0 - (x * 2.0).abs(), y)
    } else {
        true
    }
}

/// Visible where the geographic coordinates are in range.
pub(crate) fn coords_vis(lon: &[f64], lat: &[f64]) -> Vec<bool> {
    mask2(lon, lat, |lon, lat| {
        !(lon.abs() > std::f64::consts::PI || lat.abs() > std::f64::consts::PI / 2.0)
    })
}

/// Visible where the computed longitude is in range.
fn edge_vis(lon: &[f64]) -> Vec<bool> {
    lon.par_iter()
        .map(|l| !(l.abs() > std::f64::consts::PI))
        .collect()
}
