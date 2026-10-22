//! Configuration of the `elev` subcommand: the `[elev]` table of the config file.
//!
//! The original elevpasta is configured by editing its script; its per-layer
//! dictionaries become `[[elev.layers]]` tables with the same option names.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::config::{self, Schema};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct ElevConfig {
    pub elev: ElevOptions,
}

impl Schema for ElevConfig {
    const SECTIONS: &'static [&'static str] = &["elev"];
}

impl ElevConfig {
    pub fn load(path: &Path) -> Result<ElevConfig> {
        config::load(path)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElevOptions {
    /// Colored map to write.
    pub output: PathBuf,
    /// Layers, in the order they are applied: later layers paint over earlier ones.
    pub layers: Vec<Layer>,
}

impl Default for ElevOptions {
    fn default() -> Self {
        ElevOptions {
            output: PathBuf::from("color_map.png"),
            layers: Vec::new(),
        }
    }
}

impl ElevOptions {
    pub fn validate(&self) -> Result<()> {
        if self.layers.is_empty() {
            bail!("no layers: add at least one [[elev.layers]] table to the config");
        }
        for (i, layer) in self.layers.iter().enumerate() {
            layer
                .validate()
                .with_context(|| format!("elev layer {}", i + 1))?;
        }
        Ok(())
    }
}

/// One color scale applied over part of the map.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Layer {
    /// Greyscale elevation map used by this layer only.
    pub map: Option<PathBuf>,
    /// As `map`, but also kept as the base map for later layers that give neither.
    pub basemap: Option<PathBuf>,

    /// Elevation of the map's brightest and darkest pixels ...
    pub max: Option<f64>,
    pub min: Option<f64>,
    /// ... or of the greyscale mode's full white and full black.
    pub white: Option<f64>,
    pub black: Option<f64>,

    /// Elevations dividing the scale, in increasing order.
    pub steps: Vec<f64>,
    /// RGB colors. One more than `steps` for solid steps, as many as `steps` for a gradient.
    pub colors: Vec<[u8; 3]>,
    /// A single color for the whole layer, instead of `steps` and `colors`.
    pub onecolor: Option<[u8; 3]>,
    /// Interpolate colors between steps instead of coloring each interval solidly.
    pub gradient: bool,
    /// With `gradient`, raise elevations to this power before interpolating.
    pub gradexp: Option<f64>,

    /// Greyscale mask image: the layer applies where it is above half white.
    pub mask: Option<PathBuf>,
    /// Only apply above this elevation ...
    pub above: Option<f64>,
    /// ... and/or below this one.
    pub below: Option<f64>,
    /// Only apply where this layer's map is higher (`max`) or lower (`min`) than the base map.
    pub compare: Option<Compare>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Compare {
    Max,
    Min,
}

/// How grey values map to elevations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scale {
    /// The brightest and darkest pixels of the image have these elevations.
    Range { max: f64, min: f64 },
    /// Full white and full black for the image's bit depth have these elevations.
    Mode { white: f64, black: f64 },
}

impl Layer {
    /// The elevation scale, if the layer gives one.
    pub fn scale(&self) -> Result<Option<Scale>> {
        match (self.max, self.min, self.white, self.black) {
            (None, None, None, None) => Ok(None),
            (Some(max), Some(min), None, None) => Ok(Some(Scale::Range { max, min })),
            (None, None, Some(white), Some(black)) => Ok(Some(Scale::Mode { white, black })),
            _ => bail!("give either both `max` and `min`, or both `white` and `black`"),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if (self.map.is_some() || self.basemap.is_some()) && self.scale()?.is_none() {
            bail!("a layer with a map needs `max` and `min`, or `white` and `black`");
        }
        self.scale()?;
        if self.onecolor.is_some() {
            if !self.steps.is_empty() || !self.colors.is_empty() {
                bail!("`onecolor` can't be combined with `steps` or `colors`");
            }
        } else {
            if self.colors.is_empty() {
                bail!("give `onecolor`, or `steps` and `colors`");
            }
            let want = self.steps.len() + usize::from(!self.gradient);
            if self.colors.len() != want {
                bail!(
                    "{} steps need {want} colors for {} coloring, but {} were given",
                    self.steps.len(),
                    if self.gradient { "gradient" } else { "solid" },
                    self.colors.len()
                );
            }
            if !self.steps.windows(2).all(|w| w[0] < w[1]) {
                bail!("`steps` must be in strictly increasing order");
            }
        }
        if self.gradexp.is_some() && !self.gradient {
            bail!("`gradexp` requires `gradient = true`");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(toml: &str) -> Layer {
        toml::from_str(toml).unwrap()
    }

    #[test]
    fn parses_layers() {
        let cfg: ElevConfig = toml::from_str(
            r#"
            [elev]
            output = "o.png"

            [[elev.layers]]
            basemap = "e.png"
            max = 10000
            min = -10000
            steps = [0, 100]
            colors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

            [[elev.layers]]
            map = "i.png"
            white = 5000
            black = 0
            onecolor = [255, 255, 255]
            compare = "max"
            above = 0.5
            "#,
        )
        .unwrap();
        assert_eq!(cfg.elev.layers.len(), 2);
        cfg.elev.validate().unwrap();
        assert_eq!(
            cfg.elev.layers[1].scale().unwrap(),
            Some(Scale::Mode {
                white: 5000.0,
                black: 0.0
            })
        );
        assert_eq!(cfg.elev.layers[1].compare, Some(Compare::Max));
    }

    #[test]
    fn rejects_bad_layers() {
        let bad = [
            "onecolor = [0,0,0]\nmap = \"a\"\nmax = 1",
            "onecolor = [0,0,0]\nmap = \"a\"",
            "steps = [1]\ncolors = [[0,0,0]]",
            "steps = [1]\ncolors = [[0,0,0], [1,1,1]]\ngradient = true",
            "steps = [2, 1]\ncolors = [[0,0,0], [1,1,1], [2,2,2]]",
            "onecolor = [0,0,0]\ngradexp = 0.5",
            "onecolor = [0,0,0]\nsteps = [1]",
            "",
        ];
        for toml in bad {
            assert!(layer(toml).validate().is_err(), "accepted: {toml:?}");
        }
    }

    #[test]
    fn rejects_unknown_layer_options() {
        assert!(toml::from_str::<Layer>("colour = [0,0,0]").is_err());
    }
}
