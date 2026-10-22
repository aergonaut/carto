//! Elevation ("hypsometric") coloring of greyscale heightmaps.
//!
//! A port of [elevpasta](https://github.com/hersfeldtn/elevpasta) by Nikolai
//! Hersfeldt. Where the original's code is broken (it crashes on `white`/`black`
//! scaling and on `mask`), the documented behavior is implemented instead.

pub mod options;

use std::path::Path;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::raster::{AnyRaster, Raster, Sample};
use options::{Compare, ElevOptions, Layer, Scale};

/// Elevations of every pixel of a map.
struct Heights {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

impl Heights {
    /// Reads a greyscale image and scales it to elevations.
    ///
    /// With a [`Scale::Range`], the image's brightest and darkest pixels span
    /// `max - min`, as in the original, which does not shift by the image's own
    /// minimum: the result is only exact for images whose darkest pixel is 0.
    fn load(path: &Path, scale: Scale) -> Result<Heights> {
        let (width, height, samples, mode_max) = match AnyRaster::open(path)? {
            AnyRaster::U8(r) => grey(&r, u8::MAX as f64),
            AnyRaster::U16(r) => grey(&r, u16::MAX as f64),
            AnyRaster::F32(_) => Err(anyhow::anyhow!("expected a greyscale image")),
        }
        .with_context(|| format!("reading {}", path.display()))?;
        let values = match scale {
            Scale::Range { max, min } => {
                let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let span = hi - lo;
                samples
                    .iter()
                    .map(|g| g / span * (max - min) + min)
                    .collect()
            }
            Scale::Mode { white, black } => samples
                .iter()
                .map(|g| g / mode_max * (white - black) + black)
                .collect(),
        };
        Ok(Heights {
            width,
            height,
            values,
        })
    }

    fn same_size(&self, other: (usize, usize), what: &str) -> Result<()> {
        if (self.width, self.height) != other {
            bail!(
                "{what} is {}x{}, but the map is {}x{}",
                other.0,
                other.1,
                self.width,
                self.height
            );
        }
        Ok(())
    }
}

/// The first channel of an image (luma, for greyscale images with or without alpha).
fn grey<T: Sample>(r: &Raster<T>, mode_max: f64) -> Result<(usize, usize, Vec<f64>, f64)> {
    if r.channels > 2 {
        bail!(
            "expected a greyscale image, but it has {} channels",
            r.channels
        );
    }
    let samples = r.data.chunks(r.channels).map(|p| p[0].to_f64()).collect();
    Ok((r.width, r.height, samples, mode_max))
}

/// How a layer turns elevations into colors.
enum Fill<'a> {
    Solid([u8; 3]),
    Steps {
        steps: &'a [f64],
        colors: &'a [[u8; 3]],
    },
    Gradient {
        steps: Vec<f64>,
        colors: &'a [[u8; 3]],
        exp: Option<f64>,
    },
}

/// NumPy's `sign`: zero for zero.
fn sign(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.signum() }
}

fn power(x: f64, exp: f64) -> f64 {
    x.abs().powf(exp) * sign(x)
}

impl<'a> Fill<'a> {
    fn new(layer: &'a Layer) -> Fill<'a> {
        if let Some(color) = layer.onecolor {
            Fill::Solid(color)
        } else if layer.gradient {
            let steps = match layer.gradexp {
                Some(exp) => layer.steps.iter().map(|&s| power(s, exp)).collect(),
                None => layer.steps.clone(),
            };
            Fill::Gradient {
                steps,
                colors: &layer.colors,
                exp: layer.gradexp,
            }
        } else {
            Fill::Steps {
                steps: &layer.steps,
                colors: &layer.colors,
            }
        }
    }

    fn color(&self, elevation: f64) -> [u8; 3] {
        match self {
            Fill::Solid(color) => *color,
            Fill::Steps { steps, colors } => colors[steps.partition_point(|&s| elevation > s)],
            Fill::Gradient { steps, colors, exp } => {
                let d = exp.map_or(elevation, |e| power(elevation, e));
                // As in the original, later intervals overwrite earlier ones.
                let last = steps.len() - 1;
                let mut color = colors[0];
                for n in 0..=last {
                    if n == last {
                        if d > steps[n] {
                            color = colors[n];
                        }
                    } else if d > steps[n] && d <= steps[n + 1] {
                        let h = (d - steps[n]) / (steps[n + 1] - steps[n]);
                        let (c0, c1) = (colors[n], colors[n + 1]);
                        color = std::array::from_fn(|i| {
                            (c0[i] as f64 + h * (c1[i] as f64 - c0[i] as f64)) as u8
                        });
                    }
                }
                color
            }
        }
    }
}

/// Where a layer applies.
struct Mask<'a> {
    image: Option<Vec<bool>>,
    above: Option<f64>,
    below: Option<f64>,
    compare: Option<(Compare, &'a Heights)>,
}

impl Mask<'_> {
    fn applies(&self, i: usize, elevation: f64) -> bool {
        self.image.as_ref().is_none_or(|m| m[i])
            && self.above.is_none_or(|a| elevation > a)
            && self.below.is_none_or(|b| elevation < b)
            && self.compare.is_none_or(|(compare, base)| match compare {
                Compare::Max => elevation > base.values[i],
                Compare::Min => elevation < base.values[i],
            })
    }
}

/// Colors the layers' maps, returning an RGB image. Anything no layer covers is black.
pub fn render(options: &ElevOptions) -> Result<Raster<u8>> {
    options.validate()?;
    let mut base: Option<Heights> = None;
    let mut out: Option<Raster<u8>> = None;
    for (i, layer) in options.layers.iter().enumerate() {
        println!("  Layer {}/{}...", i + 1, options.layers.len());
        let scale = layer.scale()?;
        let load = |path: &Path| {
            let scale = scale.context("map needs a scale")?;
            Heights::load(path, scale)
        };
        let ctx = || format!("elev layer {}", i + 1);

        if let Some(path) = &layer.basemap {
            base = Some(load(path).with_context(ctx)?);
        }
        let own = layer
            .map
            .as_deref()
            .map(load)
            .transpose()
            .with_context(ctx)?;
        let elevations = own
            .as_ref()
            .or(base.as_ref())
            .context("no map: give `map` or `basemap`, or an earlier layer a `basemap`")
            .with_context(ctx)?;

        let out = out.get_or_insert_with(|| Raster::new(elevations.width, elevations.height, 3));
        elevations
            .same_size((out.width, out.height), "this layer's map")
            .with_context(ctx)?;

        let image = match &layer.mask {
            Some(path) => {
                let mask = Heights::load(
                    path,
                    Scale::Range {
                        max: 100.0,
                        min: 0.0,
                    },
                )
                .with_context(ctx)?;
                elevations
                    .same_size((mask.width, mask.height), "the mask")
                    .with_context(ctx)?;
                Some(mask.values.iter().map(|&v| v > 50.0).collect())
            }
            None => None,
        };
        let compare = match layer.compare {
            Some(compare) => Some((
                compare,
                base.as_ref()
                    .context("`compare` needs a base map from a `basemap`")
                    .with_context(ctx)?,
            )),
            None => None,
        };
        if let Some((_, b)) = compare {
            elevations
                .same_size((b.width, b.height), "the base map")
                .with_context(ctx)?;
        }
        let mask = Mask {
            image,
            above: layer.above,
            below: layer.below,
            compare,
        };
        let fill = Fill::new(layer);

        out.data
            .par_chunks_exact_mut(3)
            .zip(&elevations.values)
            .enumerate()
            .for_each(|(i, (pixel, &elevation))| {
                if mask.applies(i, elevation) {
                    pixel.copy_from_slice(&fill.color(elevation));
                }
            });
    }
    out.context("no layers")
}
