//! In-memory images with any channel count and sample type.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageBuffer};

/// A pixel channel value.
pub trait Sample: Copy + Default + PartialEq + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    /// Converts back from floating point like NumPy's `astype`: truncating
    /// toward zero, with NaN becoming 0 (out-of-range values saturate).
    fn from_f64(v: f64) -> Self;
}

impl Sample for u8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as u8
    }
}

impl Sample for u16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as u16
    }
}

impl Sample for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

/// A row-major image with interleaved channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Raster<T> {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<T>,
}

impl<T: Sample> Raster<T> {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Raster {
            width,
            height,
            channels,
            data: vec![T::default(); width * height * channels],
        }
    }

    pub fn pixel(&self, row: usize, col: usize) -> &[T] {
        let i = (row * self.width + col) * self.channels;
        &self.data[i..i + self.channels]
    }

    pub fn pixel_mut(&mut self, row: usize, col: usize) -> &mut [T] {
        let i = (row * self.width + col) * self.channels;
        &mut self.data[i..i + self.channels]
    }

    /// Copies out the rectangle `rows` x `cols`.
    pub fn crop(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Raster<T> {
        let mut out = Raster::new(cols.len(), rows.len(), self.channels);
        let c = self.channels;
        for (r_out, r) in rows.enumerate() {
            let src = (r * self.width + cols.start) * c;
            let dst = r_out * out.width * c;
            out.data[dst..dst + out.width * c]
                .copy_from_slice(&self.data[src..src + out.width * c]);
        }
        out
    }
}

/// An image of any supported sample type.
#[derive(Clone, Debug)]
pub enum AnyRaster {
    U8(Raster<u8>),
    U16(Raster<u16>),
    F32(Raster<f32>),
}

/// Applies a function generic over the sample type to an [`AnyRaster`].
#[macro_export]
macro_rules! with_raster {
    ($raster:expr, $r:ident => $body:expr) => {
        match $raster {
            $crate::raster::AnyRaster::U8($r) => $crate::raster::AnyRaster::U8($body),
            $crate::raster::AnyRaster::U16($r) => $crate::raster::AnyRaster::U16($body),
            $crate::raster::AnyRaster::F32($r) => $crate::raster::AnyRaster::F32($body),
        }
    };
}

fn from_buffer<P: image::Pixel>(img: ImageBuffer<P, Vec<P::Subpixel>>) -> Raster<P::Subpixel>
where
    P::Subpixel: Sample,
{
    Raster {
        width: img.width() as usize,
        height: img.height() as usize,
        channels: P::CHANNEL_COUNT as usize,
        data: img.into_raw(),
    }
}

impl AnyRaster {
    pub fn open(path: &Path) -> Result<AnyRaster> {
        let img = image::ImageReader::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .with_guessed_format()?;
        let mut decoder = img.into_decoder()?;
        // Allow arbitrarily large maps.
        image::ImageDecoder::set_limits(&mut decoder, image::Limits::no_limits())?;
        let img = DynamicImage::from_decoder(decoder)
            .with_context(|| format!("decoding {}", path.display()))?;
        Ok(match img {
            DynamicImage::ImageLuma8(b) => AnyRaster::U8(from_buffer(b)),
            DynamicImage::ImageLumaA8(b) => AnyRaster::U8(from_buffer(b)),
            DynamicImage::ImageRgb8(b) => AnyRaster::U8(from_buffer(b)),
            DynamicImage::ImageRgba8(b) => AnyRaster::U8(from_buffer(b)),
            DynamicImage::ImageLuma16(b) => AnyRaster::U16(from_buffer(b)),
            DynamicImage::ImageLumaA16(b) => AnyRaster::U16(from_buffer(b)),
            DynamicImage::ImageRgb16(b) => AnyRaster::U16(from_buffer(b)),
            DynamicImage::ImageRgba16(b) => AnyRaster::U16(from_buffer(b)),
            DynamicImage::ImageRgb32F(b) => AnyRaster::F32(from_buffer(b)),
            DynamicImage::ImageRgba32F(b) => AnyRaster::F32(from_buffer(b)),
            other => AnyRaster::U8(from_buffer(other.to_rgba8())),
        })
    }

    pub fn to_dynamic(&self) -> Result<DynamicImage> {
        fn buf<P: image::Pixel>(
            r: &Raster<P::Subpixel>,
        ) -> Result<ImageBuffer<P, Vec<P::Subpixel>>> {
            ImageBuffer::from_raw(r.width as u32, r.height as u32, r.data.clone())
                .context("image buffer size mismatch")
        }
        Ok(match self {
            AnyRaster::U8(r) => match r.channels {
                1 => DynamicImage::ImageLuma8(buf(r)?),
                2 => DynamicImage::ImageLumaA8(buf(r)?),
                3 => DynamicImage::ImageRgb8(buf(r)?),
                4 => DynamicImage::ImageRgba8(buf(r)?),
                n => bail!("unsupported channel count {n}"),
            },
            AnyRaster::U16(r) => match r.channels {
                1 => DynamicImage::ImageLuma16(buf(r)?),
                2 => DynamicImage::ImageLumaA16(buf(r)?),
                3 => DynamicImage::ImageRgb16(buf(r)?),
                4 => DynamicImage::ImageRgba16(buf(r)?),
                n => bail!("unsupported channel count {n}"),
            },
            AnyRaster::F32(r) => match r.channels {
                3 => DynamicImage::ImageRgb32F(buf(r)?),
                4 => DynamicImage::ImageRgba32F(buf(r)?),
                n => bail!("unsupported channel count {n}"),
            },
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.to_dynamic()?
            .save(path)
            .with_context(|| format!("saving {}", path.display()))
    }
}
