//! Reprojection of world maps between arbitrary projections and aspects.
//!
//! A Rust port of [projectionpasta](https://github.com/hersfeldtn/projectionpasta)
//! by Mads de Silva and Nikolai Hersfeldt.
//!
//! The [`elev`] module is a port of [elevpasta](https://github.com/hersfeldtn/elevpasta),
//! by Nikolai Hersfeldt, which colors greyscale heightmaps.
//!
//! Results are intended to match the original bit-for-bit, so floating point
//! expressions deliberately mirror its NumPy code, down to operation order.

// Tests like `!(x > 1.0)` are deliberate: they treat NaN as passing, as the
// original's `np.where(x > 1, False, True)` does.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod config;
pub mod elev;
pub mod graticule;
pub mod interp;
pub mod marker;
pub mod math;
pub mod options;
pub mod pad;
pub mod projection;
pub mod raster;
pub mod reproject;
pub mod rotation;
pub mod setup;
pub mod symmetry;
