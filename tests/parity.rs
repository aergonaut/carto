//! Checks output against images produced by the original Python projectionpasta.
//!
//! Each directory in `tests/fixtures` holds a `carto.toml` and the original's
//! `expected.png` for it. The input is the directory's own `input.png` if
//! present, otherwise `tests/fixtures/input.png`.

use std::path::Path;

use carto::options::Config;
use carto::raster::AnyRaster;
use carto::reproject::{Job, reproject};
use carto::with_raster;

fn check(name: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let dir = root.join(name);
    let config = Config::load(&dir.join("carto.toml")).unwrap();
    let input = [dir.join("input.png"), root.join("input.png")]
        .into_iter()
        .find(|p| p.exists())
        .unwrap();
    let job = Job {
        proj_in: config.map.proj_in.unwrap(),
        proj_out: config.map.proj_out.unwrap(),
        aspect_in: config.map.aspect_in,
        aspect_out: config.map.aspect_out,
        params: config.map.params(),
        output: config.output,
        procedure: config.procedure,
    };
    let input = AnyRaster::open(&input).unwrap();
    let AnyRaster::U8(actual) = with_raster!(&input, r => reproject(r, &job).unwrap()) else {
        panic!("expected 8-bit output");
    };
    let AnyRaster::U8(expected) = AnyRaster::open(&dir.join("expected.png")).unwrap() else {
        panic!("expected 8-bit fixture");
    };
    assert_eq!(
        (actual.width, actual.height),
        (expected.width, expected.height)
    );
    let differing = actual
        .data
        .chunks(actual.channels)
        .zip(expected.data.chunks(expected.channels))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differing, 0, "{differing} pixels differ from the original");
}

macro_rules! parity_tests {
    ($($name:ident),* $(,)?) => {
        $(#[test] fn $name() { check(stringify!($name)); })*
    };
}

parity_tests! {
    orthographic_hem,
    orthographic_rotated_graticules,
    winkel_tripel_linear,
    albers_bihem,
    nicolosi_crop_vis,
    mollweide,
    mollweide_to_lcc,
    pchip_hammer,
}
