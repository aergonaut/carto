//! Checks `carto elev` output against images produced by the original elevpasta.
//!
//! Each directory in `tests/fixtures` named `elev_*` holds a `carto.toml` with
//! an `[elev]` section, the maps it refers to, and the original's
//! `expected.png`. `elev_masks` exercises `white`/`black` scaling and masks,
//! which crash in the original script; its expected image comes from the
//! script with those two bugs fixed.

use std::path::Path;

use carto::elev::{self, options::ElevConfig};
use carto::raster::AnyRaster;

fn check(name: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let mut options = ElevConfig::load(&dir.join("carto.toml")).unwrap().elev;
    // Layer paths are relative to the working directory, as on the command line.
    for layer in &mut options.layers {
        for path in [&mut layer.map, &mut layer.basemap, &mut layer.mask]
            .into_iter()
            .flatten()
        {
            *path = dir.join(&*path);
        }
    }
    let actual = elev::render(&options).unwrap();
    let AnyRaster::U8(expected) = AnyRaster::open(&dir.join("expected.png")).unwrap() else {
        panic!("expected 8-bit fixture");
    };
    assert_eq!(
        (actual.width, actual.height, actual.channels),
        (expected.width, expected.height, expected.channels)
    );
    let differing = actual
        .data
        .chunks(3)
        .zip(expected.data.chunks(3))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differing, 0, "{differing} pixels differ from the original");
}

#[test]
fn elev_layers() {
    check("elev_layers");
}

#[test]
fn elev_masks() {
    check("elev_masks");
}
