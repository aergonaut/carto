# carto

Reprojects world maps between arbitrary projections and aspects. A Rust port of
[projectionpasta](https://github.com/hersfeldtn/projectionpasta) 2.0.1 by Mads de
Silva and Nikolai Hersfeldt, producing pixel-identical output (see
[Parity](#parity-with-projectionpasta)) 10–20x faster: a 14000×7000 map
reprojects to an 8000×4000 orthographic in 1.7 s, versus about 30 s for the
original.

## Usage

```sh
cargo build --release

# Interactive setup, as in the original
carto

# From a config file (see carto.example.toml); carto.toml in the current
# directory is loaded automatically
carto --config my-map.toml

# Or straight from the command line
carto -i world.png -o globe.png --proj-in Equirectangular --proj-out Orthographic --aspect-out -30,20,0

carto list                                       # numbered list of projections
carto globe -i world.png -o globe.gif --frames 36  # spinning globe animation
```

Command-line arguments override the config file. Setup is interactive unless
the config sets `skip_setup = true` or the input is given with `-i`.

## Configuration

Options are the original's, in TOML, grouped into `[map]`, `[output]` and
`[procedure]` tables. [`carto.example.toml`](carto.example.toml) documents every
option. Options the original set to `None` are simply left out.

## Projections

All 27 projections from the original's list are supported, with the same
numbering, including azimuthal hemisphere/bihemisphere layouts, conics, and the
iterative projections (Equal Earth, Winkel Tripel, Natural Earth, Nicolosi,
Ortelius). The experimental projections the original left out of its list are
not ported.

## Parity with projectionpasta

The goal is identical output for identical options. `tests/parity.rs` checks a
set of fixtures generated with the original, and during development 131
configurations were compared pixel for pixel (every projection as input and as
output, rotated aspects, every layout, reference values, graticules, crops,
sizing options, and 8-bit RGBA/RGB/grayscale input), all identical.

Getting there meant mirroring NumPy and SciPy numerics exactly: expressions
keep the original's operation order, `sin`/`cos` are kept from being fused
into `sincos`, and the rotation matrix product and SciPy's linear
interpolation use fused multiply-adds as their compiled code does on macOS
(Accelerate BLAS, clang). On other platforms the original's own results can
differ in the last bit, so rare single-pixel differences are possible there.

Where exact parity isn't possible:

- **`slinear`, `cubic`, `quintic` interpolation**: SciPy 1.18 fits these
  splines with an iterative solver (tolerance 1e-5), so its output is itself
  approximate. carto solves exactly, so many pixels differ by ±1. Values that
  overshoot the sample range are clamped, where NumPy wraps them around (e.g.
  256 becomes 0).
- **Forward projection with `linear` or `cubic`** (`proj_direction =
  "forward"`): SciPy triangulates the scattered points with Qhull. A projected
  pixel grid has many equally valid triangulations, and carto's (via
  `delaunator`) picks different diagonals in some cells. So about a third of
  pixels differ, mostly by 1 or 2 but sometimes more. On points with a unique
  triangulation, both methods agree with SciPy to about 1e-11.

Backward `nearest`, `linear`, `pchip` and `none` interpolation, and forward
`nearest`, are exact.

### Differences from the original

- The config file is TOML. As a side effect, `graticules = "true"` and
  `interp_type = "none"` now work from a config file; the original's parser
  turned those into a boolean and `None`, which broke them.
- Explicit graticule lists (e.g. `grat_lon = [0, 90]`) are read as degrees; the
  original converted them to radians twice.
- Combining `crop_in` with `is_crop_in` works; it crashed in the original.
- Graticules can be drawn on grayscale images with alpha; they crashed in the
  original.
- `carto globe --retrograde` turns by `360 / frames` degrees per frame; the
  original turned retrograde globes by 1 degree per frame.

Other quirks of the original that affect output are kept, with comments where
they appear in the code.

## License

GPL-3.0-only, as a derivative of projectionpasta, which is licensed under the
GNU General Public License version 3. See [LICENSE](LICENSE).

## Development

`cargo test` runs the unit and parity tests. To add a parity fixture, run the
original on `tests/fixtures/input.png` and add a directory with the
`carto.toml` and the original's output as `expected.png`, then list it in
`tests/parity.rs`.
