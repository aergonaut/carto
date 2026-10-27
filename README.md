# carto

`carto` is a Rust mapping toolkit with two main commands: `carto project`
reprojects world maps between arbitrary projections and aspects, and
[`carto elev`](#carto-elev-color-heightmaps) colors greyscale heightmaps by
elevation.

`carto project` is a port of
[projectionpasta](https://github.com/hersfeldtn/projectionpasta) 2.0.1 by Mads de
Silva and Nikolai Hersfeldt. It produces pixel-identical output in most cases
(see [Parity with projectionpasta](#parity-with-projectionpasta)) and is 10–20x
faster: a 14000×7000 map reprojects to an 8000×4000 orthographic in 1.7 s,
versus about 30 s for the original. `carto elev` ports
[elevpasta](https://github.com/hersfeldtn/elevpasta) by Nikolai Hersfeldt.

## Usage

```sh
cargo build --release
```

### Reproject a map

```sh
# Interactive setup, as in the original (`carto p` is a shorthand)
carto project

# From a config file (see carto.example.toml); carto.toml in the current
# directory is loaded automatically
carto project --config my-map.toml

# Or straight from the command line
carto project -i world.png -o globe.png --proj-in Equirectangular --proj-out Orthographic --aspect-out -30,20,0
```

### Color a heightmap

```sh
# Color heightmaps by elevation, as set up in the [elev] config (`carto e` for short)
carto elev
carto elev --config my-map.toml -o color_map.png
```

### Other commands

```sh
carto list                                         # numbered list of projections
carto globe -i world.png -o globe.gif --frames 36  # spinning globe animation
```

Command-line arguments override the config file. Setup is interactive unless
the config sets `skip_setup = true` or the input is given with `-i`.

## Configuration

Options are the original's, in TOML, grouped into `[map]`, `[output]` and
`[procedure]` tables. [`carto.example.toml`](carto.example.toml) documents every
option. Options the original set to `None` are simply left out.

One config file can configure several subcommands. Each subcommand reads only
its own tables (`project` reads `[map]`, `[output]` and `[procedure]`; `elev`
reads `[elev]`) and ignores the others, so the options of different subcommands
never clash. Unknown options inside a subcommand's own tables, and tables no
subcommand reads, are errors.

## `carto project`: Reproject maps

### Projections

All 27 projections from the original's list are supported, with the same
numbering, including azimuthal hemisphere/bihemisphere layouts, conics, and the
iterative projections (Equal Earth, Winkel Tripel, Natural Earth, Nicolosi,
Ortelius). The experimental projections the original left out of its list are
not ported.

### Parity with projectionpasta

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

#### Differences from the original

- The config file is TOML. As a side effect, `graticules = "true"` and
  `interp_type = "none"` now work from a config file; the original's parser
  turned those into a boolean and `None`, which broke them.
- Explicit graticule lists (e.g. `grat_lon = [0, 90]`) are read as degrees; the
  original converted them to radians twice.
- Combining `crop_in` with `is_crop_in` works; it crashed in the original.
- Graticules can be drawn on grayscale images with alpha; they crashed in the
  original.
- `markers` (with `marker_color` and `marker_length`) draws reference Xs at
  given points; the original has no equivalent.
- `carto globe --retrograde` turns by `360 / frames` degrees per frame; the
  original turned retrograde globes by 1 degree per frame.

Other quirks of the original that affect output are kept, with comments where
they appear in the code.

## `carto elev`: Color heightmaps

`carto elev` (or `carto e`) colors greyscale heightmaps in layers, as elevpasta
does. Instead of editing a script, the layers are `[[elev.layers]]` tables in the
config file, with elevpasta's option names:

```toml
[elev]
output = "color_map.png"

[[elev.layers]]
basemap = "elev_map.png"      # kept as the base map for later layers
max = 10000                   # elevation of the brightest pixel ...
min = -10000                  # ... and of the darkest
steps = [-4000, -1000]
colors = [[113, 171, 216], [150, 201, 240], [216, 242, 254]]  # one more than steps

[[elev.layers]]               # no map: reuses the base map
steps = [0, 100, 900]
colors = [[172, 208, 165], [148, 191, 139], [239, 235, 192], [245, 244, 242]]
above = 0                     # only where elevation > 0
```

Each layer gives a map (`map` for this layer, `basemap` to also keep it for
later layers; neither reuses the base map), its scale (`max` and `min`, or
`white` and `black` for the full range of the image's bit depth), and a color
scale (`steps` and `colors`, or `onecolor`). `gradient = true` interpolates
between steps, which then need as many colors as steps; `gradexp` raises
elevations to a power first. A layer can be limited with `mask` (an image,
applied where above half white), `above`, `below` and `compare` (`"max"` or
`"min"`: only where the layer's map is higher or lower than the base map).
Later layers paint over earlier ones; anything no layer covers is black. See
[`carto.example.toml`](carto.example.toml) for every option.

Differences from elevpasta:

- Scaling with `white` and `black`, and `mask`, crash in the original script;
  they are implemented as documented. Everything else gives identical output
  (`tests/elev_parity.rs`).
- With `max` and `min`, the original scales by the image's range without
  subtracting its darkest value, so the brightest and darkest pixels get exactly
  `max` and `min` only if the darkest pixel is 0, as in most heightmaps. carto
  keeps this.
- Step and color counts, step order, and combinations such as `gradexp`
  without `gradient` are checked rather than silently misbehaving.
- Images must be greyscale (8 or 16 bit); an alpha channel is ignored.

## License

GPL-3.0-only, as a derivative of projectionpasta and elevpasta, which are both
licensed under the GNU General Public License version 3. See [LICENSE](LICENSE).

## Development

`cargo test` runs the unit and parity tests. To add a parity fixture, run the
original on `tests/fixtures/input.png` and add a directory with the
`carto.toml` and the original's output as `expected.png`, then list it in
`tests/parity.rs`. `carto elev` fixtures (`tests/fixtures/elev_*`, listed in
`tests/elev_parity.rs`) hold their own maps, an `[elev]` section in
`carto.toml`, and the output of elevpasta.py as `expected.png`.
