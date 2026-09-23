//! Padding the input map's edges so interpolation doesn't produce seams.

use crate::projection::Wrap;
use crate::raster::{Raster, Sample};

/// Pixels of padding added to each side.
pub const PAD: usize = 2;

/// Special handling of conic maps' edges.
#[derive(Clone, Copy, Debug)]
pub struct ConicEdges<'a> {
    /// Which pixels lie beyond the outer edge of the fan.
    pub far: &'a [bool],
    /// Whether the map is a bihemisphere, whose hemispheres wrap into each other.
    pub bihem: bool,
}

/// Pads `raster` (whose pixel centers lie at `xs`) horizontally by [`PAD`]
/// pixels on each side, returning the padded raster and pixel centers.
///
/// Rectangular maps wrap around. Other maps fill the pixels bordering the
/// visible area (`vis`) with data mirrored across the map, so that
/// interpolation near the edge draws on the other side of the globe.
pub fn pad<T: Sample>(
    wrap: Wrap,
    raster: &Raster<T>,
    xs: &[f64],
    vis: Option<&[bool]>,
    conic: Option<ConicEdges>,
) -> (Raster<T>, Vec<f64>) {
    let (w, h, c) = (raster.width, raster.height, raster.channels);
    let wp = w + 2 * PAD;

    let mut xs_p = Vec::with_capacity(wp);
    xs_p.extend(xs[w - PAD..].iter().map(|x| x - 2.0));
    xs_p.extend_from_slice(xs);
    xs_p.extend(xs[..PAD].iter().map(|x| x + 2.0));

    let mut out = Raster::new(wp, h, c);
    for r in 0..h {
        let src = &raster.data[r * w * c..(r + 1) * w * c];
        let dst = &mut out.data[r * wp * c..(r + 1) * wp * c];
        dst[PAD * c..(PAD + w) * c].copy_from_slice(src);
        if wrap == Wrap::Rect {
            dst[..PAD * c].copy_from_slice(&src[(w - PAD) * c..]);
            dst[(PAD + w) * c..].copy_from_slice(&src[..PAD * c]);
        }
    }

    if wrap != Wrap::Rect
        && let Some(vis) = vis
    {
        wrap_edges(
            &mut out,
            &pad_mask(vis, w, h),
            conic.map(|e| (pad_mask(e.far, w, h), e.bihem)),
        );
    }
    (out, xs_p)
}

fn pad_mask(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let wp = w + 2 * PAD;
    let mut out = vec![false; wp * h];
    for r in 0..h {
        out[r * wp + PAD..r * wp + PAD + w].copy_from_slice(&mask[r * w..(r + 1) * w]);
    }
    out
}

/// Masks of invisible pixels bordering the visible area on each side.
struct Borders {
    /// Visible pixel to the right.
    left: Vec<bool>,
    /// Visible pixel to the left.
    right: Vec<bool>,
    /// Visible pixel below.
    up: Vec<bool>,
    /// Visible pixel above.
    down: Vec<bool>,
}

fn wrap_edges<T: Sample>(data: &mut Raster<T>, vis: &[bool], conic: Option<(Vec<bool>, bool)>) {
    let (w, h) = (data.width, data.height);
    let at = |r: usize, c: usize| r * w + c;
    let prev = |i: usize, n: usize| (i + n - 1) % n;
    let next = |i: usize, n: usize| (i + 1) % n;

    let mut b = Borders {
        left: vec![false; w * h],
        right: vec![false; w * h],
        up: vec![false; w * h],
        down: vec![false; w * h],
    };
    for r in 0..h {
        for c in 0..w {
            let i = at(r, c);
            if !vis[i] {
                b.left[i] = vis[at(r, next(c, w))];
                b.right[i] = vis[at(r, prev(c, w))];
                b.up[i] = vis[at(next(r, h), c)];
                b.down[i] = vis[at(prev(r, h), c)];
            }
        }
    }

    if let Some((far, bihem)) = conic {
        if bihem {
            // Wrap the outer edges of each hemisphere into the other one.
            let flip = |r: usize| h - 1 - r;
            copy_masked(
                data,
                &[
                    (&b.right, &|r, c| (flip(r), prev(c, w))),
                    (&b.left, &|r, c| (flip(r), next(c, w))),
                    (&b.down, &|r, c| (flip(prev(r, h)), c)),
                    (&b.up, &|r, c| (flip(next(r, h)), c)),
                ],
            );
        }
        // Keep the regular wrapping below from affecting the outer edges.
        for m in [&mut b.left, &mut b.right, &mut b.up, &mut b.down] {
            m.iter_mut().zip(&far).for_each(|(m, &f)| *m &= !f);
        }
    }

    let mirror = |c: usize| w - 1 - c;
    copy_masked(
        data,
        &[
            (&b.down, &|r, c| (prev(r, h), mirror(c))),
            (&b.up, &|r, c| (next(r, h), mirror(c))),
            (&b.right, &|r, c| (r, mirror(prev(c, w)))),
            (&b.left, &|r, c| (r, mirror(next(c, w)))),
        ],
    );
}

type SourceFn<'a> = &'a dyn Fn(usize, usize) -> (usize, usize);

/// For each `(mask, source)` in order, sets masked pixels to the pixel at
/// `source(row, col)`. All sources are read before any pixel is written, so
/// later masks take precedence without seeing earlier writes.
fn copy_masked<T: Sample>(data: &mut Raster<T>, steps: &[(&Vec<bool>, SourceFn)]) {
    let w = data.width;
    let mut writes = Vec::new();
    for (mask, source) in steps {
        for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
            let (r, c) = source(i / w, i % w);
            writes.push((i, data.pixel(r, c).to_vec()));
        }
    }
    for (i, px) in writes {
        data.pixel_mut(i / w, i % w).copy_from_slice(&px);
    }
}
