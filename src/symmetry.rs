//! Exploiting projection symmetry to avoid redundant work on global maps.
//!
//! A symmetric projection only needs to be evaluated over the top-left half or
//! quadrant of a grid; the rest is filled in by mirroring. Mirrored values are
//! negated (not recomputed), exactly as the original implementation does, so
//! results stay bit-identical with it.

use rayon::prelude::*;

/// The symmetry a projection's coordinate functions have about the map center.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    None,
    /// Mirror-symmetric about the vertical axis only (`symx`).
    X,
    /// Mirror-symmetric about both axes (`sym4`).
    Quad,
    /// Mirror-symmetric about both axes, with latitude depending only on the
    /// row (`sym4lat`), so a single column of latitudes suffices.
    QuadLat,
}

impl Symmetry {
    /// The symmetry usable when this projection is evaluated on the output of
    /// a projection with symmetry `prev`.
    pub fn after(self, prev: Symmetry) -> Symmetry {
        use Symmetry::*;
        match (self, prev) {
            (QuadLat, QuadLat) => QuadLat,
            (QuadLat | Quad, Quad | QuadLat) => Quad,
            (QuadLat | Quad, _) => None,
            (s, _) => s,
        }
    }

    fn halves_x(self) -> bool {
        self != Symmetry::None
    }

    fn halves_y(self) -> bool {
        matches!(self, Symmetry::Quad | Symmetry::QuadLat)
    }
}

/// The shape of a full grid and of the sub-grid actually computed under a symmetry.
#[derive(Clone, Copy, Debug)]
pub struct SymGrid {
    pub sym: Symmetry,
    pub rows: usize,
    pub cols: usize,
    pub sub_rows: usize,
    pub sub_cols: usize,
}

impl SymGrid {
    pub fn new(sym: Symmetry, rows: usize, cols: usize) -> Self {
        let sub_cols = if sym.halves_x() {
            cols.div_ceil(2)
        } else {
            cols
        };
        let sub_rows = if sym.halves_y() {
            rows.div_ceil(2)
        } else {
            rows
        };
        SymGrid {
            sym,
            rows,
            cols,
            sub_rows,
            sub_cols,
        }
    }

    /// Extracts the sub-grids of a pair of full grids. Under `QuadLat` the
    /// second grid is taken from its first column and broadcast across rows.
    pub fn slice(&self, a: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.sub_rows * self.sub_cols;
        let mut sa = vec![0.0; n];
        let mut sb = vec![0.0; n];
        let lat_only = self.sym == Symmetry::QuadLat;
        sa.par_chunks_mut(self.sub_cols)
            .zip(sb.par_chunks_mut(self.sub_cols))
            .enumerate()
            .for_each(|(r, (ra, rb))| {
                let row = r * self.cols;
                ra.copy_from_slice(&a[row..row + self.sub_cols]);
                if lat_only {
                    rb.fill(b[row]);
                } else {
                    rb.copy_from_slice(&b[row..row + self.sub_cols]);
                }
            });
        (sa, sb)
    }

    fn source(&self, r: usize, c: usize) -> (usize, bool, bool) {
        let flip_r = self.sym.halves_y() && r >= self.sub_rows;
        let flip_c = self.sym.halves_x() && c >= self.sub_cols;
        let sr = if flip_r { self.rows - 1 - r } else { r };
        let sc = if flip_c { self.cols - 1 - c } else { c };
        (sr * self.sub_cols + sc, flip_r, flip_c)
    }

    /// Rebuilds full grids from sub-grids of an (x, y)-like pair: the first is
    /// negated across the vertical axis, the second across the horizontal axis.
    pub fn join(&self, a: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.rows * self.cols;
        let mut fa = vec![0.0; n];
        let mut fb = vec![0.0; n];
        fa.par_chunks_mut(self.cols)
            .zip(fb.par_chunks_mut(self.cols))
            .enumerate()
            .for_each(|(r, (ra, rb))| {
                for c in 0..self.cols {
                    let (i, flip_r, flip_c) = self.source(r, c);
                    ra[c] = if flip_c { -a[i] } else { a[i] };
                    rb[c] = if flip_r { -b[i] } else { b[i] };
                }
            });
        (fa, fb)
    }

    /// Rebuilds a full mask from a sub-grid mask by mirroring.
    pub fn join_mask(&self, m: &[bool]) -> Vec<bool> {
        let mut full = vec![false; self.rows * self.cols];
        full.par_chunks_mut(self.cols)
            .enumerate()
            .for_each(|(r, row)| {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = m[self.source(r, c).0];
                }
            });
        full
    }

    /// Evaluates `f` over the sub-grid of `(a, b)` and mirrors the result.
    pub fn eval<F>(&self, a: &[f64], b: &[f64], f: F) -> (Vec<f64>, Vec<f64>)
    where
        F: FnOnce(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
    {
        if self.sym == Symmetry::None {
            return f(a, b);
        }
        let (sa, sb) = self.slice(a, b);
        let (ra, rb) = f(&sa, &sb);
        self.join(&ra, &rb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downgrades_like_original() {
        use Symmetry::*;
        assert_eq!(QuadLat.after(QuadLat), QuadLat);
        assert_eq!(QuadLat.after(Quad), Quad);
        assert_eq!(QuadLat.after(X), None);
        assert_eq!(Quad.after(QuadLat), Quad);
        assert_eq!(Quad.after(X), None);
        assert_eq!(X.after(Quad), X);
    }

    #[test]
    fn join_mirrors_odd_grid() {
        let g = SymGrid::new(Symmetry::Quad, 3, 3);
        assert_eq!((g.sub_rows, g.sub_cols), (2, 2));
        let (a, b) = g.join(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(a, vec![1.0, 2.0, -1.0, 3.0, 4.0, -3.0, 1.0, 2.0, -1.0]);
        assert_eq!(b, vec![5.0, 6.0, 5.0, 7.0, 8.0, 7.0, -5.0, -6.0, -5.0]);
    }
}
