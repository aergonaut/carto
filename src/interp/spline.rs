//! Spline interpolation on a regular grid (SciPy `interpn` spline methods).

use anyhow::{Result, bail};
use rayon::prelude::*;

use super::{GridSource, find_interval};
use crate::raster::{Raster, Sample};

/// The kind of spline to interpolate with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Tensor-product interpolating B-spline of the given odd degree.
    BSpline(usize),
    /// Piecewise cubic Hermite interpolation, applied one axis at a time.
    Pchip,
}

/// The highest supported B-spline degree.
const MAX_DEGREE: usize = 5;

/// Samples `src` at each point `(px[i], py[i])` with a spline of the given
/// kind, extrapolating outside the grid, and writes pixel `i` of `out`.
///
/// B-splines match SciPy's `make_ndbspl` interpolant (not-a-knot boundary
/// conditions); PCHIP matches `RegularGridInterpolator`'s recursive
/// evaluation (along x first, then along y).
pub fn sample<T: Sample>(
    src: &GridSource<T>,
    px: &[f64],
    py: &[f64],
    kind: Kind,
    out: &mut Raster<T>,
) -> Result<()> {
    // SciPy treats PCHIP as degree 3 when checking the grid size.
    let k = match kind {
        Kind::BSpline(k @ (1 | 3 | 5)) => k,
        Kind::BSpline(k) => bail!("unsupported B-spline degree {k}"),
        Kind::Pchip => 3,
    };
    if src.xs.len() <= k || src.ys.len() <= k {
        bail!(
            "{kind:?} interpolation requires at least {} pixels in each dimension",
            k + 1
        );
    }
    match kind {
        Kind::BSpline(k) => sample_bspline(src, px, py, k, out),
        Kind::Pchip => sample_pchip(src, px, py, out),
    }
    Ok(())
}

/// Fits and evaluates the interpolating tensor-product B-spline, one channel
/// at a time.
fn sample_bspline<T: Sample>(
    src: &GridSource<T>,
    px: &[f64],
    py: &[f64],
    k: usize,
    out: &mut Raster<T>,
) {
    let xk = Knots::not_a_knot(src.xs, k);
    let yk = Knots::not_a_knot(&src.ys_ascending(), k);
    let (nx, ny) = (src.xs.len(), src.ys.len());
    let (x_lu, y_lu) = (xk.collocation(src.xs), yk.collocation(&src.ys_ascending()));

    for ch in 0..out.channels {
        // The collocation system is (Ay ⊗ Ax) c = v; being a Kronecker
        // product, it separates into 1-D solves along each row, then each
        // column. `coef` holds rows in ascending-y order.
        let mut coef = vec![0.0; nx * ny];
        coef.par_chunks_mut(nx).enumerate().for_each(|(iy, row)| {
            for (ix, c) in row.iter_mut().enumerate() {
                *c = src.value(iy, ix, ch).to_f64();
            }
            x_lu.solve(row);
        });
        y_lu.solve_rows(&mut coef, nx);

        out.data
            .par_chunks_mut(out.channels)
            .enumerate()
            .for_each(|(i, pixel)| {
                let (x, y) = (px[i], py[i]);
                pixel[ch] = T::from_f64(if x.is_nan() || y.is_nan() {
                    f64::NAN
                } else {
                    let (lx, bx) = xk.basis(x);
                    let (ly, by) = yk.basis(y);
                    // Nonzero basis functions are indices l - k ..= l.
                    let (x0, y0) = (lx - k, ly - k);
                    by[..=k]
                        .iter()
                        .enumerate()
                        .map(|(a, wy)| {
                            let row = &coef[(y0 + a) * nx + x0..][..=k];
                            wy * row.iter().zip(&bx[..=k]).map(|(c, wx)| c * wx).sum::<f64>()
                        })
                        .sum()
                });
            });
    }
}

/// The knot vector of a B-spline of degree `k`.
struct Knots {
    t: Vec<f64>,
    k: usize,
}

impl Knots {
    /// Not-a-knot knots for interpolating at `x` with odd degree `k`, as in
    /// SciPy's `_not_a_knot`: the ends repeated `k + 1` times, with the
    /// interior data points as knots except for `(k - 1) / 2` at each end.
    fn not_a_knot(x: &[f64], k: usize) -> Self {
        let (first, last) = (x[0], x[x.len() - 1]);
        let k2 = k.div_ceil(2);
        let mut t = vec![first; k + 1];
        t.extend_from_slice(&x[k2..x.len() - k2]);
        t.extend(std::iter::repeat_n(last, k + 1));
        Knots { t, k }
    }

    /// The number of basis functions (and coefficients).
    fn len(&self) -> usize {
        self.t.len() - self.k - 1
    }

    /// Finds `l` with `t[l] <= v < t[l + 1]` within the base interval
    /// `t[k]..t[n]`, using the end intervals for points outside it (SciPy's
    /// `find_interval` with extrapolation), and evaluates the `k + 1` basis
    /// functions nonzero there, `B[l - k] ..= B[l]`, by de Boor's recursion.
    /// Outside the base interval this extends the end polynomial pieces.
    fn basis(&self, v: f64) -> (usize, [f64; MAX_DEGREE + 1]) {
        let (t, k, n) = (&self.t, self.k, self.len());
        let l = if v < t[k] {
            k
        } else if v >= t[n] {
            n - 1
        } else {
            k + t[k + 1..=n].partition_point(|&knot| knot <= v)
        };
        let mut h = [0.0; MAX_DEGREE + 1];
        h[0] = 1.0;
        for j in 1..=k {
            let prev = h;
            h[0] = 0.0;
            for m in 1..=j {
                let (xa, xb) = (t[l + m - j], t[l + m]);
                if xa == xb {
                    h[m] = 0.0;
                    continue;
                }
                let w = prev[m - 1] / (xb - xa);
                h[m - 1] += w * (xb - v);
                h[m] = w * (v - xa);
            }
        }
        (l, h)
    }

    /// The LU factorization of the collocation matrix `A[i][j] = B[j](x[i])`.
    fn collocation(&self, x: &[f64]) -> BandLu {
        let mut a = BandLu::zeros(x.len(), self.k);
        for (i, &xi) in x.iter().enumerate() {
            let (l, b) = self.basis(xi);
            for (m, &v) in b[..=self.k].iter().enumerate() {
                if v != 0.0 {
                    *a.at(i, l - self.k + m) = v;
                }
            }
        }
        a.factor();
        a
    }
}

/// A square band matrix with `w` diagonals on each side of the main one,
/// factored in place into `L U` (unit lower `L`).
///
/// B-spline collocation matrices are totally positive, so Gaussian
/// elimination without pivoting is stable and keeps the band structure.
struct BandLu {
    n: usize,
    w: usize,
    a: Vec<f64>,
}

impl BandLu {
    fn zeros(n: usize, w: usize) -> Self {
        BandLu {
            n,
            w,
            a: vec![0.0; n * (2 * w + 1)],
        }
    }

    fn at(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.a[i * (2 * self.w + 1) + self.w + j - i]
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.a[i * (2 * self.w + 1) + self.w + j - i]
    }

    /// The column range of row `i` below (`..i`) or above (`i + 1..`) the
    /// diagonal within the band.
    fn lower(&self, i: usize) -> std::ops::Range<usize> {
        i.saturating_sub(self.w)..i
    }

    fn upper(&self, i: usize) -> std::ops::Range<usize> {
        i + 1..(i + self.w + 1).min(self.n)
    }

    fn factor(&mut self) {
        for p in 0..self.n {
            let pivot = self.get(p, p);
            for i in self.upper(p) {
                let f = self.get(i, p) / pivot;
                *self.at(i, p) = f;
                for j in self.upper(p) {
                    let u = self.get(p, j);
                    *self.at(i, j) -= f * u;
                }
            }
        }
    }

    /// Solves `A x = b` in place.
    fn solve(&self, b: &mut [f64]) {
        for i in 0..self.n {
            b[i] -= self.lower(i).map(|j| self.get(i, j) * b[j]).sum::<f64>();
        }
        for i in (0..self.n).rev() {
            b[i] =
                (b[i] - self.upper(i).map(|j| self.get(i, j) * b[j]).sum::<f64>()) / self.get(i, i);
        }
    }

    /// Solves `A X = B` in place, where row `i` of `A` pairs with row `i` of
    /// the row-major `n × width` matrix `b`. Substitution runs row by row,
    /// with each row's update split across threads.
    fn solve_rows(&self, b: &mut [f64], width: usize) {
        const CHUNK: usize = 1024;
        for i in 0..self.n {
            let (done, rest) = b.split_at_mut(i * width);
            let lower: Vec<_> = self
                .lower(i)
                .map(|j| (self.get(i, j), &done[j * width..][..width]))
                .collect();
            rest[..width]
                .par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(c, seg)| {
                    for (f, row) in &lower {
                        for (s, r) in seg.iter_mut().zip(&row[c * CHUNK..]) {
                            *s -= f * r;
                        }
                    }
                });
        }
        for i in (0..self.n).rev() {
            let (head, done) = b.split_at_mut((i + 1) * width);
            let upper: Vec<_> = self
                .upper(i)
                .map(|j| (self.get(i, j), &done[(j - i - 1) * width..][..width]))
                .collect();
            let d = self.get(i, i);
            head[i * width..]
                .par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(c, seg)| {
                    for (f, row) in &upper {
                        for (s, r) in seg.iter_mut().zip(&row[c * CHUNK..]) {
                            *s -= f * r;
                        }
                    }
                    seg.iter_mut().for_each(|s| *s /= d);
                });
        }
    }
}

/// Evaluates SciPy's recursive PCHIP interpolation: a PCHIP along x through
/// each row, then a PCHIP along y through those row values. PCHIP is local,
/// so only the rows and columns within one node of the point's interval
/// matter.
fn sample_pchip<T: Sample>(src: &GridSource<T>, px: &[f64], py: &[f64], out: &mut Raster<T>) {
    let ys = src.ys_ascending();
    out.data
        .par_chunks_mut(out.channels)
        .enumerate()
        .for_each(|(i, pixel)| {
            let found = find_interval(&ys, py[i]).zip(find_interval(src.xs, px[i]));
            for (ch, v) in pixel.iter_mut().enumerate() {
                *v = T::from_f64(match found {
                    Some(((iy, _), (ix, _))) => pchip(&ys, iy, py[i], |r| {
                        pchip(src.xs, ix, px[i], |c| src.value(r, c, ch).to_f64())
                    }),
                    None => f64::NAN,
                });
            }
        });
}

/// Evaluates at `v` the PCHIP interpolant of the values `f(j)` at nodes
/// `x[j]`, where `v` lies in (or extrapolates from) interval `i`.
fn pchip(x: &[f64], i: usize, v: f64, f: impl Fn(usize) -> f64) -> f64 {
    let n = x.len();
    // Values at nodes lo..=hi, which suffice for the derivatives at i, i + 1.
    let lo = i.saturating_sub(1);
    let hi = (i + 2).min(n - 1);
    let mut y = [0.0; 4];
    for j in lo..=hi {
        y[j - lo] = f(j);
    }
    let h = |j: usize| x[j + 1] - x[j];
    let m = |j: usize| (y[j + 1 - lo] - y[j - lo]) / h(j);
    let d = |j: usize| {
        if j == 0 {
            pchip_end(h(0), h(1), m(0), m(1))
        } else if j == n - 1 {
            pchip_end(h(n - 2), h(n - 3), m(n - 2), m(n - 3))
        } else {
            let (m0, m1) = (m(j - 1), m(j));
            if sign(m0) != sign(m1) || m0 == 0.0 || m1 == 0.0 {
                0.0
            } else {
                // Weighted harmonic mean of the neighboring slopes.
                let (h0, h1) = (h(j - 1), h(j));
                let (w1, w2) = (2.0 * h1 + h0, h1 + 2.0 * h0);
                1.0 / ((w1 / m0 + w2 / m1) / (w1 + w2))
            }
        }
    };
    // The Hermite cubic in PPoly form, as SciPy's CubicHermiteSpline builds
    // and evaluates it.
    let (dx, slope, d0, d1) = (h(i), m(i), d(i), d(i + 1));
    let t = (d0 + d1 - 2.0 * slope) / dx;
    let c = [y[i - lo], d0, (slope - d0) / dx - t, t / dx];
    let s = v - x[i];
    let mut z = 1.0;
    let mut res = 0.0;
    for ck in c {
        res += ck * z;
        z *= s;
    }
    res
}

/// SciPy's one-sided three-point PCHIP end derivative, limited to preserve
/// shape; `h0`, `m0` belong to the end interval and `h1`, `m1` to the next.
fn pchip_end(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign(d) != sign(m0) {
        0.0
    } else if sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs() {
        3.0 * m0
    } else {
        d
    }
}

/// NumPy's `sign`, which is zero at zero (unlike `f64::signum`).
fn sign(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v.signum() }
}
