//! Interpolating scattered points onto a regular grid (SciPy `griddata`).
//!
//! Linear and cubic interpolation triangulate the points (Delaunay, like
//! Qhull) and then rasterize each triangle onto the grid, rather than locating
//! every grid position in the triangulation. Nearest-neighbour interpolation
//! queries a k-d tree.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use delaunator::{EMPTY, Point, Triangulation, next_halfedge};
use rayon::prelude::*;

use crate::options::InterpType;

/// Qhull's tolerance on barycentric coordinates when locating a point.
const EPS: f64 = 100.0 * f64::EPSILON;

/// Interpolates `values` (one slice per channel) known at scattered points
/// `(px, py)` onto the grid of all `(xs[c], ys[r])`, in row-major order.
/// Positions outside the data yield NaN.
pub fn griddata(
    px: &[f64],
    py: &[f64],
    values: &[Vec<f64>],
    xs: &[f64],
    ys: &[f64],
    method: InterpType,
) -> Result<Vec<Vec<f64>>> {
    // Points with NaN coordinates can't be placed, so they're dropped.
    let ids: Vec<usize> = (0..px.len())
        .filter(|&i| px[i].is_finite() && py[i].is_finite())
        .collect();
    match method {
        InterpType::Nearest => {
            let tree = KdTree::new(ids.iter().map(|&i| ([px[i], py[i]], i)).collect());
            let nearest = tree.nearest_on_grid(xs, ys);
            Ok(values
                .iter()
                .map(|v| {
                    nearest
                        .par_iter()
                        .map(|&i| if i == EMPTY { f64::NAN } else { v[i] })
                        .collect()
                })
                .collect())
        }
        InterpType::Linear | InterpType::Cubic => {
            let mesh = Mesh::new(px, py, ids);
            let owner = mesh.rasterize(xs, ys);
            // The values at the triangulated points, per channel.
            let data: Vec<Vec<f64>> = values
                .par_iter()
                .map(|v| mesh.ids.iter().map(|&i| v[i]).collect())
                .collect();
            if method == InterpType::Linear {
                return Ok(data
                    .iter()
                    .map(|v| {
                        mesh.evaluate(&owner, xs, ys, v, |_, _, b, f| {
                            b[0] * f[0] + b[1] * f[1] + b[2] * f[2]
                        })
                    })
                    .collect());
            }
            let neighbors = VertexNeighbors::new(&mesh);
            let grads: Vec<Vec<[f64; 2]>> = data
                .par_iter()
                .map(|v| neighbors.estimate_gradients(&mesh, v))
                .collect();
            Ok(data
                .iter()
                .zip(&grads)
                .map(|(v, grad)| {
                    mesh.evaluate(&owner, xs, ys, v, |t, bary, b, f| {
                        let df = mesh.tri_vertices(t).map(|i| grad[i]);
                        mesh.clough_tocher(t, bary, b, f, df)
                    })
                })
                .collect())
        }
        _ => bail!(
            "forward projection does not support {} interpolation (use nearest, linear or cubic)",
            method.name()
        ),
    }
}

/// A Delaunay triangulation of the valid scattered points.
struct Mesh {
    points: Vec<Point>,
    /// The original index of each point.
    ids: Vec<usize>,
    tri: Triangulation,
}

impl Mesh {
    fn new(px: &[f64], py: &[f64], ids: Vec<usize>) -> Self {
        let points: Vec<Point> = ids.iter().map(|&i| Point { x: px[i], y: py[i] }).collect();
        // Delaunator leaves duplicate points out of the triangulation, as Qhull does.
        let tri = delaunator::triangulate(&points);
        Mesh { points, ids, tri }
    }

    fn tri_vertices(&self, t: usize) -> [usize; 3] {
        [0, 1, 2].map(|k| self.tri.triangles[3 * t + k])
    }

    fn tri_points(&self, t: usize) -> [[f64; 2]; 3] {
        self.tri_vertices(t)
            .map(|i| [self.points[i].x, self.points[i].y])
    }

    /// The edge from point `i` to point `j`, with its length cubed.
    fn edge(&self, i: usize, j: usize) -> (f64, f64, f64) {
        let ex = self.points[j].x - self.points[i].x;
        let ey = self.points[j].y - self.points[i].y;
        let l = (ex * ex + ey * ey).sqrt();
        (ex, ey, l * l * l)
    }

    /// The triangle across the edge opposite vertex `k` of triangle `t`.
    fn neighbor(&self, t: usize, k: usize) -> Option<usize> {
        let e = self.tri.halfedges[3 * t + (k + 1) % 3];
        (e != EMPTY).then_some(e / 3)
    }

    /// Finds the triangle containing each grid position, or `EMPTY` outside
    /// the triangulation. A position on a shared edge goes to the triangle
    /// with the lowest index.
    fn rasterize(&self, xs: &[f64], ys: &[f64]) -> Vec<usize> {
        let owner: Vec<AtomicUsize> = (0..xs.len() * ys.len())
            .map(|_| AtomicUsize::new(EMPTY))
            .collect();
        (0..self.tri.len()).into_par_iter().for_each(|t| {
            let p = self.tri_points(t);
            if let Some(bary) = Barycentric::new(p) {
                bary.for_each_inside(p, xs, ys, |i| {
                    owner[i].fetch_min(t, Ordering::Relaxed);
                });
            }
        });
        owner.into_iter().map(AtomicUsize::into_inner).collect()
    }

    /// Evaluates `f(t, bary, b, values)` at each grid position owned by a
    /// triangle `t`, where `b` are the position's barycentric coordinates and
    /// `values` are `v` at the triangle's vertices.
    fn evaluate<F>(&self, owner: &[usize], xs: &[f64], ys: &[f64], v: &[f64], f: F) -> Vec<f64>
    where
        F: Fn(usize, &Barycentric, [f64; 3], [f64; 3]) -> f64 + Sync,
    {
        owner
            .par_iter()
            .enumerate()
            .map(|(i, &t)| {
                if t == EMPTY {
                    return f64::NAN;
                }
                // Rasterizing only assigns triangles with a valid transform.
                let bary = Barycentric::new(self.tri_points(t)).unwrap();
                let b = bary.at(xs[i % xs.len()], ys[i / xs.len()]);
                f(t, &bary, b, self.tri_vertices(t).map(|k| v[k]))
            })
            .collect()
    }

    /// Evaluates the Clough-Tocher interpolant on triangle `t` at barycentric
    /// coordinates `b`, given the values `f` and gradients `df` at its
    /// vertices. This follows SciPy's `_clough_tocher_2d_single` exactly.
    fn clough_tocher(
        &self,
        t: usize,
        bary: &Barycentric,
        b: [f64; 3],
        f: [f64; 3],
        df: [[f64; 2]; 3],
    ) -> f64 {
        let p = self.tri_points(t);
        let edge = |i: usize, j: usize| [p[j][0] - p[i][0], p[j][1] - p[i][1]];
        let (e12, e23, e31) = (edge(0, 1), edge(1, 2), edge(2, 0));
        let dot = |g: [f64; 2], e: [f64; 2]| g[0] * e[0] + g[1] * e[1];

        let df12 = dot(df[0], e12);
        let df21 = -dot(df[1], e12);
        let df23 = dot(df[1], e23);
        let df32 = -dot(df[2], e23);
        let df31 = dot(df[2], e31);
        let df13 = -dot(df[0], e31);

        let c3000 = f[0];
        let c2100 = (df12 + 3.0 * c3000) / 3.0;
        let c2010 = (df13 + 3.0 * c3000) / 3.0;
        let c0300 = f[1];
        let c1200 = (df21 + 3.0 * c0300) / 3.0;
        let c0210 = (df23 + 3.0 * c0300) / 3.0;
        let c0030 = f[2];
        let c1020 = (df31 + 3.0 * c0030) / 3.0;
        let c0120 = (df32 + 3.0 * c0030) / 3.0;

        let c2001 = (c2100 + c2010 + c3000) / 3.0;
        let c0201 = (c1200 + c0300 + c0210) / 3.0;
        let c0021 = (c1020 + c0120 + c0030) / 3.0;

        // The cross-boundary derivative is taken toward the neighbour's
        // centroid, which keeps the interpolant C1 and affine invariant.
        let g: [f64; 3] = std::array::from_fn(|k| {
            let Some(n) = self.neighbor(t, k) else {
                return -0.5;
            };
            let q = self.tri_points(n);
            let c = bary.at(
                (q[0][0] + q[1][0] + q[2][0]) / 3.0,
                (q[0][1] + q[1][1] + q[2][1]) / 3.0,
            );
            let (a, b) = match k {
                0 => (c[2], c[1]),
                1 => (c[0], c[2]),
                _ => (c[1], c[0]),
            };
            (2.0 * a + b - 1.0) / (2.0 - 3.0 * a - 3.0 * b)
        });

        let c0111 = (g[0] * (-c0300 + 3.0 * c0210 - 3.0 * c0120 + c0030)
            + (-c0300 + 2.0 * c0210 - c0120 + c0021 + c0201))
            / 2.0;
        let c1011 = (g[1] * (-c0030 + 3.0 * c1020 - 3.0 * c2010 + c3000)
            + (-c0030 + 2.0 * c1020 - c2010 + c2001 + c0021))
            / 2.0;
        let c1101 = (g[2] * (-c3000 + 3.0 * c2100 - 3.0 * c1200 + c0300)
            + (-c3000 + 2.0 * c2100 - c1200 + c2001 + c0201))
            / 2.0;

        let c1002 = (c1101 + c1011 + c2001) / 3.0;
        let c0102 = (c1101 + c0111 + c0201) / 3.0;
        let c0012 = (c1011 + c0111 + c0021) / 3.0;
        let c0003 = (c1002 + c0102 + c0012) / 3.0;

        // Coordinates in the sub-triangle (split at the centroid) holding the point.
        let min = b[0].min(b[1]).min(b[2]);
        let (b1, b2, b3, b4) = (b[0] - min, b[1] - min, b[2] - min, 3.0 * min);

        b1.powi(3) * c3000
            + 3.0 * b1.powi(2) * b2 * c2100
            + 3.0 * b1.powi(2) * b3 * c2010
            + 3.0 * b1.powi(2) * b4 * c2001
            + 3.0 * b1 * b2.powi(2) * c1200
            + 6.0 * b1 * b2 * b4 * c1101
            + 3.0 * b1 * b3.powi(2) * c1020
            + 6.0 * b1 * b3 * b4 * c1011
            + 3.0 * b1 * b4.powi(2) * c1002
            + b2.powi(3) * c0300
            + 3.0 * b2.powi(2) * b3 * c0210
            + 3.0 * b2.powi(2) * b4 * c0201
            + 3.0 * b2 * b3.powi(2) * c0120
            + 6.0 * b2 * b3 * b4 * c0111
            + 3.0 * b2 * b4.powi(2) * c0102
            + b3.powi(3) * c0030
            + 3.0 * b3.powi(2) * b4 * c0021
            + 3.0 * b3 * b4.powi(2) * c0012
            + b4.powi(3) * c0003
    }
}

/// Each point's neighbours in a triangulation, in compressed sparse row form,
/// with the edge geometry needed for gradient estimation.
struct VertexNeighbors {
    /// The neighbours of point `i` are `indices[indptr[i]..indptr[i + 1]]`.
    indptr: Vec<usize>,
    indices: Vec<usize>,
    /// For each point, the symmetric matrix `Σ 4 e eᵀ / |e|³` over its edges
    /// `e`, as `[xx, xy, yy]`; it depends only on the geometry.
    q: Vec<[f64; 3]>,
}

impl VertexNeighbors {
    fn new(mesh: &Mesh) -> Self {
        let n = mesh.points.len();
        let (tris, halves) = (&mesh.tri.triangles, &mesh.tri.halfedges);
        // Every interior edge appears as two opposite half-edges, but hull
        // edges only once, so those are also added in reverse.
        let edges = || {
            (0..tris.len()).flat_map(move |e| {
                let (a, b) = (tris[e], tris[next_halfedge(e)]);
                std::iter::once((a, b)).chain((halves[e] == EMPTY).then_some((b, a)))
            })
        };
        let mut indptr = vec![0; n + 1];
        for (a, _) in edges() {
            indptr[a + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let mut fill = indptr.clone();
        let mut indices = vec![0; indptr[n]];
        for (a, b) in edges() {
            indices[fill[a]] = b;
            fill[a] += 1;
        }
        let q = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut q = [0.0; 3];
                for &j in &indices[indptr[i]..indptr[i + 1]] {
                    let (ex, ey, l3) = mesh.edge(i, j);
                    q[0] += 4.0 * ex * ex / l3;
                    q[1] += 4.0 * ex * ey / l3;
                    q[2] += 4.0 * ey * ey / l3;
                }
                q
            })
            .collect();
        VertexNeighbors { indptr, indices, q }
    }

    /// Estimates the gradient of `data` at each point by global curvature
    /// minimization, iterating Gauss-Seidel as SciPy's
    /// `estimate_gradients_2d_global` does (tolerance 1e-6, at most 400 sweeps).
    fn estimate_gradients(&self, mesh: &Mesh, data: &[f64]) -> Vec<[f64; 2]> {
        const TOL: f64 = 1e-6;
        const MAX_ITER: usize = 400;
        let mut y = vec![[0.0; 2]; data.len()];
        for _ in 0..MAX_ITER {
            let mut err: f64 = 0.0;
            for i in 0..data.len() {
                let neighbors = &self.indices[self.indptr[i]..self.indptr[i + 1]];
                // Points left out of the triangulation (duplicates) have none.
                if neighbors.is_empty() {
                    continue;
                }
                let mut s = [0.0; 2];
                for &j in neighbors {
                    let (ex, ey, l3) = mesh.edge(i, j);
                    let df2 = -ex * y[j][0] - ey * y[j][1];
                    let w = 6.0 * (data[i] - data[j]) - 2.0 * df2;
                    s[0] += w * ex / l3;
                    s[1] += w * ey / l3;
                }
                let q = self.q[i];
                let det = q[0] * q[2] - q[1] * q[1];
                let r = [
                    (q[2] * s[0] - q[1] * s[1]) / det,
                    (-q[1] * s[0] + q[0] * s[1]) / det,
                ];
                let change = (y[i][0] + r[0]).abs().max((y[i][1] + r[1]).abs());
                y[i] = [-r[0], -r[1]];
                err = err.max(change / r[0].abs().max(r[1].abs()).max(1.0));
            }
            if err < TOL {
                break;
            }
        }
        y
    }
}

/// The affine map from a position to its barycentric coordinates in a
/// triangle, set up as Qhull does (relative to the last vertex).
struct Barycentric {
    inv: [[f64; 2]; 2],
    origin: [f64; 2],
}

impl Barycentric {
    /// Returns `None` for a triangle too degenerate for Qhull to use
    /// (reciprocal condition number below 1000 ε).
    fn new(p: [[f64; 2]; 3]) -> Option<Self> {
        let [a, b] = [p[0][0] - p[2][0], p[1][0] - p[2][0]];
        let [c, d] = [p[0][1] - p[2][1], p[1][1] - p[2][1]];
        let det = a * d - b * c;
        let inv = [[d / det, -b / det], [-c / det, a / det]];
        let norm = (a.abs() + c.abs()).max(b.abs() + d.abs());
        let inv_norm = (inv[0][0].abs() + inv[1][0].abs()).max(inv[0][1].abs() + inv[1][1].abs());
        (1.0 / (norm * inv_norm) >= 1000.0 * f64::EPSILON)
            .then_some(Barycentric { inv, origin: p[2] })
    }

    fn at(&self, x: f64, y: f64) -> [f64; 3] {
        let (dx, dy) = (x - self.origin[0], y - self.origin[1]);
        let c0 = self.inv[0][0] * dx + self.inv[0][1] * dy;
        let c1 = self.inv[1][0] * dx + self.inv[1][1] * dy;
        [c0, c1, 1.0 - c0 - c1]
    }

    /// Calls `f` with the row-major index of every grid position inside the
    /// triangle `p` (within Qhull's tolerance), visiting only candidates near
    /// the triangle.
    fn for_each_inside(&self, p: [[f64; 2]; 3], xs: &[f64], ys: &[f64], mut f: impl FnMut(usize)) {
        let (nx, ny) = (xs.len(), ys.len());
        let y_min = p[0][1].min(p[1][1]).min(p[2][1]);
        let y_max = p[0][1].max(p[1][1]).max(p[2][1]);
        // Candidates are widened by one row and column to absorb rounding.
        let r0 = ys.partition_point(|&y| y > y_max).saturating_sub(1);
        let r1 = (ys.partition_point(|&y| y >= y_min) + 1).min(ny);
        for (r, &y) in ys.iter().enumerate().take(r1).skip(r0) {
            let dy = y - self.origin[1];
            // Along the row, coordinate k is slope[k] * dx + base[k], with dx
            // relative to the origin; each must be at least -EPS.
            let slope = [
                self.inv[0][0],
                self.inv[1][0],
                -self.inv[0][0] - self.inv[1][0],
            ];
            let (b0, b1) = (self.inv[0][1] * dy, self.inv[1][1] * dy);
            let base = [b0, b1, 1.0 - b0 - b1];
            let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
            for k in 0..3 {
                let bound = (-EPS - base[k]) / slope[k];
                if slope[k] > 0.0 {
                    lo = lo.max(bound);
                } else if slope[k] < 0.0 {
                    hi = hi.min(bound);
                } else if base[k] < -EPS {
                    hi = f64::NEG_INFINITY;
                }
            }
            if lo > hi {
                continue;
            }
            let (lo, hi) = (lo + self.origin[0], hi + self.origin[0]);
            let c0 = xs.partition_point(|&x| x < lo).saturating_sub(1);
            let c1 = (xs.partition_point(|&x| x <= hi) + 1).min(nx);
            for (c, &x) in xs.iter().enumerate().take(c1).skip(c0) {
                if self
                    .at(x, y)
                    .iter()
                    .all(|&b| (-EPS..=1.0 + EPS).contains(&b))
                {
                    f(r * nx + c);
                }
            }
        }
    }
}

/// A static 2-d tree for nearest-neighbour queries, laid out implicitly: a
/// node spanning `items[lo..hi]` holds its median `items[mid]`, where
/// `mid = (lo + hi) / 2`, and splits on axis `depth % 2` into children
/// `items[lo..mid]` (at or below the median on that axis) and
/// `items[mid + 1..hi]` (at or above it).
struct KdTree {
    /// Positions with their original indices.
    items: Vec<([f64; 2], usize)>,
    /// The bounding box of all positions, as `[min, max]`.
    bounds: [[f64; 2]; 2],
}

/// Nodes at most this size are searched linearly.
const LEAF_SIZE: usize = 8;

impl KdTree {
    fn new(mut items: Vec<([f64; 2], usize)>) -> Self {
        fn build(items: &mut [([f64; 2], usize)], depth: usize) {
            if items.len() <= LEAF_SIZE {
                return;
            }
            let mid = items.len() / 2;
            items.select_nth_unstable_by(mid, |a, b| a.0[depth % 2].total_cmp(&b.0[depth % 2]));
            let (left, right) = items.split_at_mut(mid);
            rayon::join(
                || build(left, depth + 1),
                || build(&mut right[1..], depth + 1),
            );
        }
        build(&mut items, 0);
        let bounds = items.par_iter().fold_with(
            [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]],
            |[min, max], (p, _)| {
                [
                    [min[0].min(p[0]), min[1].min(p[1])],
                    [max[0].max(p[0]), max[1].max(p[1])],
                ]
            },
        );
        let bounds = bounds.reduce(
            || [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]],
            |[a0, a1], [b0, b1]| {
                [
                    [a0[0].min(b0[0]), a0[1].min(b0[1])],
                    [a1[0].max(b1[0]), a1[1].max(b1[1])],
                ]
            },
        );
        KdTree { items, bounds }
    }

    /// The original index of the item nearest each grid position, row-major,
    /// or `EMPTY` if the tree is empty.
    fn nearest_on_grid(&self, xs: &[f64], ys: &[f64]) -> Vec<usize> {
        let mut out = vec![EMPTY; xs.len() * ys.len()];
        if self.items.is_empty() || xs.is_empty() {
            return out;
        }
        out.par_chunks_mut(xs.len()).zip(ys).for_each(|(row, &y)| {
            // Neighbouring positions usually share a nearest item, so the
            // previous answer gives a tight starting bound.
            let mut prev = 0;
            for (o, &x) in row.iter_mut().zip(xs) {
                let q = [x, y];
                let off = [0, 1].map(|k| {
                    (self.bounds[0][k] - q[k])
                        .max(q[k] - self.bounds[1][k])
                        .max(0.0)
                });
                let mut best = (dist2(q, self.items[prev].0), prev);
                self.search(0, self.items.len(), 0, q, off, &mut best);
                prev = best.1;
                *o = self.items[prev].1;
            }
        });
        out
    }

    /// Updates `best` (squared distance, item) with the nearest item to `q`
    /// in the node spanning `lo..hi` at `depth`, whose cell lies `off` away
    /// from `q` along each axis.
    fn search(
        &self,
        lo: usize,
        hi: usize,
        depth: usize,
        q: [f64; 2],
        off: [f64; 2],
        best: &mut (f64, usize),
    ) {
        if hi - lo <= LEAF_SIZE {
            for i in lo..hi {
                let d = dist2(q, self.items[i].0);
                if d < best.0 {
                    *best = (d, i);
                }
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let d = dist2(q, self.items[mid].0);
        if d < best.0 {
            *best = (d, mid);
        }
        let axis = depth % 2;
        let diff = q[axis] - self.items[mid].0[axis];
        let (near, far) = if diff < 0.0 {
            ((lo, mid), (mid + 1, hi))
        } else {
            ((mid + 1, hi), (lo, mid))
        };
        self.search(near.0, near.1, depth + 1, q, off, best);
        // The far cell is at least `diff` away along the split axis.
        let mut far_off = off;
        far_off[axis] = diff.abs();
        if dist2(far_off, [0.0; 2]) < best.0 {
            self.search(far.0, far.1, depth + 1, q, far_off, best);
        }
    }
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A jittered 12 × 12 lattice on [0, 1]², with no four points cocircular.
    fn scattered() -> (Vec<f64>, Vec<f64>) {
        let jitter = |i: usize, s: f64| ((i as f64 * s).sin() * 43758.5453).fract() * 0.03;
        (0..144)
            .map(|i| {
                (
                    (i % 12) as f64 / 11.0 + jitter(i, 12.9898),
                    (i / 12) as f64 / 11.0 + jitter(i, 78.233),
                )
            })
            .unzip()
    }

    fn grid() -> (Vec<f64>, Vec<f64>) {
        let xs = (0..30).map(|i| -0.2 + i as f64 * 0.05).collect();
        let ys = (0..30).map(|i| 1.2 - i as f64 * 0.05).collect();
        (xs, ys)
    }

    #[test]
    fn linear_and_cubic_reproduce_planes_inside_the_hull() {
        let (px, py) = scattered();
        let (xs, ys) = grid();
        let plane = |x: f64, y: f64| 2.0 * x - 3.0 * y + 1.0;
        let values = vec![px.iter().zip(&py).map(|(&x, &y)| plane(x, y)).collect()];
        for (method, tol) in [(InterpType::Linear, 1e-12), (InterpType::Cubic, 1e-5)] {
            let out = griddata(&px, &py, &values, &xs, &ys, method).unwrap();
            let mut inside = 0;
            for (i, &v) in out[0].iter().enumerate() {
                let (x, y) = (xs[i % 30], ys[i / 30]);
                if v.is_nan() {
                    // Only positions outside the (roughly unit square) hull are missing.
                    assert!(
                        !(0.05..=0.95).contains(&x) || !(0.05..=0.95).contains(&y),
                        "{method:?} at ({x}, {y})"
                    );
                } else {
                    inside += 1;
                    assert!(
                        (v - plane(x, y)).abs() < tol,
                        "{method:?} at ({x}, {y}): {v}"
                    );
                }
            }
            assert!(inside > 300);
        }
    }

    #[test]
    fn nearest_matches_brute_force_and_skips_nan_points() {
        let (mut px, mut py) = scattered();
        px.push(f64::NAN);
        py.push(0.5);
        let (xs, ys) = grid();
        let values = vec![(0..px.len()).map(|i| i as f64).collect()];
        let out = griddata(&px, &py, &values, &xs, &ys, InterpType::Nearest).unwrap();
        for (i, &v) in out[0].iter().enumerate() {
            let q = [xs[i % 30], ys[i / 30]];
            let best = (0..px.len() - 1)
                .min_by(|&a, &b| dist2(q, [px[a], py[a]]).total_cmp(&dist2(q, [px[b], py[b]])));
            assert_eq!(v, best.unwrap() as f64);
        }
    }

    #[test]
    fn duplicate_points_are_tolerated() {
        let (mut px, mut py) = scattered();
        px.extend_from_within(..20);
        py.extend_from_within(..20);
        let (xs, ys) = grid();
        let values = vec![vec![1.0; px.len()]];
        for method in [InterpType::Linear, InterpType::Cubic] {
            let out = griddata(&px, &py, &values, &xs, &ys, method).unwrap();
            assert!(out[0].iter().all(|&v| v.is_nan() || (v - 1.0).abs() < 1e-9));
        }
    }

    #[test]
    fn rejects_grid_only_methods() {
        assert!(griddata(&[], &[], &[], &[], &[], InterpType::Pchip).is_err());
    }
}
