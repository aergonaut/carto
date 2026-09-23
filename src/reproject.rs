//! Reprojecting an image from one projection and aspect to another.

use std::f64::consts::PI;
use std::ops::Range;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::graticule;
use crate::interp::{self, GridSource};
use crate::math::{arange, pixel_centers_x, pixel_centers_y, py_mod};
use crate::options::{
    Direction, GraticuleMode, GraticuleSpec, InterpType, OutputOptions, Procedure,
    ProjectionParams, Scale,
};
use crate::pad::{self, ConicEdges};
use crate::projection::{AzimType, ProjCtx, Projection, Side, Wrap};
use crate::raster::{Raster, Sample};
use crate::rotation::Aspect;
use crate::symmetry::SymGrid;

/// Everything describing a reprojection besides the image itself.
#[derive(Clone, Debug)]
pub struct Job {
    pub proj_in: Projection,
    pub proj_out: Projection,
    /// Central longitude, central latitude, and rotation, in degrees.
    pub aspect_in: [f64; 3],
    pub aspect_out: [f64; 3],
    pub params: ProjectionParams,
    pub output: OutputOptions,
    pub procedure: Procedure,
}

/// A map projection evaluated for one of the two maps.
#[derive(Clone, Copy)]
struct MapProj<'a> {
    proj: Projection,
    ctx: &'a ProjCtx,
    aspect: Aspect,
}

/// The result of locating every point of one map on the other.
struct Located {
    /// Positions on the other map.
    x: Vec<f64>,
    y: Vec<f64>,
    /// Coordinates on the starting map, treating it as normal aspect.
    start: Option<(Vec<f64>, Vec<f64>)>,
    /// True coordinates.
    tru: Option<(Vec<f64>, Vec<f64>)>,
    /// Coordinates on the other map, treating it as normal aspect.
    end: Option<(Vec<f64>, Vec<f64>)>,
}

/// Which intermediate coordinates [`locate`] should keep.
#[derive(Clone, Copy, Default)]
struct Keep {
    start: bool,
    tru: bool,
    end: bool,
}

/// Builds full coordinate grids from pixel-center coordinates.
fn mesh(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len() * ys.len();
    let mut mx = vec![0.0; n];
    let mut my = vec![0.0; n];
    mx.par_chunks_mut(xs.len())
        .zip(my.par_chunks_mut(xs.len()))
        .zip(ys)
        .for_each(|((rx, ry), &y)| {
            rx.copy_from_slice(xs);
            ry.fill(y);
        });
    (mx, my)
}

/// Finds the position on map `to` of every pixel center `(xs, ys)` of map `from`.
fn locate(
    xs: &[f64],
    ys: &[f64],
    from: MapProj,
    to: MapProj,
    use_sym: bool,
    keep: Keep,
) -> Located {
    let (rows, cols) = (ys.len(), xs.len());
    println!("  Determining lat/lon from {} map...", from.proj);
    let (lon, lat) = if use_sym {
        let g = SymGrid::new(from.proj.symmetry(), rows, cols);
        let (sx, sy) = mesh(&xs[..g.sub_cols], &ys[..g.sub_rows]);
        let (lon, lat) = from.proj.coords(&sx, &sy, from.ctx);
        g.join(&lon, &lat)
    } else {
        let (mx, my) = mesh(xs, ys);
        from.proj.coords(&mx, &my, from.ctx)
    };

    let fmt_aspect = |a: Aspect| {
        format!(
            "({}, {}, {})",
            a.lon.to_degrees(),
            a.lat.to_degrees(),
            a.rot.to_degrees()
        )
    };
    let snapshot =
        |keep: bool, lon: &Vec<f64>, lat: &Vec<f64>| keep.then(|| (lon.clone(), lat.clone()));

    let start = snapshot(keep.start, &lon, &lat);
    let (mut lon, mut lat) = (lon, lat);
    if !from.aspect.is_zero() {
        println!(
            "  Rotating from {} orientation of {}...",
            from.ctx.side.name(),
            fmt_aspect(from.aspect)
        );
        from.aspect.rotate_from(&mut lon, &mut lat);
    }
    let tru = snapshot(keep.tru, &lon, &lat);
    if !to.aspect.is_zero() {
        println!(
            "  Rotating to {} orientation of {}...",
            to.ctx.side.name(),
            fmt_aspect(to.aspect)
        );
        to.aspect.rotate_to(&mut lon, &mut lat);
    }
    let end = snapshot(keep.end, &lon, &lat);

    println!("  Determining corresponding position on {} map...", to.proj);
    // Longitude loops.
    lon.par_iter_mut()
        .for_each(|l| *l = py_mod(*l + PI, 2.0 * PI) - PI);
    let (x, y) = if use_sym && from.aspect.is_zero() && to.aspect.is_zero() {
        let g = SymGrid::new(to.proj.symmetry().after(from.proj.symmetry()), rows, cols);
        g.eval(&lon, &lat, |lon, lat| to.proj.pos(lon, lat, to.ctx))
    } else {
        to.proj.pos(&lon, &lat, to.ctx)
    };
    Located {
        x,
        y,
        start,
        tru,
        end,
    }
}

/// Visibility of every pixel of a map, exploiting symmetry when allowed. For
/// conic maps, also returns which pixels lie beyond the outer edge.
fn quick_vis(
    proj: Projection,
    xs: &[f64],
    ys: &[f64],
    lonlat: Option<(&[f64], &[f64])>,
    ctx: &ProjCtx,
    use_sym: bool,
) -> (Vec<bool>, Option<Vec<bool>>) {
    if proj.has_default_vis() {
        return (vec![true; xs.len() * ys.len()], None);
    }
    if !use_sym {
        let (mx, my) = mesh(xs, ys);
        return proj.vis_far(&mx, &my, lonlat, ctx);
    }
    let g = SymGrid::new(proj.symmetry(), ys.len(), xs.len());
    let (sx, sy) = mesh(&xs[..g.sub_cols], &ys[..g.sub_rows]);
    let sub_lonlat = lonlat.map(|(lon, lat)| g.slice(lon, lat));
    let (vis, far) = proj.vis_far(
        &sx,
        &sy,
        sub_lonlat
            .as_ref()
            .map(|(a, b)| (a.as_slice(), b.as_slice())),
        ctx,
    );
    (g.join_mask(&vis), far.map(|f| g.join_mask(&f)))
}

/// Resolves a Python-style slice `start:stop` (negative indices count from the end).
fn py_slice(start: i64, stop: i64, len: usize) -> Range<usize> {
    let resolve = |i: i64| {
        let i = if i < 0 { i + len as i64 } else { i };
        i.clamp(0, len as i64) as usize
    };
    let (a, b) = (resolve(start), resolve(stop));
    a..b.max(a)
}

/// Resolves a Python-style index, which may count from the end.
fn py_index(values: &[f64], i: i64) -> Result<f64> {
    let j = if i < 0 { i + values.len() as i64 } else { i };
    usize::try_from(j)
        .ok()
        .and_then(|j| values.get(j).copied())
        .context(format!("crop index {i} out of range"))
}

/// Reprojects `input` according to `job`.
pub fn reproject<T: Sample>(input: &Raster<T>, job: &Job) -> Result<Raster<T>> {
    let (proj_in, proj_out) = (job.proj_in, job.proj_out);
    let mut procedure = job.procedure.clone();
    let mut output = job.output.clone();
    let ctx_in = job.params.ctx(Side::Input, &procedure);
    let ctx_out = job.params.ctx(Side::Output, &procedure);

    let direction = if procedure.interp_type == InterpType::None {
        Direction::Backward
    } else {
        match procedure.proj_direction {
            Direction::AvoidIter
                if proj_out.has_iterative_coords() && !proj_in.has_iterative_pos() =>
            {
                Direction::Forward
            }
            Direction::AvoidIter => Direction::Backward,
            d => d,
        }
    };

    // Whether the input map covers the whole globe. (The original applies
    // this test to every input projection, not just azimuthal and conic ones.)
    let mut global_in = true;
    let hem_in = ctx_in.azim_type(false);
    if hem_in == AzimType::Hem || (hem_in == AzimType::Global && proj_in.is_nonglobal()) {
        global_in = false;
        if proj_in.is_azimuthal() {
            procedure.avoid_seam = false;
        }
    }
    if proj_in.is_nonglobal() {
        global_in = false;
    }

    let uncropped;
    let input = match output.is_crop_in {
        Some([left, right, top, bottom, full_w, full_h]) => {
            global_in = false;
            let (full_w, full_h) = (full_w.max(0) as usize, full_h.max(0) as usize);
            let rows = py_slice(top, bottom, full_h);
            let cols = py_slice(left, right, full_w);
            if rows.len() != input.height || cols.len() != input.width {
                bail!(
                    "is_crop_in area is {}x{} but the input image is {}x{}",
                    cols.len(),
                    rows.len(),
                    input.width,
                    input.height
                );
            }
            let mut full = Raster::new(full_w, full_h, input.channels);
            let c = input.channels;
            for (r_in, r) in rows.enumerate() {
                let dst = (r * full_w + cols.start) * c;
                full.data[dst..dst + input.width * c].copy_from_slice(
                    &input.data[r_in * input.width * c..(r_in + 1) * input.width * c],
                );
            }
            uncropped = full;
            &uncropped
        }
        None => input,
    };
    let (w_in, h_in) = (input.width, input.height);
    let x_in = pixel_centers_x(w_in);
    let y_in = pixel_centers_y(h_in);

    let ratio = match output.force_ratio {
        Some(r) => r,
        None => {
            let mut r = proj_out.ratio(&ctx_out);
            if output.relative_ratio {
                r *= (w_in as f64 / h_in as f64) / proj_in.ratio(&ctx_in);
            }
            r
        }
    };
    let (mut w_out, mut h_out) = (w_in, h_in);
    match output.force_scale {
        Some(Scale::Width(w)) => {
            w_out = w as usize;
            h_out = (w as f64 / ratio).round_ties_even() as usize;
        }
        Some(Scale::Size([w, h])) => (w_out, h_out) = (w as usize, h as usize),
        None => {
            let rel = ratio / (w_in as f64 / h_in as f64);
            if rel > 1.0 {
                w_out = (w_in as f64 * rel).round_ties_even() as usize;
            } else if rel < 1.0 {
                h_out = (h_in as f64 / rel).round_ties_even() as usize;
            }
        }
    }
    let (full_w_out, full_h_out) = (w_out, h_out);
    let mut x_out = pixel_centers_x(w_out);
    let mut y_out = pixel_centers_y(h_out);

    if let Some([left, right, top, bottom]) = output.crop_out {
        procedure.use_sym = false;
        println!("   Cropping output area;");
        println!(
            "    For future use of is_crop_in use ({left},{right},{top},{bottom},{w_out},{h_out})"
        );
        x_out = x_out[py_slice(left, right, w_out)].to_vec();
        y_out = y_out[py_slice(top, bottom, h_out)].to_vec();
    }

    let map_in = MapProj {
        proj: proj_in,
        ctx: &ctx_in,
        aspect: Aspect::from_degrees(job.aspect_in),
    };
    let map_out = MapProj {
        proj: proj_out,
        ctx: &ctx_out,
        aspect: Aspect::from_degrees(job.aspect_out),
    };

    let result = match direction {
        Direction::Forward => project_forward(
            input, &x_in, &y_in, &x_out, &y_out, map_in, map_out, &procedure, &output,
        )?,
        _ => {
            let backward = Backward {
                job,
                input,
                x_in: &x_in,
                y_in: &y_in,
                x_out: &x_out,
                y_out: &y_out,
                map_in,
                map_out,
                global_in,
                full_size_out: (full_w_out, full_h_out),
            };
            backward.run(&procedure, &mut output)?
        }
    };
    println!(" Operation complete");
    Ok(result)
}

struct Backward<'a, T> {
    job: &'a Job,
    input: &'a Raster<T>,
    x_in: &'a [f64],
    y_in: &'a [f64],
    x_out: &'a [f64],
    y_out: &'a [f64],
    map_in: MapProj<'a>,
    map_out: MapProj<'a>,
    global_in: bool,
    full_size_out: (usize, usize),
}

impl<T: Sample> Backward<'_, T> {
    fn run(&self, procedure: &Procedure, output: &mut OutputOptions) -> Result<Raster<T>> {
        let (proj_in, proj_out) = (self.map_in.proj, self.map_out.proj);
        let (ctx_in, ctx_out) = (self.map_in.ctx, self.map_out.ctx);
        let (w_out, h_out) = (self.x_out.len(), self.y_out.len());
        println!(" Working backwards from output to input projection");

        let grat_mode = output.graticules;
        let keep = Keep {
            start: grat_mode == Some(GraticuleMode::Out)
                || output.crop_coords_out.is_some()
                || (output.truncate_out && proj_out.uses_lonlat_vis()),
            tru: grat_mode == Some(GraticuleMode::True) || output.crop_coords_true.is_some(),
            end: grat_mode == Some(GraticuleMode::In) || output.crop_coords_in.is_some(),
        };
        let located = locate(
            self.x_out,
            self.y_out,
            self.map_out,
            self.map_in,
            procedure.use_sym,
            keep,
        );
        let (x_ind, y_ind) = (&located.x, &located.y);

        let padded;
        let (source, xs_src) = if procedure.avoid_seam && output.truncate_in {
            println!("  Padding input map to avoid seams...");
            let wrap = proj_in.wrap();
            let (vis, far) = if wrap == Wrap::Rect {
                (None, None)
            } else {
                let (vis, far) = quick_vis(
                    proj_in,
                    self.x_in,
                    self.y_in,
                    None,
                    ctx_in,
                    procedure.use_sym,
                );
                (Some(vis), far)
            };
            let hem = ctx_in.azim_type(false);
            let conic = far
                .as_deref()
                .filter(|_| hem.is_hemispheric() || proj_in.is_nonglobal())
                .map(|far| ConicEdges {
                    far,
                    bihem: hem == AzimType::Bihem,
                });
            let (mut data, xs) = pad::pad(wrap, self.input, self.x_in, vis.as_deref(), conic);
            if let Some([left, right, top, bottom, ..]) = output.is_crop_in {
                // Allow padding partial input maps but erase anything outside the input area.
                // (As in the original, the unpadded indices are applied to the padded map.)
                let rows = py_slice(top, bottom, data.height);
                let cols = py_slice(left, right, data.width);
                for r in 0..data.height {
                    for c in 0..data.width {
                        if !(rows.contains(&r) && cols.contains(&c)) {
                            data.pixel_mut(r, c).fill(T::default());
                        }
                    }
                }
            }
            padded = data;
            (&padded, xs)
        } else {
            (self.input, self.x_in.to_vec())
        };

        let mut out = Raster::new(w_out, h_out, self.input.channels);
        if procedure.interp_type == InterpType::None {
            println!("  Copying data from nearest input pixels...");
            let (wp, hp) = (source.width as f64, source.height as f64);
            let x_off = ((wp - self.input.width as f64) / 2.0).round_ties_even();
            let y_off = ((hp - self.input.height as f64) / 2.0).round_ties_even();
            // NaN positions become index 0, as NumPy's float-to-int cast does.
            let index = |v: f64, max: f64| v.round_ties_even().clamp(0.0, max) as usize;
            out.data
                .par_chunks_mut(out.channels)
                .enumerate()
                .for_each(|(i, px)| {
                    let c = index((x_ind[i] + 1.0) / 2.0 * wp - 0.5 + x_off, wp - 1.0);
                    let r = index((1.0 - y_ind[i]) / 2.0 * hp - 0.5 + y_off, hp - 1.0);
                    px.copy_from_slice(source.pixel(r, c));
                });
        } else {
            println!("  Interpolating data from input map...");
            let src = GridSource {
                xs: &xs_src,
                ys: self.y_in,
                raster: source,
            };
            interp::sample_grid(&src, x_ind, y_ind, procedure.interp_type, &mut out)?;
        }
        println!(" Projection complete");

        let coords_for = |mode: GraticuleMode| match mode {
            GraticuleMode::In => located.end.as_ref(),
            GraticuleMode::Out => located.start.as_ref(),
            GraticuleMode::True => located.tru.as_ref(),
        };

        if let Some(mode) = grat_mode {
            println!("  Adding graticules...");
            let (lon, lat) = coords_for(mode).expect("graticule coordinates kept");
            let lons = self.graticule_lines(&output.grat_lon, true, mode, output, proj_out);
            let lats = self.graticule_lines(&output.grat_lat, false, mode, output, proj_out);
            let grats = graticule::mask(lon, lat, h_out, w_out, &lons, &lats, output.grat_width);
            blend(&mut out, &grats, output.grat_color);
        }

        let crop_coords = [
            (output.crop_coords_true, GraticuleMode::True),
            (output.crop_coords_in, GraticuleMode::In),
            (output.crop_coords_out, GraticuleMode::Out),
        ];
        if (output.truncate_in && !self.global_in)
            || (output.truncate_out && proj_out.wrap() != Wrap::Rect)
            || output.crop_in.is_some()
            || output.crop_out.is_some()
            || crop_coords.iter().any(|(c, _)| c.is_some())
        {
            println!("  Trimming visible map area...");
        }

        // Areas outside the input image.
        let mut vis_all: Vec<bool> = x_ind
            .par_iter()
            .zip(y_ind)
            .map(|(x, y)| !(x.abs() > 1.0 || y.abs() > 1.0))
            .collect();
        if output.truncate_in && !self.global_in {
            let vis = proj_in.vis(x_ind, y_ind, None, ctx_in);
            and_assign(&mut vis_all, &vis);
            if let Some(is_crop) = output.is_crop_in {
                // Limit to the area of the input using the crop_in method.
                let area = [is_crop[0], is_crop[1], is_crop[2], is_crop[3]];
                output.crop_in = Some(match output.crop_in {
                    None => area,
                    Some(c) => [
                        c[0].max(area[0]),
                        c[1].min(area[1]),
                        c[2].max(area[2]),
                        c[3].min(area[3]),
                    ],
                });
            }
        }
        if output.truncate_out {
            let lonlat = located
                .start
                .as_ref()
                .map(|(a, b)| (a.as_slice(), b.as_slice()));
            let (vis, _) = quick_vis(
                proj_out,
                self.x_out,
                self.y_out,
                lonlat,
                ctx_out,
                procedure.use_sym,
            );
            and_assign(&mut vis_all, &vis);
        }
        if let Some(crop) = output.crop_in {
            // As in the original, the bounds come from the (possibly padded) grid.
            let (lenx, leny) = (xs_src.len() as i64, self.y_in.len() as i64);
            let minx = if crop[0] <= 0 {
                -2.0
            } else {
                py_index(&xs_src, crop[0] + 1)? + 1.0 / lenx as f64
            };
            let maxx = if crop[1] >= lenx {
                2.0
            } else {
                py_index(&xs_src, crop[1] + 1)? + 1.0 / lenx as f64
            };
            let maxy = if crop[2] <= 0 {
                2.0
            } else {
                py_index(self.y_in, crop[2])? + 1.0 / leny as f64
            };
            let miny = if crop[3] >= leny {
                -2.0
            } else {
                py_index(self.y_in, crop[3])? + 1.0 / leny as f64
            };
            vis_all
                .par_iter_mut()
                .zip(x_ind.par_iter().zip(y_ind))
                .for_each(|(v, (&x, &y))| {
                    *v &= !(x < minx || x > maxx || y < miny || y > maxy);
                });
        }
        for (crop, mode) in crop_coords {
            let Some([lon0, lon1, lat0, lat1]) = crop else {
                continue;
            };
            let (lon, lat) = coords_for(mode).expect("crop coordinates kept");
            let minlon = if lon0 <= -180.0 {
                -1000.0
            } else {
                lon0.to_radians()
            };
            let maxlon = if lon1 >= 180.0 {
                1000.0
            } else {
                lon1.to_radians()
            };
            let minlat = if lat0 <= -90.0 {
                -1000.0
            } else {
                lat0.to_radians()
            };
            let maxlat = if lat1 >= 90.0 {
                1000.0
            } else {
                lat1.to_radians()
            };
            vis_all
                .par_iter_mut()
                .zip(lon.par_iter().zip(lat))
                .for_each(|(v, (&lon, &lat))| {
                    if minlon > maxlon {
                        // The range crosses the antimeridian.
                        *v &= lon <= maxlon || lon >= minlon;
                    } else {
                        *v &= !(lon < minlon || lon > maxlon);
                    }
                    *v &= !(lat < minlat || lat > maxlat);
                });
        }

        out.data
            .par_chunks_mut(out.channels)
            .zip(&vis_all)
            .for_each(|(px, &v)| {
                if !v {
                    px.fill(T::default());
                }
            });

        if output.crop_vis {
            let bounds = vis_all.iter().enumerate().filter(|(_, v)| **v).fold(
                None::<(usize, usize, usize, usize)>,
                |b, (i, _)| {
                    let (r, c) = (i / w_out, i % w_out);
                    Some(match b {
                        None => (r, r + 1, c, c + 1),
                        Some((r0, r1, c0, c1)) => {
                            (r0.min(r), r1.max(r + 1), c0.min(c), c1.max(c + 1))
                        }
                    })
                },
            );
            let Some((y_min, y_max, x_min, x_max)) = bounds else {
                bail!("no visible area to crop to");
            };
            let (x_off, y_off) = match output.crop_out {
                Some(crop) => (
                    py_slice(crop[0], crop[1], self.full_size_out.0).start,
                    py_slice(crop[2], crop[3], self.full_size_out.1).start,
                ),
                None => (0, 0),
            };
            println!("   Cropping to visible area");
            println!(
                "    For future use of is_crop_in use ({},{},{},{},{},{})",
                x_min + x_off,
                x_max + x_off,
                y_min + y_off,
                y_max + y_off,
                self.full_size_out.0,
                self.full_size_out.1
            );
            out = out.crop(y_min..y_max, x_min..x_max);
        }
        Ok(out)
    }

    /// The graticule lines to draw, in degrees. Given an interval, lines are
    /// spaced from the antimeridian or south pole, leaving out lines that
    /// would fall on the edge of the map.
    fn graticule_lines(
        &self,
        spec: &GraticuleSpec,
        is_lon: bool,
        mode: GraticuleMode,
        output: &OutputOptions,
        proj_out: Projection,
    ) -> Vec<f64> {
        let step = match spec {
            GraticuleSpec::List(lines) => return lines.clone(),
            GraticuleSpec::Interval(step) => *step,
        };
        let max = if is_lon { 180.0 } else { 90.0 };
        let mut lines = arange(-max, max, step);
        if !(output.truncate_out || proj_out.wrap() == Wrap::Rect) {
            return lines;
        }
        let (a_in, a_out) = (self.job.aspect_in, self.job.aspect_out);
        let params = &self.job.params;
        // As in the original, the input map's layout is consulted here.
        let mut hem = self.map_in.ctx.azim_type(proj_out.is_nonglobal());
        let global_out =
            params.azim_type_out == AzimType::Global || params.azim_type == Some(AzimType::Global);
        if proj_out == Projection::LambertConic && global_out {
            hem = AzimType::Global;
        }
        let zero = |a: [f64; 3]| a.iter().all(|v| *v == 0.0);
        let azimuthal = proj_out.is_azimuthal();
        let conic_hem = proj_out.is_conic() && hem.is_hemispheric();
        let mut remove = Vec::new();
        if proj_out == Projection::Stereographic && global_out {
        } else if azimuthal
            && hem == AzimType::Global
            && !matches!(
                proj_out,
                Projection::NicolosiGlobular | Projection::OrteliusOval
            )
        {
            if is_lon
                && a_out[1].abs() == 90.0
                && (mode == GraticuleMode::True
                    || (mode == GraticuleMode::In && a_in[1].abs() == 90.0))
            {
                remove.push(-max);
            }
        } else if mode == GraticuleMode::Out
            || (zero(a_out)
                && (mode == GraticuleMode::True || (mode == GraticuleMode::In && zero(a_in))))
        {
            remove.push(-max);
            if (is_lon && azimuthal && hem == AzimType::Bihem) || (!is_lon && conic_hem) {
                remove.push(0.0);
            } else if is_lon && azimuthal && hem == AzimType::Hem {
                remove.extend([-90.0, 90.0]);
            }
        } else if a_out[1] == 0.0 {
            let meridians = |center: f64, remove: &mut Vec<f64>| {
                if azimuthal {
                    match hem {
                        AzimType::Bihem => remove.push(py_mod(center + 180.0, 360.0) - 180.0),
                        AzimType::Hem => remove.extend([
                            py_mod(center + 90.0, 360.0) - 180.0,
                            py_mod(center + 270.0, 360.0) - 180.0,
                        ]),
                        _ => {}
                    }
                }
            };
            if mode == GraticuleMode::True {
                if is_lon {
                    remove.push(py_mod(a_out[0], 360.0) - 180.0);
                    meridians(a_out[0], &mut remove);
                } else {
                    remove.push(-max);
                    if conic_hem {
                        remove.push(0.0);
                    }
                }
            } else if mode == GraticuleMode::In && a_in[1] == 0.0 {
                if is_lon {
                    let diff = a_out[0] - a_in[0];
                    if diff > 0.0 {
                        remove.push(py_mod(diff, 360.0) - 180.0);
                    }
                    meridians(diff, &mut remove);
                } else {
                    remove.push(-max);
                    if conic_hem {
                        remove.push(0.0);
                    }
                }
            }
        }
        for r in remove {
            if let Some(i) = lines.iter().position(|&l| l == r) {
                lines.remove(i);
            }
        }
        lines
    }
}

fn and_assign(acc: &mut [bool], mask: &[bool]) {
    acc.par_iter_mut().zip(mask).for_each(|(a, &m)| *a &= m);
}

/// Blends `color` (RGBA) over the pixels in `mask`, preserving any alpha channel.
fn blend<T: Sample>(raster: &mut Raster<T>, mask: &[bool], color: [u8; 4]) {
    let alpha = color[3] as f64 / 255.0;
    let rgb = [color[0] as f64, color[1] as f64, color[2] as f64];
    let color_channels = match raster.channels {
        1 | 2 => 1,
        _ => 3,
    };
    raster
        .data
        .par_chunks_mut(raster.channels)
        .zip(mask)
        .for_each(|(px, &m)| {
            if m {
                for (v, col) in px.iter_mut().zip(&rgb[..color_channels]) {
                    *v = T::from_f64(v.to_f64() * (1.0 - alpha) + col * alpha);
                }
            }
        });
}

/// Forward projection: scatter every visible input pixel onto the output map
/// and interpolate between them.
#[allow(clippy::too_many_arguments)]
fn project_forward<T: Sample>(
    input: &Raster<T>,
    x_in: &[f64],
    y_in: &[f64],
    x_out: &[f64],
    y_out: &[f64],
    map_in: MapProj,
    map_out: MapProj,
    procedure: &Procedure,
    output: &OutputOptions,
) -> Result<Raster<T>> {
    println!(" Working forwards from input to output projection");
    let keep = Keep {
        start: true,
        tru: false,
        end: procedure.avoid_seam,
    };
    let located = locate(x_in, y_in, map_in, map_out, procedure.use_sym, keep);
    let (lon_in, lat_in) = located.start.as_ref().expect("input coordinates kept");
    let (vis_in, _) = quick_vis(
        map_in.proj,
        x_in,
        y_in,
        Some((lon_in, lat_in)),
        map_in.ctx,
        procedure.use_sym,
    );

    let channels = input.channels;
    let visible: Vec<usize> = (0..vis_in.len()).filter(|&i| vis_in[i]).collect();
    let mut px: Vec<f64> = visible.iter().map(|&i| located.x[i]).collect();
    let mut py: Vec<f64> = visible.iter().map(|&i| located.y[i]).collect();
    let mut sources = visible;

    if procedure.avoid_seam {
        // Duplicate points near the antimeridian onto the other side.
        let (lon_out, _) = located.end.as_ref().expect("output coordinates kept");
        let extras: Vec<usize> = (0..lon_in.len())
            .filter(|&i| lon_in[i].abs() > 170f64.to_radians())
            .collect();
        let lon_ext: Vec<f64> = extras
            .iter()
            .map(|&i| {
                if lon_out[i] > 0.0 {
                    lon_out[i] - 2.0 * PI
                } else {
                    lon_out[i] + 2.0 * PI
                }
            })
            .collect();
        // As in the original, latitudes are taken from the input map.
        let lat_ext: Vec<f64> = extras.iter().map(|&i| lat_in[i]).collect();
        let (x_ext, y_ext) = map_out.proj.pos(&lon_ext, &lat_ext, map_out.ctx);
        px.extend(x_ext);
        py.extend(y_ext);
        sources.extend(extras);
    }

    let values: Vec<Vec<f64>> = (0..channels)
        .map(|ch| {
            sources
                .iter()
                .map(|&i| input.data[i * channels + ch].to_f64())
                .collect()
        })
        .collect();
    let gridded =
        interp::scattered::griddata(&px, &py, &values, x_out, y_out, procedure.interp_type)?;

    let mut out = Raster::new(x_out.len(), y_out.len(), channels);
    for (ch, plane) in gridded.iter().enumerate() {
        for (i, v) in plane.iter().enumerate() {
            out.data[i * channels + ch] = T::from_f64(*v);
        }
    }
    if output.truncate_out {
        let (vis_out, _) = quick_vis(
            map_out.proj,
            x_out,
            y_out,
            None,
            map_out.ctx,
            procedure.use_sym,
        );
        out.data
            .par_chunks_mut(channels)
            .zip(&vis_out)
            .for_each(|(px, &v)| {
                if !v {
                    px.fill(T::default());
                }
            });
    }
    Ok(out)
}
