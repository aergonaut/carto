use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame};

use carto::options::{Config, MapInfo};
use carto::projection::{AzimType, Projection};
use carto::raster::AnyRaster;
use carto::reproject::{Job, reproject};
use carto::{setup, with_raster};

const DEFAULT_CONFIG: &str = "carto.toml";

/// Reprojects maps between arbitrary projections and aspects.
///
/// Options are read from a TOML config file (by default `carto.toml` in the
/// current directory, if present). Unless the config sets `skip_setup` or the
/// maps are given on the command line, the maps and projections are prompted for.
#[derive(Parser)]
#[command(version, args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    project: ProjectArgs,
}

#[derive(Subcommand)]
enum Command {
    /// List the available projections.
    List,
    /// Render an animated GIF of a rotating orthographic globe.
    Globe(GlobeArgs),
}

#[derive(Args)]
struct ConfigArg {
    /// Config file to load options from.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
}

impl ConfigArg {
    fn load(&self) -> Result<Config> {
        let path = match &self.config {
            Some(path) => path.as_path(),
            None if Path::new(DEFAULT_CONFIG).exists() => Path::new(DEFAULT_CONFIG),
            None => return Ok(Config::default()),
        };
        println!("Loading options from {}...", path.display());
        Config::load(path)
    }
}

#[derive(Args)]
struct ProjectArgs {
    #[command(flatten)]
    config: ConfigArg,
    /// Input map file.
    #[arg(short, long)]
    input: Option<PathBuf>,
    /// Output map file.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Input map projection (name or number from `carto list`).
    #[arg(long)]
    proj_in: Option<Projection>,
    /// Output map projection (name or number from `carto list`).
    #[arg(long)]
    proj_out: Option<Projection>,
    /// Input map aspect: central longitude, central latitude, rotation (degrees).
    #[arg(long, value_parser = parse_aspect, allow_hyphen_values = true)]
    aspect_in: Option<[f64; 3]>,
    /// Output map aspect: central longitude, central latitude, rotation (degrees).
    #[arg(long, value_parser = parse_aspect, allow_hyphen_values = true)]
    aspect_out: Option<[f64; 3]>,
}

#[derive(Args)]
struct GlobeArgs {
    #[command(flatten)]
    config: ConfigArg,
    /// Input map file.
    #[arg(short, long)]
    input: PathBuf,
    /// Output GIF file.
    #[arg(short, long, default_value = "globe.gif")]
    output: PathBuf,
    /// Input map projection (name or number from `carto list`).
    #[arg(long, default_value = "Equirectangular")]
    proj_in: Projection,
    /// Initial globe aspect: central longitude, central latitude, rotation (degrees).
    #[arg(long, value_parser = parse_aspect, allow_hyphen_values = true, default_value = "0,0,0")]
    aspect: [f64; 3],
    /// Number of frames in a full rotation.
    #[arg(long, default_value_t = 36)]
    frames: u32,
    /// Duration of each frame in milliseconds.
    #[arg(long, default_value_t = 100)]
    duration: u32,
    /// Spin westward rather than eastward.
    #[arg(long)]
    retrograde: bool,
}

fn parse_aspect(s: &str) -> Result<[f64; 3], String> {
    let values: Vec<f64> = s
        .trim_matches(|c| c == '(' || c == ')' || c == '[' || c == ']')
        .split(',')
        .map(|v| v.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    values
        .try_into()
        .map_err(|_| "expected three comma-separated values: lon,lat,rot".to_string())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::List) => {
            setup::print_projection_list();
            Ok(())
        }
        Some(Command::Globe(args)) => globe(args),
        None => project(cli.project),
    }
}

fn banner() {
    println!(
        "carto {} (a port of Projection Pasta 2.0.1)\n\
         For Reprojection of maps between arbitrary aspects\n\
         Originally made 2023 by Mads de Silva and Nikolai Hersfeldt\n",
        env!("CARGO_PKG_VERSION")
    );
}

fn project(args: ProjectArgs) -> Result<()> {
    banner();
    let mut config = args.config.load()?;
    let map = &mut config.map;
    let given_on_cli = args.input.is_some();
    map.file_in = args.input.or(map.file_in.take());
    map.file_out = args.output.or(map.file_out.take());
    map.proj_in = args.proj_in.or(map.proj_in);
    map.proj_out = args.proj_out.or(map.proj_out);
    map.aspect_in = args.aspect_in.unwrap_or(map.aspect_in);
    map.aspect_out = args.aspect_out.unwrap_or(map.aspect_out);

    if map.skip_setup || given_on_cli {
        println!("Skipping setup");
    } else {
        setup::run(map)?;
    }
    println!("\nWorking...");

    let MapInfo {
        file_in,
        file_out,
        proj_in,
        proj_out,
        aspect_in,
        aspect_out,
        ..
    } = &config.map;
    let file_in = file_in.as_ref().context("no input file given")?;
    let file_out = file_out.as_ref().context("no output file given")?;
    let job = Job {
        proj_in: proj_in.context("no input projection given")?,
        proj_out: proj_out.context("no output projection given")?,
        aspect_in: *aspect_in,
        aspect_out: *aspect_out,
        params: config.map.params(),
        output: config.output.clone(),
        procedure: config.procedure.clone(),
    };

    let input = AnyRaster::open(file_in)?;
    let result = with_raster!(&input, r => reproject(r, &job)?);
    println!("  Saving image...");
    result.save(file_out)?;
    println!(" Map saved to {}", file_out.display());
    Ok(())
}

fn globe(args: GlobeArgs) -> Result<()> {
    let config = args.config.load()?;
    let mut params = config.map.params();
    params.azim_type_out = AzimType::Hem;
    let mut job = Job {
        proj_in: args.proj_in,
        proj_out: Projection::Orthographic,
        aspect_in: [0.0; 3],
        aspect_out: args.aspect,
        params,
        output: config.output,
        procedure: config.procedure,
    };
    let input = AnyRaster::open(&args.input)?;
    let step = 360.0 / args.frames as f64 * if args.retrograde { 1.0 } else { -1.0 };

    let file = std::fs::File::create(&args.output)
        .with_context(|| format!("creating {}", args.output.display()))?;
    let mut encoder = GifEncoder::new(file);
    encoder.set_repeat(Repeat::Infinite)?;
    for f in 0..args.frames {
        println!("Frame {f}");
        let frame = with_raster!(&input, r => reproject(r, &job)?);
        let delay = Delay::from_numer_denom_ms(args.duration, 1);
        encoder.encode_frame(Frame::from_parts(
            frame.to_dynamic()?.to_rgba8(),
            0,
            0,
            delay,
        ))?;
        job.aspect_out[0] += step;
    }
    println!(" Globe saved to {}", args.output.display());
    Ok(())
}
