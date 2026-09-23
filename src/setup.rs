//! Interactive command-line setup, prompting for the maps and projections.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::options::{MapInfo, RefValues};
use crate::projection::{AzimType, Projection, RefPrompt, Side};

/// Prints the numbered list of projections, grouped into sections.
pub fn print_projection_list() {
    println!("Projection Options and Codes:");
    let width = Projection::ALL
        .iter()
        .map(|p| p.name().len())
        .max()
        .unwrap_or(0);
    for (i, p) in Projection::ALL.iter().enumerate() {
        if let Some(section) = p.section() {
            println!(
                "  ----  {section}  {}",
                "-".repeat(80usize.saturating_sub(section.len()))
            );
        }
        println!("{i:>2}: {:<width$}  {}", p.name(), p.info());
    }
}

struct Prompter<R> {
    input: R,
}

impl<R: BufRead> Prompter<R> {
    fn line(&mut self, prompt: &str) -> Result<String> {
        print!("{prompt}");
        io::stdout().flush()?;
        let mut s = String::new();
        if self.input.read_line(&mut s)? == 0 {
            bail!("input ended during setup");
        }
        Ok(s.trim().to_string())
    }

    /// Prompts until `parse` accepts the input.
    fn ask<T>(&mut self, prompt: &str, parse: impl Fn(&str) -> Option<T>) -> Result<T> {
        let mut s = self.line(prompt)?;
        loop {
            if let Some(v) = parse(&s) {
                return Ok(v);
            }
            s = self.line(" Invalid input, try again: ")?;
        }
    }

    fn ask_f64(&mut self, prompt: &str) -> Result<f64> {
        self.ask(prompt, |s| s.parse().ok())
    }

    fn ask_choice(&mut self, prompt: &str, count: usize) -> Result<usize> {
        self.ask(prompt, |s| s.parse().ok().filter(|&n: &usize| n < count))
    }

    fn ask_ref(&mut self, prompt: &RefPrompt) -> Result<Vec<f64>> {
        println!(" {}", prompt.prompt);
        let choice = if prompt.presets.is_empty() {
            0
        } else {
            println!("  0: Enter custom value");
            for (i, (label, _)) in prompt.presets.iter().enumerate() {
                println!("  {}: {label}", i + 1);
            }
            self.ask_choice("  Choose option: ", prompt.presets.len() + 1)?
        };
        if choice > 0 {
            return Ok(vec![prompt.presets[choice - 1].1]);
        }
        if prompt.count == 1 {
            return Ok(vec![self.ask_f64("  Input custom value (in degrees): ")?]);
        }
        (1..=prompt.count)
            .map(|n| self.ask_f64(&format!("  Input custom value {n} (in degrees): ")))
            .collect()
    }
}

/// Prompts for the input and output maps, filling in `map`.
pub fn run(map: &mut MapInfo) -> Result<()> {
    let mut p = Prompter {
        input: io::stdin().lock(),
    };
    println!();
    print_projection_list();
    for side in [Side::Input, Side::Output] {
        println!(
            "{}",
            if side == Side::Input {
                "Input Image"
            } else {
                "Output Image"
            }
        );
        let file = loop {
            let file = PathBuf::from(p.line(" Filename: ")?);
            if side == Side::Output || file.exists() {
                break file;
            }
            println!("  No file found at {}", file.display());
        };
        let proj = p.ask(" Projection: ", |s| {
            s.parse().ok().and_then(Projection::from_index)
        })?;

        if proj.is_azimuthal() {
            println!(" Projection subtype: ");
            if !proj.is_nonglobal() {
                println!("  0: single global map");
            } else if proj.global_ref_prompt().is_some() {
                println!("  0: large map of custom size");
            }
            println!("  1: single hemisphere");
            println!("  2: bihemisphere; 2 maps of opposite hemispheres");
            let azim = [AzimType::Global, AzimType::Hem, AzimType::Bihem]
                [p.ask_choice("  Choose subtype: ", 3)?];
            match side {
                Side::Input => map.azim_type_in = azim,
                Side::Output => map.azim_type_out = azim,
            }
        }

        let azim = match side {
            Side::Input => map.azim_type_in,
            Side::Output => map.azim_type_out,
        };
        let mut reference: Option<Vec<f64>> = None;
        let global_prompt = proj
            .global_ref_prompt()
            .filter(|_| azim == AzimType::Global);
        for prompt in [global_prompt, proj.ref_prompt()].into_iter().flatten() {
            let values = p.ask_ref(&prompt)?;
            reference = Some(
                reference
                    .unwrap_or_default()
                    .into_iter()
                    .chain(values)
                    .collect(),
            );
        }
        if let Some(r) = reference {
            match side {
                Side::Input => map.ref_in = Some(RefValues(r)),
                Side::Output => map.ref_out = Some(RefValues(r)),
            }
        }

        let aspect = [
            p.ask_f64(" Center longitude (-180 to 180): ")?,
            p.ask_f64(" Center latitude (-90 to 90): ")?,
            p.ask_f64(" Clockwise rotation from north (0 to 360): ")?,
        ];
        println!();
        match side {
            Side::Input => {
                (map.file_in, map.proj_in, map.aspect_in) = (Some(file), Some(proj), aspect)
            }
            Side::Output => {
                (map.file_out, map.proj_out, map.aspect_out) = (Some(file), Some(proj), aspect)
            }
        }
    }
    Ok(())
}
