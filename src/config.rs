//! Loading of the TOML config file shared by all subcommands.
//!
//! Each subcommand has its own schema, which owns some top-level tables of the
//! file and ignores the rest, so one file can configure several subcommands.
//! Unknown options inside a schema's own tables are errors; top-level tables
//! that no schema owns are errors too, so typos are still caught.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;

use crate::elev::options::ElevConfig;
use crate::options::Config;

/// The config schema of one subcommand.
pub trait Schema: DeserializeOwned {
    /// The top-level tables this schema reads.
    const SECTIONS: &'static [&'static str];
}

/// Loads the schema `T` from the config file at `path`.
pub fn load<T: Schema>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text).with_context(|| format!("parsing {}", path.display()))
}

fn parse<T: Schema>(text: &str) -> Result<T> {
    let table: toml::Table = toml::from_str(text)?;
    let known = [Config::SECTIONS, ElevConfig::SECTIONS];
    if let Some(key) = table
        .keys()
        .find(|k| !known.iter().any(|s| s.contains(&k.as_str())))
    {
        let all: Vec<_> = known.iter().flat_map(|s| s.iter()).collect();
        bail!("unknown section `{key}` (expected one of {all:?})");
    }
    Ok(toml::from_str(text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: &str = r#"
        [map]
        file_in = "in.png"

        [elev]
        output = "out.png"
        [[elev.layers]]
        onecolor = [1, 2, 3]
        max = 1
        min = 0
        map = "m.png"
    "#;

    #[test]
    fn each_schema_ignores_the_others_sections() {
        let project: Config = parse(BOTH).unwrap();
        assert_eq!(project.map.file_in.unwrap().to_str(), Some("in.png"));
        let elev: ElevConfig = parse(BOTH).unwrap();
        assert_eq!(elev.elev.layers.len(), 1);
    }

    #[test]
    fn example_config_is_valid_for_every_schema() {
        let example = include_str!("../carto.example.toml");
        parse::<Config>(example).unwrap();
        parse::<ElevConfig>(example)
            .unwrap()
            .elev
            .validate()
            .unwrap();
    }

    #[test]
    fn rejects_unknown_sections() {
        let err = parse::<Config>("[mapp]\nfile_in = \"x\"").unwrap_err();
        assert!(err.to_string().contains("unknown section `mapp`"));
    }

    #[test]
    fn rejects_unknown_options_in_own_section() {
        assert!(parse::<ElevConfig>("[elev]\nbogus = 1").is_err());
        assert!(parse::<Config>("[map]\nbogus = 1").is_err());
    }
}
