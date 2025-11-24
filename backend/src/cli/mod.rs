pub mod runner;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "ragme")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)] // requires `derive` feature
pub enum Commands {
    #[command(arg_required_else_help = true)]
    Ask {
        query: Vec<String>,
    },
    // for sentences
    Remember {
        content: String,
    },
    // for files
    Upload {
        // add content type for ingesting data
        path: PathBuf,
    },
    Forget {
        // the content to forget
        #[arg(group = "forget")]
        content_id: Option<String>,
        // forget all content
        #[arg(short, long, group = "forget", default_value = "false")]
        all: bool,
    },
    List {
        // how many items you want to skip from the beginning
        #[arg(short, long, default_value = "0")]
        start: u16,
        // how many items you want to get
        #[arg(short, long, default_value = "10")]
        limit: u16,
    },
}
