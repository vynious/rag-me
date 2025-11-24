use crate::{
    cli::{Cli, Commands},
    data::database::get_all_content,
    data::ingest,
    qa::answer_query,
    utils::get_current_working_dir,
};
use clap::Parser;
use std::{error::Error, time::Duration};
use tokio::{
    io::{self, AsyncBufReadExt, BufReader},
    time::timeout,
};

pub async fn run_repl() -> Result<(), Box<dyn Error>> {
    println!("please enter 'ask', 'forget', 'remember', 'upload' for cli tools!");

    let stdin = io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed_line = line.trim();
        if trimmed_line.is_empty() {
            continue;
        }
        println!("receiving cmd: {}", trimmed_line);

        let args = match shell_words::split(trimmed_line) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("error parsing commands: {}", e);
                continue;
            }
        };

        let mut cli_args = vec!["ragme"];
        cli_args.extend(args.iter().map(String::as_str));
        match Cli::try_parse_from(cli_args) {
            Ok(cli) => match cli.command {
                Commands::Ask { query } => {
                    let query_str = query.join(" ");
                    eprintln!("query = {}", query_str);
                    let response = answer_query(&query_str).await;
                    match response {
                        Ok(answer) => {
                            println!("answer: {}", answer);
                        }
                        Err(e) => {
                            eprintln!("error asking question: {}", e);
                        }
                    }
                }
                Commands::Remember { content } => {
                    eprintln!("content = {}", content);
                }
                Commands::List { start, limit } => {
                    if let Ok(content) = get_all_content(start, limit).await {
                        for c in content {
                            println!("content: {:?}", c);
                        }
                    }
                }
                Commands::Forget { content_id, all } => {
                    eprintln!("forget called; content_id={:?}, all={}", content_id, all);
                }
                Commands::Upload { path } => {
                    eprintln!("path = {:?}", path);
                    let cwd = get_current_working_dir().unwrap();
                    let path = cwd.join(path);

                    let result =
                        match timeout(Duration::from_secs(5), tokio::fs::metadata(&path)).await {
                            Ok(result) => result,
                            Err(e) => {
                                eprintln!("Failed to get metadata for path due to: {:?}", e);
                                continue;
                            }
                        };
                    let metadata = match result {
                        Ok(metadata) => metadata,
                        Err(e) => {
                            eprintln!("Failed to get metadata for path: {:?}", e);
                            continue;
                        }
                    };
                    if !metadata.is_file() {
                        eprintln!("Path is not a file: {:?}", path);
                        continue;
                    };

                    if let Some(extension) = path.extension() {
                        match extension.to_str() {
                            Some("txt") => {
                                eprintln!("ingesting txt file");
                                ingest::ingest_via_txt(&path).await.unwrap();
                            }
                            Some("pdf") => {
                                eprintln!("ingesting pdf file");
                                ingest::ingest_via_pdf(&path).await.unwrap();
                            }
                            _ => {
                                eprintln!("Unsupported file type: {:?}", extension);
                            }
                        }
                    } else {
                        eprintln!("File has no extension: {:?}", path);
                    };
                }
            },
            Err(e) => {
                eprintln!("error parsing commands: {}", e);
            }
        }
    }

    Ok(())
}
