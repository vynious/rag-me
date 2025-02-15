use axum::{routing::get, routing::post, Router};
use clap::Parser;
use lib::cli::{Cli, Commands};
use lib::database::get_all_content;
use lib::handler::{ask_question_cli, ask_question_api};
use std::{error::Error, time::Duration};
use tokio::{
    io::{self, AsyncBufReadExt, BufReader},
    time::timeout,
};
use lib::utils::get_current_working_dir;
use lib::ingest;


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // server for api
    let server_task = tokio::spawn(async {
        // server routes
        let app: Router = Router::new()
            .route("/api", get(|| async { "hello" }))
            .route("/api/ask", post(ask_question_api));

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    println!("http server is running!");
    println!("please enter 'ask', 'forget', 'remember', 'upload' for cli tools!");

    // read from stdin and parse into clap cli commands for actions
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

        // prepend the command
        let mut cli_args = vec!["ragme"];
        cli_args.extend(args.iter().map(String::as_str));
        match Cli::try_parse_from(cli_args) {
            Ok(cli) => {
                match cli.command {
                    Commands::Ask { query } => {
                        eprintln!("query = {}", query);
                        let response = ask_question_cli(&query).await;
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
                    Commands::Forget { content_id, all } => {}
                    Commands::Upload { path } => {
                        eprintln!("path = {:?}", path);
                        // get current working directory
                        let cwd = get_current_working_dir().unwrap();
                        // join path with current working directory
                        let path = cwd.join(path);

                        // check file type and call ingest
                        // attempt to open file and get metadata with 5s timeout
                        let result =
                            match timeout(Duration::from_secs(5), tokio::fs::metadata(&path)).await
                            {
                                Ok(result) => result,
                                Err(e) => {
                                    eprintln!("Failed to get metadata for path due to: {:?}", e);
                                    continue;
                                }
                            };
                        // guard statement
                        let metadata = match result {
                            Ok(metadata) => metadata,
                            Err(e) => {
                                eprintln!("Failed to get metadata for path: {:?}", e);
                                continue;
                            }
                        };
                        // guard statement
                        if !metadata.is_file() {
                            eprintln!("Path is not a file: {:?}", path);
                            continue;
                        };
                        // check file extension
                        if let Some(extension) = path.extension() {
                            match extension.to_str() {
                                Some("txt") => {
                                    eprintln!("ingesting txt file");
                                    // Call ingest function for txt files
                                    ingest::ingest_via_txt(&path).await.unwrap();
                                }
                                Some("pdf") => {
                                    eprintln!("ingesting pdf file");
                                    // Call ingest function for csv files
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
                }
            }
            // need to exclude '--help' as it falls into this condition
            Err(e) => {
                eprintln!("error parsing commands: {}", e);
            }
        }
    }

    let _ = server_task.await;

    Ok(())
}
