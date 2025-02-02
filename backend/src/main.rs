use axum::{routing::get, routing::post, Router};
use bert::get_embeddings;
use clap::Parser;
use cli::{Cli, Commands};
use database::get_db;
use handler::ask_question_cli;
use serde_json::to_vec;
use tokio::io::{self, AsyncBufReadExt, BufReader, Lines};
use std::error::Error;

mod inference;
mod handler;
mod cli;
mod bert;
mod database;
mod ingest;
mod utils;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // server for api 
    let server_task = tokio::spawn(async {
        // server routes
        let app: Router = Router::new()
            .route("/api", get(|| async { "hello" }))
            .route("/api/ask", post(handler::ask_question_api));

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    println!("http server is running!");
    println!("please enter 'ask', 'forget', 'remember' for cli tools!");

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
            Ok(tokens)  => tokens,
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

                        // process the question
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

                    }
                    Commands::List { start, limit } => {

                    }
                    Commands::Forget { content_id, all } => {

                    }
                    Commands::Upload { path } => {

                    }
                }
            }, 
            // need to exclude '--help' as it falls into this condition
            Err(e) => {
                eprintln!("error parsing commands: {}", e);
            }
        }
    }

    let _ = server_task.await;

    Ok(())
}  
