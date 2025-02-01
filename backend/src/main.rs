use axum::{routing::get, routing::post, Router};
use bert::get_embeddings;
use clap::Parser;
use database::get_db;
use crate::cli::{Cli, Commands};

mod handler;
mod cli;
mod bert;
mod database;
mod ingest;
mod utils;

#[tokio::main]
async fn main() {
    // // cli commands
    // let args = Cli::parse();
    // match args.command {
    //     Commands::Ask { query } => {}
    //     Commands::Remember { content } => {}
    //     Commands::List { start, limit } => {}
    //     Commands::Forget { content_id, all } => {}
    //     Commands::Upload { path } => {}
    // }

    // server routes
    let app: Router = Router::new()
        .route("/api", get(|| async { "hello" }))
        .route("/api/ask", post(handler::ask_question));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}  
