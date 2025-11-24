use axum::Router;
use lib::{cli, http};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let server_task = tokio::spawn(async {
        let app: Router = http::router();
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        println!("http server is running!");
        axum::serve(listener, app).await.unwrap();
    });

    cli::runner::run_repl().await?;
    let _ = server_task.await;

    Ok(())
}
