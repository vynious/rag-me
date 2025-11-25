use axum::Router;
use lib::{
    ai::{embedding::Embedder, inference::TextGeneration, AI},
    cli, http,
    utils::device,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let device = Arc::new(device(false)?);
    let embedding_serivce = Embedder::new("").await?;
    let inference_service = TextGeneration::new(
        "Demonthos/dolphin-2_6-phi-2-candle",
        398752958,
        Some(0.8),
        Some(0.7),
        1.1,
        64,
        device.clone(),
    )
    .await?;
    let ai_service = AI::new(embedder, inferrer);
    
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
