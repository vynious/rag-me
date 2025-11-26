use axum::Router;
use lib::{
    ai::{embedding::Embedder, inference::TextGeneration, worker_pool::WorkerPool, AI},
    cli,
    data::database::VDB,
    http,
    utils::device,
};
use std::{error::Error, sync::Arc};
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // AI
    let device = Arc::new(device(false)?);
    let embedding_serivce = Arc::new(Embedder::new("").await?);
    let inference_pool = Arc::new(Mutex::new(WorkerPool::new(3, 5, device.clone(), "")));
    let ai_service = AI::new(embedding_serivce.clone(), inference_pool);

    // Database
    let vdb = Arc::new(VDB::new(embedding_service.clone()));
    let server_task = tokio::spawn(async {
        let app: Router = http::router();
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        println!("http server is running!");
        axum::serve(listener, app).await.unwrap();
    });

    cli::runner::run_repl(vdb.clone()).await?;
    let _ = server_task.await;

    Ok(())
}
