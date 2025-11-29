use lib::{
    ai::{embedding::Embedder, worker_pool::WorkerPool, AI},
    cli,
    data::database::VDB,
    utils::device,
};
use std::{error::Error, sync::Arc};
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let device = Arc::new(device(false)?);
    let embedding_serivce =
        Arc::new(Embedder::new("ibm-granite/granite-embedding-30m-sparse").await?);
    let inference_pool = Arc::new(Mutex::new(
        WorkerPool::new(1, 5, device.clone(), "allenai/Olmo-3-7B-Think").await?,
    ));
    let ai_service = Arc::new(AI::new(embedding_serivce.clone(), inference_pool));

    let vdb = Arc::new(VDB::new(embedding_serivce.clone()).await?);

    cli::runner::run_repl(vdb.clone(), ai_service.clone()).await?;

    Ok(())
}
