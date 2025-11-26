pub mod embedding;
pub mod inference;
pub mod worker_pool;

use anyhow::Result;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    ai::worker_pool::{InferenceResult, WorkerPool},
    data::database::VectorIndex,
};
pub use embedding::EmbeddingEngine;

pub struct AI {
    pub embedder: Arc<dyn EmbeddingEngine + Send + Sync>,
    pub inference_pool: Arc<Mutex<WorkerPool>>,
}

impl AI {
    pub fn new(
        embedder: Arc<dyn EmbeddingEngine + Send + Sync>,
        inference_pool: Arc<Mutex<WorkerPool>>,
    ) -> Self {
        AI {
            embedder,
            inference_pool,
        }
    }

    pub async fn answer_question_with_context(
        &self,
        query: &str,
        references: &[VectorIndex],
    ) -> Result<InferenceResult> {
        let mut context = Vec::new();
        for reference in references {
            context.push(json!({
                "content": reference.content_chunk,
                "metadata": reference.metadata
            }));
        }

        let context_str = serde_json::to_string(&context)?;
        let prompt = format!(
            "You are a friendly AI agent. Context: {} Query: {}",
            context_str, query
        );

        let result = self.inference_pool.lock().await.accept(prompt, 400).await?;
        Ok(result)
    }
}
