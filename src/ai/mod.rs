pub mod embedding;
pub mod inference;
pub mod worker_pool;

use anyhow::Result;
use serde_json::json;
use std::sync::Arc;

use crate::{ai::inference::InferenceEngine, data::database::VectorIndex};
pub use embedding::EmbeddingEngine;

pub struct AI {
    pub embedder: Arc<dyn EmbeddingEngine + Send + Sync>,
    pub inferrer: Arc<dyn InferenceEngine + Send + Sync>,
}

impl AI {
    pub fn new(
        embedder: Arc<dyn EmbeddingEngine + Send + Sync>,
        inferrer: Arc<dyn InferenceEngine + Send + Sync>,
    ) -> Self {
        AI { embedder, inferrer }
    }

    pub fn answer_question_with_context(
        &self,
        query: &str,
        references: &[VectorIndex],
    ) -> Result<String> {
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

        self.inferrer.run(&prompt, 400)
    }
}
