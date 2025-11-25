pub mod embedding;
pub mod inference;

use anyhow::{Ok, Result};
pub use embedding::EmbeddingEngine;
use serde_json::json;

use crate::{ai::inference::InferenceEngine, data::database::VectorIndex};

pub struct AI<E: EmbeddingEngine, I: InferenceEngine> {
    pub embedder: E,
    pub inferrer: I,
}

impl<E: EmbeddingEngine, I: InferenceEngine> AI<E, I> {
    pub async fn new(embedder: E, inferrer: I) -> Result<Self> {
        Ok(AI { embedder, inferrer })
    }

    pub async fn answer_question_with_context(
        &mut self,
        query: &str,
        references: &Vec<VectorIndex>,
    ) -> Result<String> {
        let mut context = Vec::new();
        for reference in references.clone() {
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
        let response = self.inferrer.run(&prompt, 400).map_err(|e| {
            eprintln!("error generating response: {}", e);
            e
        })?;
        Ok(response)
    }
}
