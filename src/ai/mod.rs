pub mod embedding;
pub mod inference;

use anyhow::{Ok, Result};
pub use embedding::EmbeddingEngine;

use crate::ai::{
    embedding::Embedder,
    inference::{InferenceEngine, TextGeneration},
};

struct AI<E: EmbeddingEngine, I: InferenceEngine> {
    pub embedder: E,
    pub inferrer: I,
}

impl<E: EmbeddingEngine, I: InferenceEngine> AI<E, I> {
    pub async fn new(embedder: E, inferrer: I) -> Result<Self> {
        Ok(AI { embedder, inferrer })
    }
}
