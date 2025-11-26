use std::sync::Arc;

use anyhow::Error;

use crate::{
    ai::AI,
    data::database::{VectorIndex, VDB},
};


pub struct 

pub async fn answer_query(query: &str) -> Result<String, Error> {
    let context = build_context_for_query(query).await?;
    let answer = answer_question_with_context(query, &context).await?;
    Ok(answer)
}

pub async fn build_context_for_query(
    ai: Arc<AI>,
    vdb: Arc<VDB>,
    query: &str,
) -> Result<Vec<VectorIndex>, Error> {
    let query_embedding: Vec<f32> = ai
        .embedder
        .get_embeddings(query)?
        .reshape((384,))?
        .to_vec1()?;
    let related_content = vdb.get_related_chunks(query_embedding).await?;
    let mut context: Vec<VectorIndex> = vec![];
    for related in related_content.iter() {
        let content = vdb
            .get_adjacent_chunks(related.content_id.clone(), 1, 1, related.chunk_number)
            .await?;
        context.extend(content);
    }
    Ok(context)
}
