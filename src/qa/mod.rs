use anyhow::Error;

use crate::{
    ai::embedding::get_embeddings,
    ai::inference::answer_question_with_context,
    data::database::{get_related_chunks, VectorIndex},
};

pub async fn answer_query(query: &str) -> Result<String, Error> {
    let context = build_context_for_query(query).await?;
    let answer = answer_question_with_context(query, &context).await?;
    Ok(answer)
}

pub async fn build_context_for_query(query: &str) -> Result<Vec<VectorIndex>, Error> {
    let query_embedding: Vec<f32> = get_embeddings(query).await?.reshape((384,))?.to_vec1()?;
    let related_content = get_related_chunks(query_embedding).await?;
    let mut context: Vec<VectorIndex> = vec![];
    for related in related_content.iter() {
        let content = related.get_adjacent_chunks(1, 1).await?;
        context.extend(content);
    }
    Ok(context)
}
