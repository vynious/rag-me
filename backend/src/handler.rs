use anyhow::{Error, Ok};
use axum::{http::StatusCode, Json};

use crate::{bert::get_embeddings, database::{get_related_chunks, Content, VectorIndex}, inference::answer_question_with_context};


pub async fn ask_question_api(
    Json(payload): Json<String>
) -> (StatusCode, Json<String>) {
    (StatusCode::OK, Json(String::from(payload)))
}


// will not terminate the main thread
pub async fn ask_question_cli(
    query: &str
) -> Result<String, Error> {
    let query_embedding: Vec<f32> = get_embeddings(query).await?.reshape((384,))?.to_vec1()?;
    let related_content = get_related_chunks(query_embedding).await?;
    let mut context: Vec<VectorIndex> = vec![];
    for related in related_content.iter() {
        let content = related.get_adjacent_chunks(1, 1).await?;
        context.extend(content);
    }
    let answer = answer_question_with_context(query, &context).await?;
    Ok(answer)
}