use anyhow::{Error, Ok};
use axum::{http::StatusCode, Json};

use crate::{
    bert::get_embeddings,
    database::{get_related_chunks, VectorIndex},
    inference::answer_question_with_context,
};

pub async fn ask_question_api(Json(payload): Json<String>) -> (StatusCode, Json<String>) {
    let context = get_context_for_query(&payload).await.unwrap();
    let answer = answer_question_with_context(&payload, &context).await.unwrap();
    (StatusCode::OK, Json(String::from(answer)))
}

pub async fn ask_question_cli(query: &str) -> Result<String, Error> {
    let context = get_context_for_query(query).await?;
    let answer = answer_question_with_context(query, &context).await?;
    Ok(answer)
}


pub async fn get_context_for_query(query: &str) -> Result<Vec<VectorIndex>, Error> {
    let query_embedding: Vec<f32> = get_embeddings(query).await?.reshape((384,))?.to_vec1()?;
    let related_content = get_related_chunks(query_embedding).await?;
    let mut context: Vec<VectorIndex> = vec![];
    for related in related_content.iter() {
        let content = related.get_adjacent_chunks(1, 1).await?;
        context.extend(content);
    }
    Ok(context)
}