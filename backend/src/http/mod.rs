use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use crate::qa::answer_query;

pub fn router() -> Router {
    Router::new()
        .route("/api", get(|| async { "hello" }))
        .route("/api/ask", post(ask_question))
}

async fn ask_question(Json(payload): Json<String>) -> (StatusCode, Json<String>) {
    match answer_query(&payload).await {
        Ok(answer) => (StatusCode::OK, Json(answer)),
        Err(err) => {
            eprintln!("error answering question: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json("failed to answer question".to_string()),
            )
        }
    }
}
