use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};


pub async fn ask_question(
    Json(payload): Json<String>
) -> (StatusCode, Json<String>) {
    (StatusCode::OK, Json(String::from(payload)))
}