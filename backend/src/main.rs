use axum::{routing::get, routing::post, Router};
mod handler;

#[tokio::main]
async fn main() {
    let app: Router = Router::new()
        .route("/api", get(|| async { "hello" }))
        .route("/api/ask", post(handler::ask_question));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}  
