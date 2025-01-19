use surrealdb::{engine::local::{Db, RocksDb}, Surreal};
use surrealdb::opt::auth::Root;
use tokio::sync::OnceCell;

static DB: OnceCell<Surreal<Db>> = OnceCell::const_new();

pub async fn get_db() -> &'static Surreal<Db> {
    DB.get_or_try_init(|| async {
        let db = Surreal::new::<RocksDb>("ragme.db")
            .await
            .expect("Unable to connect to DB");

        db.signin(Root {
            username: "root",
            password: "root",
        })
        .await
        .expect("Failed to authenticate");

        db.use_ns("rag-me")
            .use_db("content")
            .await
            .expect("Failed to switch to namespace and database");

        db.query(
            "
                DEFINE TABLE content SCHEMAFULL;
                DEFINE FIELD id ON TABLE content TYPE record;
                DEFINE FIELD title ON TABLE content TYPE string;
                DEFINE FIELD text ON TABLE content TYPE string;
                DEFINE FIELD created_at ON TABLE content TYPE datetime DEFAULT time::now();
                DEFINE INDEX contentIdIndex ON TABLE content COLUMNS id UNIQUE;
            ",
        )
        .await
        .expect("Failed to define content table");

        db.query(
            "
                DEFINE TABLE vector_index SCHEMAFULL;
                DEFINE FIELD id ON TABLE vector_index TYPE record;
                DEFINE FIELD content_id ON TABLE vector_index TYPE record<content>;
                DEFINE FIELD content_chunk ON TABLE vector_index TYPE string;
                DEFINE FIELD chunk_number ON TABLE vector_index TYPE int;
                DEFINE FIELD vector ON TABLE vector_index TYPE array<float>;
                DEFINE FIELD vector.* ON TABLE vector_index TYPE float;
                DEFINE FIELD metadata ON TABLE vector_index FLEXIBLE TYPE object;
                DEFINE FIELD created_at ON TABLE vector_index TYPE datetime DEFAULT time::now();
                DEFINE INDEX vectorIdIndex ON TABLE vector_index COLUMNS id UNIQUE;
            ",
        )
        .await
        .expect("Failed to define vector_index table");

        Ok::<_, surrealdb::Error>(db)
    })
    .await
    .expect("Failed to initialize the database")
}
