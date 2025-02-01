use anyhow::{Context, Error};
use serde::{Deserialize, Serialize};
use surrealdb::{
    engine::local::{
        Db, 
        RocksDb
    }, 
    sql::{
        thing, 
        Thing
    }, 
    Datetime, 
    Surreal, 
    Uuid
};
use surrealdb::opt::auth::Root;
use tokio::sync::OnceCell;

use crate::bert::get_embeddings;

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

        Ok::<Surreal<Db>, Error>(db)
    })
    .await
    .expect("Failed to initialize the database")
}

#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct Content {
    pub id: Thing,
    pub title: String,
    pub text: String,
    pub created_at: Datetime
}

impl Content {
    #[allow(dead_code)]
    pub async fn get_vector_indexes(&self) -> Result<Vec<VectorIndex>, Error> {   
        let db = get_db().await.clone();
        let mut result = db
            .query("SELECT * FROM vector_index WHERE content_id = $content")
            .bind(("content", self.id.clone()))
            .await?;
        let vindexes: Vec<VectorIndex> = result.take(0)?;
        Ok(vindexes)
    }
}

#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct VectorIndex {
    pub id: Thing,
    pub content_id: Thing,
    pub content_chunk: String,
    pub chunk_number: u16, 
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
    pub created_at: Datetime
}

impl VectorIndex{
    #[allow(dead_code)]
    pub async fn get_content(&self) -> Result<Content,Error> {
        let db = get_db().await.clone();

        let result: Vec<Content> = db
            .select(self.content_id.to_string())
            .await?;
        
        let content = result
            .into_iter()
            .next()
            .context("No content found")?;
        Ok(content)
    }

    #[allow(dead_code)]
    pub async fn get_adjacent_chunks(
        &self,
        upper: u16, 
        lower: u16,
    ) -> Result<Vec<VectorIndex>, Error> {
        let db = get_db().await.clone();

        // guard statement to check underflow
        let start = if self.chunk_number > lower {
            self.chunk_number - lower
        } else{
            0
        };

        let mut result = db
            .query("SELECT * FROM vector_index WHERE content_id = $content AND chunk_number >= $start AND chunk_number <= $end ORDER BY chunk_number ASC")
            .bind(("content", self.content_id.clone()))
            .bind(("start", start))
            .bind(("end", self.chunk_number + upper))
            .await?;
        let vector_indexes = result.take(0)?;
        Ok(vector_indexes)
    }
}

pub async fn insert_content(
    title: &str,
    text: &str
) -> anyhow::Result<Content, Error> {
        
    let db = get_db().await.clone();
    let id = Uuid::new_v4().to_string().replace("-", "");
    let id = thing(format!("content:{}", id).as_str())?;
    let content = db
        .create(("content", id.to_string()))
        .content(Content{
                id: id.clone(),
                title: title.to_string(),
                text: text.to_string(),
                created_at: Datetime::default()
        })
        .await?
        .context("failed to insert content")?;

    Ok(content)
}


pub async fn insert_into_vdb(
    content_id: Thing,
    chunk_number : u16,
    content_chunk: &str,
    metadata: serde_json::Value
) -> anyhow::Result<VectorIndex,Error> {

    let db = get_db().await.clone();
    let id = Uuid::new_v4().to_string().replace("-", "");
    let id = thing(format!("vector_index:{}", id).as_str())?;
    let content_chunk = content_chunk
        .chars()
        .filter(|c| c.is_ascii())
        .collect::<String>();

    let content_chunk = content_chunk.trim();

    if content_chunk.is_empty() {
        return Err(anyhow::anyhow!("content chunk is empty!"))
    }

    let vector = get_embeddings(&content_chunk)
        .await?
        .reshape((384,))? // apparently 384 is the optimal for vector embeddings? idk.
        .to_vec1()?;

    let vector_index: VectorIndex = db
        .create(("vector_index", id.to_string()))
        .content(VectorIndex {
            id: id.clone(),
            content_id,
            chunk_number,
            content_chunk: content_chunk.to_string(),
            metadata,
            vector,
            created_at: Datetime::default()
        })
        .await?
        .context("unable to insert vector index")?;
    
    Ok(vector_index)

}

// vector -> key -> content 
pub async fn process_content(
    title: &str,
    text: &str,
    metadata: serde_json::Value,
) -> anyhow::Result<Content, Error> {
    // insert into content
    let content = insert_content(title, text).await?;

    // parse the chunks, split into array of strings and remove empty.
    let mut chunks = text.split("\n").collect::<Vec<&str>>();
    chunks.retain(|c| !c.is_empty());

    // recursively split the chunks into smaller chunks if the length is more than 1000.
    let mut index = 0;
    while index < chunks.len() {
        if chunks[index].len() > 1000 {
            let split_chunks = chunks[index]
                .split(".")
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .collect::<Vec<&str>>();
        
            chunks.remove(index);
            chunks.splice(index..index, split_chunks);
        } else {
            index += 1;
        }
    }

    for (i, chunk) in chunks.clone().into_iter().enumerate() {
        println!("memorizing chunk {}", i);
        let res = insert_into_vdb(content.id.clone(), i as u16, chunk, metadata.clone()).await;
        match res {
            Ok(_) => {}
            Err(e) => {
                if e.to_string().contains("content chunk is empty!") {
                    continue
                }
                println!("unable to insert vector index: {}", e)
            }
        }
    }

    Ok(content)
}

// using cosine similarity to find nearby vectors
pub async fn get_related_chunks(query: Vec<f32>) -> Result<Vec<VectorIndex>, Error> {
    let db = get_db().await.clone();
    let mut result = db
        .query("SELECT *, vector::similarity::cosine(vector, $query) AS score FROM vector_index ORDER BY score DESC LIMIT 4")
        .bind(("query", query))
        .await?;
    let vector_indexes = result.take(0)?;
    Ok(vector_indexes)
}

pub async fn get_all_content(start: u16, limit: u16) -> Result<Vec<Content>, Error> {
    let db = get_db().await.clone();
    let mut result = db
        .query("SELECT * FROM content ORDER BY created_at DESC LIMIT $limit START $start")
        .bind(("start", start))
        .bind(("limit", limit))
        .await?;
    let content: Vec<Content> = result.take(0)?;

    Ok(content)
}


// delete content by id from content and vector index table
pub async fn delete_content(id: &str) -> Result<(), Error> {
    let db = get_db().await.clone();
    let id = thing(format!("content:{}", id).as_str())?;
    
    let _ = db.query("DELETE FROM vector_index WHERE content_id = $id")
        .bind(("id", id.clone()))
        .await?.check().context("unable to delete content");

    let _ = db.query("DELETE FROM content WHERE id = $id")
        .bind(("id", id.clone()))
        .await?.check().context("Unable to delete content")?;

    Ok(())
}