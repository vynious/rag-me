use anyhow::Context;
use serde_json::json;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::Arc,
};
use surrealdb::Datetime;

use crate::data::database::VDB;

// 1. take in the path name
// 2. open the file
// 3. parse the content in the file
// 4. insert the content
pub async fn ingest_via_txt(vdb: &Arc<VDB>, path: &PathBuf) -> anyhow::Result<()> {
    let file_name = path
        .file_name()
        .context("bopes")?
        .to_str()
        .context("shag")?
        .to_string();

    let file = File::open(path).context("unable to open file").unwrap();
    let reader = BufReader::new(file);
    let content = reader
        .lines()
        .map(|l| l.unwrap())
        .collect::<Vec<String>>()
        .join("\n");

    // content now is continous with `\n`
    let _content = vdb
        .process_content(
            &file_name,
            &content,
            json!({"source": file_name, "upload_time": Datetime::default()}),
        )
        .await?;
    Ok(())
}

pub async fn ingest_via_pdf(vdb: &Arc<VDB>, path: &PathBuf) -> anyhow::Result<()> {
    let bytes = std::fs::read(path.clone()).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();

    let file_name = path
        .file_name()
        .context("bopes")?
        .to_str()
        .context("shag")?;

    let _content = vdb
        .process_content(
            &file_name,
            &out,
            json!({"source": file_name, "upload_time": Datetime::default()}),
        )
        .await?;
    Ok(())
}
