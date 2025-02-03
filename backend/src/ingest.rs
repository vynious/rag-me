use crate::database::process_content;
use anyhow::{anyhow, Context, Ok};
use serde_json::json;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};
use surrealdb::Datetime;

// 1. take in the path name
// 2. open the file
// 3. parse the content in the file
// 4. insert the content
pub async fn ingest_via_txt(path: &PathBuf) -> anyhow::Result<()> {
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
    let content = process_content(
        &file_name,
        &content,
        json!({"source": file_name, "upload_time": Datetime::default()}),
    )
    .await?;

    println!("memorised {}", &content.title);
    Ok(())
}

pub async fn ingest_via_pdf(path: &PathBuf) -> anyhow::Result<()> {
    let bytes = std::fs::read(path.clone()).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();

    let file_name = path
        .file_name()
        .context("bopes")?
        .to_str()
        .context("shag")?;

    println!("processing file via pdf for {}", file_name);

    let content = process_content(
        &file_name,
        &out,
        json!({"source": file_name, "upload_time": Datetime::default()}),
    )
    .await?;

    println!("memorised {}", content.title);
    Ok(())
}
