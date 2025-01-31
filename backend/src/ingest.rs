use std::{fs::File, io::{BufRead, BufReader}, path::PathBuf};

use anyhow::{anyhow, Context, Ok};





// 1. take in the path name 
// 2. open the file
// 3. parse the content in the file
// 4. insert the content
pub async fn ingest_via_txt(path: PathBuf) -> anyhow::Result<()> {
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
    Ok(())

}