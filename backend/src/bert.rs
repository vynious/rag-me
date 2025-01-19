use candle_core::Tensor;
use anyhow::{Context, Error as E, Result};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, DTYPE, Config};
use hf_hub::{api::tokio::Api, Repo};
use tokenizers::{PaddingParams, Tokenizer};
use tokio::sync::OnceCell;

use crate::utils::device;

static AI: OnceCell<(BertModel, Tokenizer)> = OnceCell::const_new();

pub async fn load_model() -> Result<(BertModel, Tokenizer)> {
    // Initialize the API for Hugging Face Hub and fetch model files
    let api = Api::new()?.repo(Repo::model("sentence-transformers/all-MiniLM-L6-v2".to_string()));
    let config_filename = api.get("config.json").await?;
    let tokenizer_filename = api.get("tokenizer.json").await?;
    let weights_filename = api.get("pytorch_model.bin").await?;

    // Load model configuration from the downloaded JSON file
    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;

    // Load the tokenizer
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // Load the model weights and initialize the BERT model
    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &device(false)?)?;
    let model = BertModel::load(vb, &config)?;

    // Set padding strategy for the tokenizer
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    // Return the model and tokenizer
    Ok((model, tokenizer))
}



pub async fn get_embeddings(sentence: &str) -> Result<Tensor> {
    let (model, tokenizer) = AI.get_or_try_init(|| async { 
        load_model().await
    })
    .await.expect("Failed to get AI");

    // drop any non-ascii characters
    let sentence = sentence
        .chars()
        .filter(|c| c.is_ascii())
        .collect::<String>();

    let tokens = tokenizer
        .encode_batch(vec![sentence], true)
        .map_err(E::msg)
        .context("Unable to encode sentence")?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device(false)?)?)
        })
        .collect::<Result<Vec<_>>>()
        .context("Unable to get token ids")?;

    let token_ids = Tensor::stack(&token_ids, 0).context("Unable to stack token ids")?;
    let token_type_ids = token_ids
        .zeros_like()
        .context("Unable to get token type ids")?;


    let embeddings = model
        .forward(&token_ids, &token_type_ids, None)
        .context("Unable to get embeddings")?;

    let (_n_sentence, n_tokens, _hidden_size) = embeddings
        .dims3()
        .context("Unable to get embeddings dimensions")?;
    let embeddings =
        (embeddings.sum(1)? / (n_tokens as f64)).context("Unable to get embeddings sum")?;
    let embeddings = embeddings
        .broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)
        .context("Unable to get embeddings broadcast div")?;

    Ok(embeddings)
}