use candle_core::Tensor;
use anyhow::{Context, Error as E, Result};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, DTYPE, Config};
use hf_hub::{api::tokio::Api, Repo};
use tokenizers::{PaddingParams, Tokenizer};
use tokio::sync::OnceCell;

use crate::utils::device;

static AI: OnceCell<(BertModel, Tokenizer)> = OnceCell::const_new();

pub async fn load_embedding_model() -> Result<(BertModel, Tokenizer)> {
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


/// get_embeddings takes in a string and returns the result tensor
pub async fn get_embeddings(sentence: &str) -> Result<Tensor> {
    // Initialize or retrieve the AI model and tokenizer.
    // `get_or_try_init` ensures the model and tokenizer are loaded only once and reused.
    let (model, tokenizer) = AI.get_or_try_init(|| async { 
        load_embedding_model().await // Load the model and tokenizer asynchronously
    })
    .await.expect("Failed to get AI"); // Panic if the model/tokenizer fails to load

    // Preprocess the input sentence by filtering out non-ASCII characters.
    // This ensures the input is compatible with the tokenizer.
    let sentence = sentence
        .chars()
        .filter(|c| c.is_ascii()) // Keep only ASCII characters
        .collect::<String>(); // Convert the filtered characters back into a string

    // Tokenize the sentence using the tokenizer.
    // `encode_batch` tokenizes a batch of sentences (here, just one sentence).
    let tokens = tokenizer
        .encode_batch(vec![sentence], true) // Tokenize the sentence
        .map_err(E::msg) // Convert errors to a custom error type
        .context("Unable to encode sentence")?; // Add context to the error if tokenization fails

    // Convert the tokenized input into tensor format.
    // Each token is mapped to its corresponding ID and converted into a tensor.
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec(); // Get token IDs as a vector
            Ok(Tensor::new(tokens.as_slice(), &device(false)?)?) // Create a tensor from the token IDs
        })
        .collect::<Result<Vec<_>>>() // Collect all token tensors into a vector
        .context("Unable to get token ids")?; // Add context to the error if tensor creation fails

    // Stack the token tensors into a single tensor.
    // This is necessary for batch processing (even if there's only one sentence).
    let token_ids = Tensor::stack(&token_ids, 0).context("Unable to stack token ids")?;

    // Create a tensor of zeros with the same shape as `token_ids`.
    // This is used as `token_type_ids` (typically for segmenting sentences in models like BERT).
    let token_type_ids = token_ids
        .zeros_like()
        .context("Unable to get token type ids")?;

    // Pass the token IDs and token type IDs through the model to get embeddings.
    let embeddings = model
        .forward(&token_ids, &token_type_ids, None) // Forward pass through the model
        .context("Unable to get embeddings")?; // Add context to the error if the forward pass fails

    // Get the dimensions of the embeddings tensor.
    // The shape is (number of sentences, number of tokens, hidden size).
    let (_n_sentence, n_tokens, _hidden_size) = embeddings
        .dims3()
        .context("Unable to get embeddings dimensions")?;

    // Compute the mean embedding for each sentence by summing over the tokens
    // and dividing by the number of tokens.
    let embeddings =
        (embeddings.sum(1)? / (n_tokens as f64)).context("Unable to get embeddings sum")?;

    // Normalize the embeddings by dividing each embedding vector by its L2 norm.
    // This ensures the embeddings are unit vectors, which is often useful for similarity calculations.
    let embeddings = embeddings
        .broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)
        .context("Unable to get embeddings broadcast div")?;

    // Return the final normalized embeddings tensor.
    Ok(embeddings)
}