use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::HashMap, fs::File};

use crate::utils::device;
use anyhow::{Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::olmo;
use hf_hub::{api::sync::Api, Repo};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct SafetensorsIndex {
    // hf format
    weight_map: HashMap<String, String>,
}

pub trait InferenceEngine {
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String>;
}

pub struct TextGeneration {
    name: String,
    model: olmo::Model,
    device: Arc<Device>,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

async fn load_inference_model(name: &str) -> Result<(olmo::Model, tokenizers::Tokenizer)> {
    let api = Api::new()?.repo(Repo::model(name.to_string()));

    // get tokenizer
    let tokenizer_path = api.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // get config
    let config_path = api.get("config.json")?;
    let config_file = File::open(&config_path)?;
    let config: olmo::Config =
        serde_json::from_reader(config_file).context("failed to parse config.json")?;

    // load the index JSON
    let index_path = api.get("model.safetensors.index.json")?;
    let index_file = File::open(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_reader(index_file)
        .context("failed to parse model.safetensors.index.json")?;

    // collect unique shard filenames from the weight_map
    let mut shard_names: HashSet<String> = HashSet::new();
    for fname in index.weight_map.values() {
        shard_names.insert(fname.clone());
    }

    // download each shard and build the Vec<PathBuf> for VarBuilder
    let mut shard_paths: Vec<PathBuf> = Vec::new();
    for fname in shard_names {
        let p = api.get(&fname)?;
        shard_paths.push(p);
    }

    shard_paths.sort();

    // build VarBuilder from all shards
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, DType::F16, &device(false)?)? };

    // init the model
    let model = olmo::Model::new(&config, vb)?;

    Ok((model, tokenizer))
}
impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        name: &str,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: Arc<Device>,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        let (model, tokenizer) = load_inference_model(name)
            .await
            .expect("Failed to load inference model");
        Ok(Self {
            name: name.to_string(),
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device,
        })
    }
}

impl InferenceEngine for TextGeneration {
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        println!("Encoded tokens: {:?}", tokens.get_ids());
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        let start_gen = std::time::Instant::now();

        let mut response = String::new();

        self.model.clear_kv_cache();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let seqlen_offset = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, seqlen_offset)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            println!("Generated token: {}", next_token);
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == 198 {
                println!("EOS token generated, stopping generation");
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            response += &token;
        }
        let dt = start_gen.elapsed();
        println!("Final response: {}", response);
        Ok(response.trim().to_string())
    }
}
