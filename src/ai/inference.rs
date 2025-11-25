use std::sync::Arc;

use crate::data::database::VectorIndex;
use crate::utils::device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use hf_hub::{api::sync::Api, Repo};
use serde_json::json;
use tokenizers::Tokenizer;

pub trait InferenceEngine {
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String>;
}

pub struct TextGeneration {
    name: String,
    model: QMixFormer,
    device: Arc<Device>,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

async fn load_inference_model(name: &str) -> Result<(QMixFormer, Tokenizer)> {
    let api = Api::new()?.repo(Repo::model(name.to_string()));
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("model-q4k.gguf")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config = Config::v2();
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &weights_filename,
        &device(false)?,
    )?;
    let model = QMixFormer::new_v2(&config, vb)?;
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

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
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

pub async fn answer_question_with_context(
    query: &str,
    references: &Vec<VectorIndex>,
) -> Result<String> {
    let mut context = Vec::new();
    for reference in references.clone() {
        context.push(json!({
            "content": reference.content_chunk,
            "metadata": reference.metadata
        }));
    }

    let context_str = serde_json::to_string(&context)?;
    let prompt = format!(
        "You are a friendly AI agent. Context: {} Query: {}",
        context_str, query
    );

    let device = device(false)?;
    // Remove hardcoded AI name
    let mut pipeline = TextGeneration::new(
        "Demonthos/dolphin-2_6-phi-2-candle",
        398752958,
        Some(0.8),
        Some(0.7),
        1.1,
        64,
        Arc::new(device),
    )
    .await?;

    let response = pipeline.run(&prompt, 400).map_err(|e| {
        eprintln!("error generating response: {}", e);
        e
    })?;
    Ok(response)
}
