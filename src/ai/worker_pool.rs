use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};

use crate::ai::inference::InferenceEngine;

struct InferenceJob {
    prompt: String,
    sample_len: usize,
}

struct InferenceResult(String);

pub struct Worker {
    id: usize,
    inference_engine: Box<dyn InferenceEngine + Send + Sync>,
    shared_rx: Arc<Mutex<mpsc::Receiver<InferenceJob>>>,
    result_tx: mpsc::Sender<InferenceResult>,
}

pub struct WorkerPool {
    job_tx: mpsc::Sender<InferenceJob>,
    result_rx: Arc<Mutex<mpsc::Receiver<InferenceResult>>>,
}

impl Worker {
    pub async fn run(mut self) {
        loop {
            let job_opt = {
                let mut rx = self.shared_rx.lock().await;
                rx.recv().await
            };

            let job = match job_opt {
                Some(job) => job,
                None => break,
            };

            let result = self.inference_engine.run(&job.prompt, job.sample_len);
            let inference_result: String = match result {
                Ok(result) => result,
                Err(e) => format!("id: {}, inference error: {}", self.id, e),
            };
            if let Err(e) = self.result_tx.send(InferenceResult(inference_result)).await {
                eprintln!(
                    "Worker {} received an error while sending inference result, error: {}",
                    self.id, e
                );
            }
        }
    }
}
