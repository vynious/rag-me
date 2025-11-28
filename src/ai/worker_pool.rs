use candle_core::Device;
use std::{fmt, sync::Arc};
use tokio::{
    sync::{mpsc, oneshot},
    task::spawn_blocking,
};

use crate::ai::inference::{InferenceEngine, TextGeneration};

struct InferenceJob {
    prompt: String,
    sample_len: usize,
    reply_tx: oneshot::Sender<InferenceResult>,
}

pub struct InferenceResult(String);
impl fmt::Display for InferenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct Worker {
    id: usize,
    inference_engine: Box<dyn InferenceEngine + Send + 'static>,
    rx: mpsc::Receiver<InferenceJob>,
}

pub struct WorkerPool {
    workers: Vec<mpsc::Sender<InferenceJob>>,
    next: usize,
}

impl Worker {
    pub fn new(
        id: usize,
        buffer: usize,
        inference_engine: Box<dyn InferenceEngine + Send + 'static>,
    ) -> (Self, mpsc::Sender<InferenceJob>) {
        let (tx, rx) = mpsc::channel(buffer);
        (
            Self {
                id,
                inference_engine,
                rx,
            },
            tx,
        )
    }

    pub fn run(mut self) {
        loop {
            // receive from queue
            let job = match self.rx.blocking_recv() {
                Some(job) => job,
                None => continue,
            };

            let result = self.inference_engine.run(&job.prompt, job.sample_len);
            let inference_result: String = match result {
                Ok(result) => result,
                Err(e) => format!("id: {}, inference error: {}", self.id, e),
            };
            // send back to oneshot channel
            let _ = job.reply_tx.send(InferenceResult(inference_result));
        }
    }
}

impl WorkerPool {
    pub async fn new(
        size: usize,
        buffer: usize,
        device: Arc<Device>,
        name: &str,
    ) -> anyhow::Result<Self> {
        let mut workers: Vec<mpsc::Sender<InferenceJob>> = Vec::with_capacity(size);
        for i in 0..size {
            let inference_service = TextGeneration::new(
                name,
                398752958,
                Some(0.8),
                Some(0.7),
                1.1,
                64,
                device.clone(),
            )
            .await?;
            let (worker, tx) = Worker::new(i, buffer, Box::new(inference_service));
            spawn_blocking(move || worker.run());
            workers.push(tx);
        }
        Ok(Self { workers, next: 0 })
    }

    pub async fn accept(
        &mut self,
        prompt: String,
        sample_len: usize,
    ) -> anyhow::Result<InferenceResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let job = InferenceJob {
            prompt,
            sample_len,
            reply_tx,
        };
        let idx = self.next % self.workers.len();
        self.next = self.next.wrapping_add(1);
        self.workers[idx].send(job).await?;
        let result = reply_rx.await?;
        Ok(result)
    }
}
