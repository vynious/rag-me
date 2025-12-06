use candle_core::Device;
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot},
    task::spawn_blocking,
    time::Instant,
};

use crate::ai::inference::{InferenceEngine, TextGeneration};

const MAX_SESSION: usize = 10;
const SESSION_TTL: Duration = Duration::from_secs(30 * 60);

pub struct InferenceJob {
    prompt: String,
    session_id: String,
    sample_len: usize,
    reply_tx: oneshot::Sender<InferenceResult>,
}

#[derive(Debug)]
pub struct InferenceResult(pub String);
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

struct Sticky {
    worker: usize,
    last_used: Instant,
}

pub struct WorkerPool {
    workers: Vec<mpsc::Sender<InferenceJob>>,
    pub session_sticky_map: HashMap<String, Sticky>,
    session_order: VecDeque<String>, // LRU cache
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
                None => break,
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
        Ok(Self {
            workers,
            session_order: VecDeque::new(),
            session_sticky_map: HashMap::new(),
            next: 0,
        })
    }

    // updating LRU Cache
    fn hit_session(&mut self, session_id: &str, worker: usize) {
        let now = Instant::now();
        if !self.session_sticky_map.contains_key(session_id)
            && self.session_order.len() >= MAX_SESSION
        {
            if let Some(old) = self.session_order.pop_front() {
                self.session_sticky_map.remove(&old);
            }
        }

        self.session_sticky_map.insert(
            session_id.to_string(),
            Sticky {
                worker,
                last_used: now,
            },
        );
        self.session_order.retain(|s| s != session_id);
        self.session_order.push_back(session_id.to_string());
    }

    // prune stale
    fn prune_stale(&mut self) {
        let now = Instant::now();
        loop {
            let Some(key) = self.session_order.pop_front() else {
                break;
            };
            match self.session_sticky_map.get(&key) {
                Some(sticky) if now.duration_since(sticky.last_used) > SESSION_TTL => {
                    self.session_sticky_map.remove(&key);
                    continue;
                }
                Some(_) => {
                    self.session_order.push_back(key);
                    break;
                }
                None => {
                    continue;
                }
            }
        }
    }

    pub async fn accept(
        &mut self,
        prompt: String,
        sample_len: usize,
        session_id: &str,
    ) -> anyhow::Result<InferenceResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let job = InferenceJob {
            session_id: session_id.to_string(),
            prompt,
            sample_len,
            reply_tx,
        };

        let worker_id = match self.session_sticky_map.get(session_id) {
            // enable sticky session for workers
            Some(sticky) => sticky.worker,
            None => {
                // load balance
                let idx = self.next % self.workers.len();
                self.next = self.next.wrapping_add(1);
                idx
            }
        };
        self.workers[worker_id].send(job).await?;
        let result = reply_rx.await?;
        Ok(result)
    }
}
