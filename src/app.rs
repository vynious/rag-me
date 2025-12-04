
use crate::{ai::AI, data::database::VDB};

use std::sync::Arc;

pub struct App {
    pub vector_db: Arc<VDB>,
    pub ai_service: Arc<AI>,
    // pub task_rx: Arc<mpsc::Receiver<String>>,
}

impl App {
    pub fn new(ai_service: Arc<AI>, vector_db: Arc<VDB>) -> Self {
        Self {
            vector_db,
            ai_service,
        }
    }
}
