use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::Device;
use std::env;
use std::path::PathBuf;

// https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn get_current_working_dir() -> std::io::Result<PathBuf> {
    env::current_dir()
}
