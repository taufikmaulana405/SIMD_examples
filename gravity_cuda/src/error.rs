use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] rustacuda::error::CudaError),
    #[error("invalid CUDA string: {0}")]
    CString(#[from] std::ffi::NulError),
    #[error("particle count mismatch: backend has {active}, parameters specify {requested}")]
    CountMismatch { active: usize, requested: usize },
}
