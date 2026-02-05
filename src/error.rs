use thiserror::Error;

#[derive(Error, Debug)]
pub enum SmeshError {
    #[error("Redis connection failed: {0}")]
    RedisConnection(#[from] redis::RedisError),

    #[error("Signal not found: {signal_type} for {entity_id}")]
    SignalNotFound {
        signal_type: String,
        entity_id: String,
    },

    #[error("Concurrency limit reached: {current}/{max} calls")]
    ConcurrencyLimit { current: usize, max: usize },

    #[error("Payer rate limit: {payer_id} has {current}/{max} active calls")]
    PayerRateLimit {
        payer_id: String,
        current: usize,
        max: usize,
    },

    #[error("Max retries exceeded for inquiry {inquiry_id}: {attempts}/{max}")]
    MaxRetriesExceeded {
        inquiry_id: String,
        attempts: u32,
        max: u32,
    },

    #[error("Scheduling error: {0}")]
    Scheduling(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}
