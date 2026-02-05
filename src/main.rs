mod config;
mod error;
mod signal;
mod store;
mod orchestrator;
mod scheduler;

use anyhow::Result;
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    tracing::info!("SMESH-Rust starting");

    let config = config::SmeshConfig::load()?;
    tracing::info!(
        concurrency = config.concurrency.max_concurrent_calls,
        "Configuration loaded"
    );

    let store = store::RedisSignalStore::connect(&config.redis.url).await?;
    tracing::info!("Redis signal store connected");

    let orchestrator = orchestrator::Orchestrator::new(config, store);
    orchestrator.run().await?;

    Ok(())
}
