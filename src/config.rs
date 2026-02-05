use serde::Deserialize;
use anyhow::Result;

#[derive(Debug, Deserialize, Clone)]
pub struct SmeshConfig {
    pub redis: RedisConfig,
    pub concurrency: ConcurrencyConfig,
    pub retry: RetryConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RedisConfig {
    pub url: String,
    #[serde(default = "default_key_prefix")]
    pub key_prefix: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ConcurrencyConfig {
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_calls: usize,
    #[serde(default = "default_max_per_payer")]
    pub max_per_payer: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RetryConfig {
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_base_delay_secs")]
    pub base_delay_secs: u64,
    #[serde(default = "default_max_delay_secs")]
    pub max_delay_secs: u64,
    #[serde(default = "default_jitter")]
    pub jitter: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SignalConfig {
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f64,
    #[serde(default = "default_reinforce_amount")]
    pub reinforce_amount: f64,
    #[serde(default = "default_success_threshold")]
    pub success_threshold: f64,
}

fn default_key_prefix() -> String { "smesh:".into() }
fn default_max_concurrent() -> usize { 10 }
fn default_max_per_payer() -> usize { 3 }
fn default_max_retries() -> u32 { 3 }
fn default_base_delay_secs() -> u64 { 60 }
fn default_max_delay_secs() -> u64 { 900 }
fn default_jitter() -> bool { true }
fn default_decay_rate() -> f64 { 0.05 }
fn default_reinforce_amount() -> f64 { 0.3 }
fn default_success_threshold() -> f64 { 0.7 }

impl SmeshConfig {
    pub fn load() -> Result<Self> {
        let config_path = std::env::var("SMESH_CONFIG")
            .unwrap_or_else(|_| "config.toml".into());
        let content = std::fs::read_to_string(&config_path)?;
        let config: SmeshConfig = toml::from_str(&content)?;
        Ok(config)
    }
}
