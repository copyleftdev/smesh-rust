use anyhow::Result;
use redis::aio::ConnectionManager;
use uuid::Uuid;

use crate::signal::Signal;

/// Redis-backed signal store.
///
/// Stores signals as JSON in Redis sorted sets, keyed by entity.
/// Sorted set score = signal strength (enables priority queries).
pub struct RedisSignalStore {
    conn: ConnectionManager,
    prefix: String,
}

impl RedisSignalStore {
    pub async fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)?;
        let conn = ConnectionManager::new(client).await?;
        Ok(Self {
            conn,
            prefix: "smesh:".into(),
        })
    }

    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefix = prefix;
        self
    }

    fn key(&self, namespace: &str, entity_id: &str) -> String {
        format!("{}{}:{}", self.prefix, namespace, entity_id)
    }

    /// Store a signal in the sorted set for its entity.
    pub async fn emit(&mut self, namespace: &str, entity_id: &str, signal: &Signal) -> Result<()> {
        let key = self.key(namespace, entity_id);
        let json = serde_json::to_string(signal)?;
        redis::cmd("ZADD")
            .arg(&key)
            .arg(signal.strength)
            .arg(signal.id.to_string())
            .query_async::<()>(&mut self.conn)
            .await?;
        let detail_key = format!("{}signal:{}", self.prefix, signal.id);
        redis::cmd("SET")
            .arg(&detail_key)
            .arg(&json)
            .arg("EX")
            .arg(86400_u64) // 24h TTL
            .query_async::<()>(&mut self.conn)
            .await?;
        Ok(())
    }

    /// Get a signal by ID.
    pub async fn get(&mut self, signal_id: Uuid) -> Result<Option<Signal>> {
        let key = format!("{}signal:{}", self.prefix, signal_id);
        let json: Option<String> = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut self.conn)
            .await?;
        match json {
            Some(j) => Ok(Some(serde_json::from_str(&j)?)),
            None => Ok(None),
        }
    }

    /// Count active signals in a namespace for an entity.
    pub async fn count(&mut self, namespace: &str, entity_id: &str) -> Result<usize> {
        let key = self.key(namespace, entity_id);
        let count: usize = redis::cmd("ZCARD")
            .arg(&key)
            .query_async(&mut self.conn)
            .await?;
        Ok(count)
    }

    /// Get total active calls (across all payers).
    pub async fn active_call_count(&mut self) -> Result<usize> {
        let pattern = format!("{}calls:*", self.prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut self.conn)
            .await?;
        let mut total = 0usize;
        for key in &keys {
            let count: usize = redis::cmd("ZCARD")
                .arg(key)
                .query_async(&mut self.conn)
                .await?;
            total += count;
        }
        Ok(total)
    }

    /// Get active call count for a specific payer.
    pub async fn payer_call_count(&mut self, payer_id: &str) -> Result<usize> {
        self.count("calls", payer_id).await
    }

    /// Remove a signal from a sorted set.
    pub async fn remove(&mut self, namespace: &str, entity_id: &str, signal_id: Uuid) -> Result<()> {
        let key = self.key(namespace, entity_id);
        redis::cmd("ZREM")
            .arg(&key)
            .arg(signal_id.to_string())
            .query_async::<()>(&mut self.conn)
            .await?;
        let detail_key = format!("{}signal:{}", self.prefix, signal_id);
        redis::cmd("DEL")
            .arg(&detail_key)
            .query_async::<()>(&mut self.conn)
            .await?;
        Ok(())
    }
}
