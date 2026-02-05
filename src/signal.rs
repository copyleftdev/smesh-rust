use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Signal types emitted and consumed by the SMESH mesh.
///
/// Signals decay over time and are reinforced on success,
/// enabling adaptive routing and priority decisions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SignalType {
    /// A new verification inquiry has been submitted
    InquirySubmitted { inquiry_id: Uuid },

    /// A call has been dispatched to ElevenLabs
    CallDispatched {
        inquiry_id: Uuid,
        conversation_id: String,
    },

    /// A call completed (success or failure)
    CallCompleted {
        inquiry_id: Uuid,
        conversation_id: String,
        success: bool,
        duration_secs: u32,
    },

    /// A call needs retry
    RetryNeeded {
        inquiry_id: Uuid,
        attempt: u32,
        reason: String,
    },

    /// Post-call extraction completed
    ExtractionComplete {
        inquiry_id: Uuid,
        confidence: f64,
        fields_extracted: u32,
    },

    /// Verification result delivered to provider
    ResultDelivered {
        inquiry_id: Uuid,
        provider_id: String,
    },

    /// Payer-level signal (adaptive learning)
    PayerSignal {
        payer_id: String,
        signal_name: String,
        value: f64,
    },
}

/// A timestamped signal with strength (for decay/reinforcement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,
    pub signal_type: SignalType,
    pub strength: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

impl Signal {
    pub fn new(signal_type: SignalType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            signal_type,
            strength: 1.0,
            created_at: now,
            updated_at: now,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Apply time-based decay to signal strength.
    pub fn decay(&mut self, rate: f64) {
        let age_secs = (Utc::now() - self.updated_at).num_seconds() as f64;
        self.strength *= (-rate * age_secs / 3600.0).exp();
        self.updated_at = Utc::now();
    }

    /// Reinforce signal strength (capped at 1.0).
    pub fn reinforce(&mut self, amount: f64) {
        self.strength = (self.strength + amount).min(1.0);
        self.updated_at = Utc::now();
    }
}
