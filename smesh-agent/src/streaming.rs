//! SSE (Server-Sent Events) streaming support for Claude API
//!
//! This module provides parsing utilities for Claude's streaming responses.

use crate::backend::{
    ContentBlock, ContentDelta, LlmError, MessageDeltaData, StopReason, StreamError, StreamEvent,
    StreamMessageStart, StreamUsage,
};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::Value;

// ============================================================================
// SSE Event Types (Claude-specific)
// ============================================================================

/// SSE event from Claude streaming API
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseEvent {
    MessageStart {
        message: SseMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: SseContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: SseDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: SseMessageDelta,
        usage: SseUsage,
    },
    MessageStop,
    Ping,
    Error {
        error: SseError,
    },
}

#[derive(Debug, Deserialize)]
pub struct SseMessageStart {
    pub id: String,
    pub model: String,
    pub role: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
pub struct SseMessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SseUsage {
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct SseError {
    pub message: String,
}

// ============================================================================
// Parsing Functions
// ============================================================================

/// Parse an SSE event string into a StreamEvent
pub fn parse_sse_event(event_str: &str) -> Option<StreamEvent> {
    let mut event_type = None;
    let mut data = None;

    for line in event_str.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event_type = Some(rest.trim());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data = Some(rest.trim());
        }
    }

    let data = data?;

    match event_type? {
        "message_start" => {
            let sse: SseEvent = serde_json::from_str(data).ok()?;
            if let SseEvent::MessageStart { message } = sse {
                Some(StreamEvent::MessageStart {
                    message: StreamMessageStart {
                        id: message.id,
                        model: message.model,
                        role: message.role,
                    },
                })
            } else {
                None
            }
        }
        "content_block_start" => {
            let sse: SseEvent = serde_json::from_str(data).ok()?;
            if let SseEvent::ContentBlockStart {
                index,
                content_block,
            } = sse
            {
                let block = match content_block {
                    SseContentBlock::Text { text } => ContentBlock::Text { text },
                    SseContentBlock::ToolUse { id, name, input } => {
                        ContentBlock::ToolUse { id, name, input }
                    }
                };
                Some(StreamEvent::ContentBlockStart {
                    index,
                    content_block: block,
                })
            } else {
                None
            }
        }
        "content_block_delta" => {
            let sse: SseEvent = serde_json::from_str(data).ok()?;
            if let SseEvent::ContentBlockDelta { index, delta } = sse {
                let delta = match delta {
                    SseDelta::TextDelta { text } => ContentDelta::TextDelta { text },
                    SseDelta::InputJsonDelta { partial_json } => {
                        ContentDelta::InputJsonDelta { partial_json }
                    }
                };
                Some(StreamEvent::ContentBlockDelta { index, delta })
            } else {
                None
            }
        }
        "content_block_stop" => {
            let sse: SseEvent = serde_json::from_str(data).ok()?;
            if let SseEvent::ContentBlockStop { index } = sse {
                Some(StreamEvent::ContentBlockStop { index })
            } else {
                None
            }
        }
        "message_delta" => {
            let sse: SseEvent = serde_json::from_str(data).ok()?;
            if let SseEvent::MessageDelta { delta, usage } = sse {
                Some(StreamEvent::MessageDelta {
                    delta: MessageDeltaData {
                        stop_reason: delta.stop_reason.map(|s| parse_stop_reason(&s)),
                    },
                    usage: Some(StreamUsage {
                        output_tokens: usage.output_tokens,
                    }),
                })
            } else {
                None
            }
        }
        "message_stop" => Some(StreamEvent::MessageStop),
        "ping" => Some(StreamEvent::Ping),
        "error" => {
            let sse: SseEvent = serde_json::from_str(data).ok()?;
            if let SseEvent::Error { error } = sse {
                Some(StreamEvent::Error {
                    error: StreamError {
                        message: error.message,
                    },
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Parse stop reason string to StopReason enum
pub fn parse_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

/// Collect streaming text content into a complete string
pub async fn collect_text<S>(stream: S) -> Result<String, LlmError>
where
    S: Stream<Item = Result<StreamEvent, LlmError>>,
{
    futures::pin_mut!(stream);
    let mut text = String::new();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text: chunk },
                ..
            } => {
                text.push_str(&chunk);
            }
            StreamEvent::Error { error } => {
                return Err(LlmError::GenerationFailed(error.message));
            }
            _ => {}
        }
    }

    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_message_start() {
        let event = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"model\":\"claude-3\",\"role\":\"assistant\"}}";
        let parsed = parse_sse_event(event);
        assert!(parsed.is_some());
        if let Some(StreamEvent::MessageStart { message }) = parsed {
            assert_eq!(message.id, "msg_123");
            assert_eq!(message.model, "claude-3");
        } else {
            panic!("Expected MessageStart event");
        }
    }

    #[test]
    fn test_parse_text_delta() {
        let event = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}";
        let parsed = parse_sse_event(event);
        assert!(parsed.is_some());
        if let Some(StreamEvent::ContentBlockDelta { index, delta }) = parsed {
            assert_eq!(index, 0);
            if let ContentDelta::TextDelta { text } = delta {
                assert_eq!(text, "Hello");
            } else {
                panic!("Expected TextDelta");
            }
        } else {
            panic!("Expected ContentBlockDelta event");
        }
    }

    #[test]
    fn test_parse_message_stop() {
        let event = "event: message_stop\ndata: {\"type\":\"message_stop\"}";
        let parsed = parse_sse_event(event);
        assert!(matches!(parsed, Some(StreamEvent::MessageStop)));
    }

    #[test]
    fn test_parse_ping() {
        let event = "event: ping\ndata: {\"type\":\"ping\"}";
        let parsed = parse_sse_event(event);
        assert!(matches!(parsed, Some(StreamEvent::Ping)));
    }

    #[test]
    fn test_stop_reason_parsing() {
        assert_eq!(parse_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(parse_stop_reason("stop_sequence"), StopReason::StopSequence);
        assert_eq!(parse_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(parse_stop_reason("unknown"), StopReason::EndTurn);
    }
}
