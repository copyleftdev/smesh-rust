//! Turning an incoming A2A message into a mesh task.
//!
//! An A2A message carries free text and optional structured parts; the mesh
//! needs a task type, because the task type is what agents claim on. The type
//! is taken from the most explicit source available:
//!
//! 1. `taskType` in the message metadata,
//! 2. `taskType` in a data part,
//! 3. keywords in the text.
//!
//! The keyword rule is a fallback for clients that know nothing of SMESH. It is
//! deliberately simple and deterministic, so a client can predict it.

use a2a::{Message, PartContent};
use serde_json::Value;
use smesh_agent::{TaskDefinition, TaskType};

/// Metadata key a client sets to choose the task type explicitly.
pub const TASK_TYPE_KEY: &str = "taskType";
/// Metadata key a client sets to choose the task priority (0.0 to 1.0).
pub const PRIORITY_KEY: &str = "priority";

/// Priority used when the message does not set one.
pub const DEFAULT_PRIORITY: f64 = 0.5;

/// Why a message could not become a task.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ClassifyError {
    /// The message has no text to work on.
    #[error("message has no text part")]
    NoText,
    /// The client named a task type the mesh does not have.
    #[error("unknown task type {0:?}; expected one of code_write, code_review, testing, documentation, analysis")]
    UnknownTaskType(String),
}

/// Build the mesh task for an A2A message.
pub fn task_from_message(message: &Message) -> Result<TaskDefinition, ClassifyError> {
    let text = message_text(message).ok_or(ClassifyError::NoText)?;

    let explicit = message
        .metadata
        .as_ref()
        .and_then(|m| m.get(TASK_TYPE_KEY))
        .or_else(|| data_field(message, TASK_TYPE_KEY))
        .and_then(Value::as_str);

    let task_type = match explicit {
        Some(name) => name
            .parse::<TaskType>()
            .map_err(|_| ClassifyError::UnknownTaskType(name.to_string()))?,
        None => infer_task_type(&text),
    };

    let priority = message
        .metadata
        .as_ref()
        .and_then(|m| m.get(PRIORITY_KEY))
        .or_else(|| data_field(message, PRIORITY_KEY))
        .and_then(Value::as_f64)
        .map(|p| p.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_PRIORITY);

    Ok(TaskDefinition {
        task_type: task_type.as_str().to_string(),
        description: text,
        priority,
    })
}

/// All text parts of a message, joined by blank lines. `None` if there are none
/// or they are all whitespace.
pub fn message_text(message: &Message) -> Option<String> {
    let text = message
        .parts
        .iter()
        .filter_map(|p| p.as_text())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!text.trim().is_empty()).then_some(text)
}

fn data_field<'a>(message: &'a Message, key: &str) -> Option<&'a Value> {
    message.parts.iter().find_map(|p| match &p.content {
        PartContent::Data(v) => v.get(key),
        _ => None,
    })
}

/// Guess a task type from text. Checked in a fixed order, first match wins;
/// anything unmatched is analysis.
pub fn infer_task_type(text: &str) -> TaskType {
    let lower = text.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| lower.contains(w));

    if has(&["review", "audit", "critique"]) {
        TaskType::CodeReview
    } else if has(&["test", "unit test", "coverage"]) {
        TaskType::Testing
    } else if has(&["document", "docs", "readme", "explain how to use"]) {
        TaskType::Documentation
    } else if has(&[
        "implement",
        "write a function",
        "write code",
        "refactor",
        "fix the bug",
    ]) {
        TaskType::CodeWrite
    } else {
        TaskType::Analysis
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2a::{Part, Role};
    use std::collections::HashMap;

    fn msg(parts: Vec<Part>) -> Message {
        Message::new(Role::User, parts)
    }

    #[test]
    fn test_metadata_task_type_wins_over_keywords() {
        let mut m = msg(vec![Part::text("please review this")]);
        m.metadata = Some(HashMap::from([(
            TASK_TYPE_KEY.to_string(),
            Value::from("documentation"),
        )]));
        assert_eq!(task_from_message(&m).unwrap().task_type, "documentation");
    }

    #[test]
    fn test_data_part_task_type_used_when_no_metadata() {
        let m = msg(vec![
            Part::text("anything"),
            Part::data(serde_json::json!({"taskType": "testing", "priority": 0.9})),
        ]);
        let task = task_from_message(&m).unwrap();
        assert_eq!(task.task_type, "testing");
        assert_eq!(task.priority, 0.9);
    }

    #[test]
    fn test_unknown_task_type_is_an_error_not_a_guess() {
        let mut m = msg(vec![Part::text("x")]);
        m.metadata = Some(HashMap::from([(
            TASK_TYPE_KEY.to_string(),
            Value::from("poetry"),
        )]));
        assert_eq!(
            task_from_message(&m).unwrap_err(),
            ClassifyError::UnknownTaskType("poetry".into())
        );
    }

    #[test]
    fn test_message_without_text_is_rejected() {
        let m = msg(vec![Part::data(serde_json::json!({"taskType": "testing"}))]);
        assert_eq!(task_from_message(&m).unwrap_err(), ClassifyError::NoText);
    }

    #[test]
    fn test_priority_is_clamped() {
        let mut m = msg(vec![Part::text("x")]);
        m.metadata = Some(HashMap::from([(
            PRIORITY_KEY.to_string(),
            Value::from(7.0),
        )]));
        assert_eq!(task_from_message(&m).unwrap().priority, 1.0);
    }

    #[test]
    fn test_keyword_inference() {
        assert_eq!(infer_task_type("Review my PR"), TaskType::CodeReview);
        assert_eq!(infer_task_type("add unit tests"), TaskType::Testing);
        assert_eq!(
            infer_task_type("update the README"),
            TaskType::Documentation
        );
        assert_eq!(infer_task_type("implement a parser"), TaskType::CodeWrite);
        assert_eq!(infer_task_type("why is latency high?"), TaskType::Analysis);
    }

    #[test]
    fn test_text_parts_are_joined() {
        let m = msg(vec![Part::text("a"), Part::text("b")]);
        assert_eq!(message_text(&m).as_deref(), Some("a\n\nb"));
    }
}
