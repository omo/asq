// Allow dead code on deserialization-only fields (used by serde at runtime)
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest {
    pub(crate) stream: bool,
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_completion_tokens: Option<u32>,
    pub(crate) messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub(crate) role: String,
    pub(crate) content: String,
}

// ─── SSE chunk types (OpenAI-compatible chat completions format) ──────────

/// A single SSE chunk from Brave's streaming chat completions.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct SseChunk {
    #[serde(default)]
    pub(crate) choices: Vec<Choice>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct Choice {
    #[serde(default)]
    pub(crate) delta: Delta,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct Delta {
    #[serde(default)]
    pub(crate) content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Request serialization tests ─────────────────────────────────────────

    #[test]
    fn chat_request_serializes_correctly() {
        let req = ChatRequest {
            stream: true,
            model: "brave-pro".into(),
            max_completion_tokens: Some(4096),
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""stream":true"#));
        assert!(json.contains(r#""model":"brave-pro""#));
        assert!(json.contains(r#""max_completion_tokens":4096"#));
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""content":"hello""#));
    }

    #[test]
    fn message_fields_roundtrip() {
        let msg = Message {
            role: "assistant".into(),
            content: "some text".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"assistant","content":"some text"}"#);
    }

    // ── SSE chunk deserialization tests ─────────────────────────────────────

    #[test]
    fn sse_chunk_with_text_delta() {
        let json = r#"{"choices":[{"delta":{"content":"Hello "}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(
            chunk.choices[0].delta.content.as_deref(),
            Some("Hello ")
        );
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn sse_chunk_with_finish_reason() {
        let json = r#"{"choices":[{"finish_reason":"stop"}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert!(chunk.choices[0].delta.content.is_none());
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("stop")
        );
    }

    #[test]
    fn sse_chunk_with_both_text_and_finish() {
        let json =
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(
            chunk.choices[0].delta.content.as_deref(),
            Some("hello")
        );
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("stop")
        );
    }

    #[test]
    fn sse_chunk_multiple_choices() {
        let json =
            r#"{"choices":[{"delta":{"content":"a"}},{"delta":{"content":"b"}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 2);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("a"));
        assert_eq!(chunk.choices[1].delta.content.as_deref(), Some("b"));
    }

    #[test]
    fn sse_chunk_empty_choices() {
        let json = r#"{"choices":[]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices.is_empty());
    }

    #[test]
    fn sse_chunk_missing_choices_defaults_to_empty() {
        let json = r#"{"id":"abc123"}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices.is_empty());
    }

    #[test]
    fn sse_chunk_extra_fields_ignored() {
        let json = r#"{"id":"1","object":"chat.completion.chunk","created":1694268190,"model":"brave-pro","choices":[{"index":0,"delta":{"content":"hi","role":"assistant"},"finish_reason":null}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }
}
