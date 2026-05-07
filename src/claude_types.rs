// Allow dead code on deserialization-only fields (used by serde at runtime)
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct MessagesRequest {
    pub(crate) model: String,
    pub(crate) max_tokens: u32,
    pub(crate) stream: bool,
    pub(crate) messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub(crate) tool_type: String,
    pub(crate) name: String,
}

// ─── SSE event types ──────────────────────────────────────────────────────

/// A tagged union of all Claude SSE events, discriminated by the `type` field
/// in the JSON payload (which always matches the `event:` line).
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart,
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u32,
        delta: Delta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        index: u32,
    },
    #[serde(rename = "message_delta")]
    MessageDelta,
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

/// A content block inside a `content_block_start` event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text(TextBlock),
    #[serde(rename = "tool_use")]
    ToolUse(ToolUseBlock),
    /// Server-initiated tool use (e.g. web_search). Same shape as `tool_use`.
    #[serde(rename = "server_tool_use")]
    ServerToolUse(ToolUseBlock),
    /// Catch-all for unknown block types (e.g. web_search_tool_result).
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TextBlock {
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) citations: Vec<Citation>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolUseBlock {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) input: serde_json::Value,
}

/// A delta inside a `content_block_delta` event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum Delta {
    #[serde(rename = "text_delta")]
    Text(TextDelta),
    #[serde(rename = "citations_delta")]
    Citations(CitationsDelta),
    #[serde(rename = "thinking_delta")]
    Thinking,
    #[serde(rename = "signature_delta")]
    Signature,
    #[serde(rename = "input_json_delta")]
    InputJson,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TextDelta {
    pub(crate) text: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CitationsDelta {
    pub(crate) citation: Citation,
}

/// A web citation / source URL from a text block or citations delta.
#[derive(Debug, Deserialize)]
pub(crate) struct Citation {
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) title: Option<String>,
    #[serde(default)]
    pub(crate) start_index: Option<u32>,
    #[serde(default)]
    pub(crate) end_index: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Request serialization tests ─────────────────────────────────────────

    #[test]
    fn messages_request_serializes() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".into(),
            max_tokens: 4096,
            stream: true,
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
            tools: Some(vec![Tool {
                tool_type: "web_search_20250305".into(),
                name: "web_search".into(),
            }]),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""model":"claude-sonnet-4-6""#));
        assert!(json.contains(r#""max_tokens":4096"#));
        assert!(json.contains(r#""stream":true"#));
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""type":"web_search_20250305""#));
        assert!(json.contains(r#""name":"web_search""#));
    }

    #[test]
    fn messages_request_without_tools_skips_field() {
        let req = MessagesRequest {
            model: "m".into(),
            max_tokens: 100,
            stream: false,
            messages: vec![Message {
                role: "user".into(),
                content: "q".into(),
            }],
            tools: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
    }

    #[test]
    fn message_serializes() {
        let msg = Message {
            role: "assistant".into(),
            content: "some text".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"assistant","content":"some text"}"#);
    }

    #[test]
    fn tool_uses_type_rename() {
        let tool = Tool {
            tool_type: "custom".into(),
            name: "my_tool".into(),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(json, r#"{"type":"custom","name":"my_tool"}"#);
    }

    // ── SSE event deserialization tests ─────────────────────────────────────

    #[test]
    fn content_block_start_text_with_citations() {
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": "Hello world",
                "citations": [
                    {"url": "https://a.com", "title": "A"},
                    {"url": "https://b.com"}
                ]
            }
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockStart { index, content_block } => {
                assert_eq!(index, 0);
                match content_block {
                    ContentBlock::Text(tb) => {
                        assert_eq!(tb.text, "Hello world");
                        assert_eq!(tb.citations.len(), 2);
                        assert_eq!(tb.citations[0].url, "https://a.com");
                        assert_eq!(tb.citations[0].title.as_deref(), Some("A"));
                        assert_eq!(tb.citations[1].url, "https://b.com");
                        assert!(tb.citations[1].title.is_none());
                    }
                    _ => panic!("expected Text block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }

    #[test]
    fn content_block_start_server_tool_use() {
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "toolu_1",
                "name": "web_search",
                "input": {"query": "population Tokyo"}
            }
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockStart { index, content_block } => {
                assert_eq!(index, 0);
                match content_block {
                    ContentBlock::ServerToolUse(tu) => {
                        assert_eq!(tu.id, "toolu_1");
                        assert_eq!(tu.name, "web_search");
                    }
                    _ => panic!("expected ServerToolUse block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }

    #[test]
    fn content_block_start_tool_use() {
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "web_search",
                "input": {"query": "weather"}
            }
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockStart { index, content_block } => {
                assert_eq!(index, 0);
                match content_block {
                    ContentBlock::ToolUse(tu) => {
                        assert_eq!(tu.id, "toolu_1");
                        assert_eq!(tu.name, "web_search");
                    }
                    _ => panic!("expected ToolUse block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }

    #[test]
    fn content_block_delta_text_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "}
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    Delta::Text(td) => assert_eq!(td.text, "Hello "),
                    _ => panic!("expected Text delta"),
                }
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn content_block_delta_citations_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {"url": "https://c.com", "title": "C"}
            }
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    Delta::Citations(cd) => {
                        assert_eq!(cd.citation.url, "https://c.com");
                        assert_eq!(cd.citation.title.as_deref(), Some("C"));
                    }
                    _ => panic!("expected Citations delta"),
                }
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn content_block_delta_thinking_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "..."}
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert!(matches!(delta, Delta::Thinking));
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn content_block_delta_signature_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig123"}
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert!(matches!(delta, Delta::Signature));
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn content_block_delta_input_json_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"query\":"}
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert!(matches!(delta, Delta::InputJson));
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn content_block_stop_event() {
        let json = r#"{"type": "content_block_stop", "index": 0}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockStop { index } => assert_eq!(index, 0),
            _ => panic!("expected ContentBlockStop"),
        }
    }

    #[test]
    fn message_stop_event() {
        let json = r#"{"type": "message_stop"}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, SseEvent::MessageStop));
    }

    #[test]
    fn message_start_event() {
        let json = r#"{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude-sonnet-4-6", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 5}}}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, SseEvent::MessageStart));
    }

    #[test]
    fn message_delta_event() {
        let json = r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 15}}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, SseEvent::MessageDelta));
    }

    #[test]
    fn ping_event() {
        let json = r#"{"type": "ping"}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, SseEvent::Ping));
    }

    #[test]
    fn text_block_without_citations() {
        let json = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "text",
                "text": "plain text"
            }
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        match event {
            SseEvent::ContentBlockStart { index, content_block } => {
                assert_eq!(index, 1);
                match content_block {
                    ContentBlock::Text(tb) => {
                        assert_eq!(tb.text, "plain text");
                        assert!(tb.citations.is_empty());
                    }
                    _ => panic!("expected Text block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }
}
