// Allow dead code on deserialization-only fields (used by serde at runtime)
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct GptRequest {
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<String>,
    pub(crate) input: Vec<InputMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) stream: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct InputMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub(crate) tool_type: String,
}

// ─── SSE event types ──────────────────────────────────────────────────────

/// A generic SSE event envelope - we parse based on the `type` field.
#[derive(Debug, Deserialize)]
pub(crate) struct SseEvent {
    #[serde(rename = "type")]
    pub(crate) event_type: String,
    /// Present in `response.output_text.delta`
    #[serde(default)]
    pub(crate) delta: Option<String>,
    /// Present in `response.completed`
    #[serde(default)]
    pub(crate) response: Option<ResponseEnvelope>,
    /// Present in `response.web_search_call.completed`
    #[serde(default)]
    pub(crate) item_id: Option<String>,
    /// Present in `response.output_item.done`
    #[serde(default)]
    pub(crate) item: Option<OutputItem>,
}

/// The full response object (inside `response.completed`).
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseEnvelope {
    pub(crate) id: String,
    pub(crate) status: Option<String>,
    #[serde(default)]
    pub(crate) output: Vec<OutputItem>,
    #[serde(default)]
    pub(crate) usage: Option<serde_json::Value>,
}

/// An output item in the response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum OutputItem {
    #[serde(rename = "web_search_call")]
    WebSearchCall(WebSearchCall),
    #[serde(rename = "message")]
    Message(MessageOutput),
}

#[derive(Debug, Deserialize)]
pub(crate) struct WebSearchCall {
    pub(crate) id: Option<String>,
    pub(crate) status: Option<String>,
    #[serde(default)]
    pub(crate) action: Option<SearchAction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SearchAction {
    #[serde(rename = "type")]
    pub(crate) action_type: Option<String>,
    #[serde(default)]
    pub(crate) query: Option<String>,
    #[serde(default)]
    pub(crate) queries: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MessageOutput {
    pub(crate) id: Option<String>,
    pub(crate) role: Option<String>,
    #[serde(default)]
    pub(crate) content: Vec<ContentPart>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ContentPart {
    #[serde(rename = "output_text")]
    OutputText(OutputText),
}

#[derive(Debug, Deserialize)]
pub(crate) struct OutputText {
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) annotations: Vec<Annotation>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Annotation {
    #[serde(rename = "type")]
    pub(crate) annotation_type: Option<String>,
    #[serde(default)]
    pub(crate) url: Option<String>,
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

    #[test]
    fn gpt_request_with_tools_serializes() {
        let req = GptRequest {
            model: "gpt-5.4-mini".into(),
            instructions: None,
            input: vec![InputMessage {
                role: "user".into(),
                content: "hello".into(),
            }],
            tools: Some(vec![Tool {
                tool_type: "web_search_preview".into(),
            }]),
            max_output_tokens: Some(4096),
            stream: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""model":"gpt-5.4-mini""#));
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""content":"hello""#));
        assert!(json.contains(r#""stream":true"#));
        assert!(json.contains(r#""max_output_tokens":4096"#));
        assert!(json.contains(r#""type":"web_search_preview""#));
        assert!(json.contains(r#""input""#));
    }

    #[test]
    fn gpt_request_without_tools() {
        let req = GptRequest {
            model: "gpt-5.4-mini".into(),
            instructions: None,
            input: vec![InputMessage {
                role: "user".into(),
                content: "q".into(),
            }],
            tools: None,
            max_output_tokens: None,
            stream: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
        assert!(!json.contains("max_output_tokens"));
    }

    #[test]
    fn output_text_delta_deserializes() {
        let json = r#"{"type":"response.output_text.delta","delta":"Hello","content_index":0,"item_id":"msg_1","output_index":0,"sequence_number":1}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "response.output_text.delta");
        assert_eq!(event.delta.as_deref(), Some("Hello"));
        assert!(event.response.is_none());
    }

    #[test]
    fn completed_event_deserializes() {
        let json = r#"{
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "weather Tokyo",
                            "queries": ["weather Tokyo"]
                        }
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "It is 75°F in Tokyo.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://weather.example.com",
                                        "title": "Tokyo Weather",
                                        "start_index": 0,
                                        "end_index": 10
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "response.completed");
        let resp = event.response.unwrap();
        assert_eq!(resp.id, "resp_1");
        assert_eq!(resp.output.len(), 2);

        // Check web search call
        match &resp.output[0] {
            OutputItem::WebSearchCall(w) => {
                assert_eq!(w.action.as_ref().unwrap().query.as_deref(), Some("weather Tokyo"));
            }
            _ => panic!("expected WebSearchCall"),
        }

        // Check message with annotation
        match &resp.output[1] {
            OutputItem::Message(m) => {
                let content = &m.content[0];
                match content {
                    ContentPart::OutputText(t) => {
                        assert_eq!(t.text.as_deref(), Some("It is 75°F in Tokyo."));
                        assert_eq!(t.annotations.len(), 1);
                        assert_eq!(t.annotations[0].url.as_deref(), Some("https://weather.example.com"));
                        assert_eq!(t.annotations[0].title.as_deref(), Some("Tokyo Weather"));
                    }
                }
            }
            _ => panic!("expected Message"),
        }
    }

    #[test]
    fn completed_event_minimal() {
        let json = r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "response.completed");
        let resp = event.response.unwrap();
        assert!(resp.output.is_empty());
    }

    #[test]
    fn output_item_done_with_web_search() {
        let json = r#"{"type":"response.output_item.done","item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","queries":["test query"],"query":"test query"}},"output_index":0,"sequence_number":6}"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "response.output_item.done");
        let item = event.item.unwrap();
        match item {
            OutputItem::WebSearchCall(w) => {
                assert_eq!(w.action.as_ref().unwrap().query.as_deref(), Some("test query"));
            }
            _ => panic!("expected WebSearchCall"),
        }
    }

    #[test]
    fn input_message_serializes() {
        let msg = InputMessage {
            role: "user".into(),
            content: "some text".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"some text"}"#);
    }

    #[test]
    fn tool_serializes() {
        let tool = Tool {
            tool_type: "web_search_preview".into(),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(json, r#"{"type":"web_search_preview"}"#);
    }
}
