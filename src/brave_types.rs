use serde::Serialize;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
