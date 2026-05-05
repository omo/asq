use serde::Serialize;

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest {
    pub(crate) stream: bool,
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
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
        };
        let json = serde_json::to_string(&req).unwrap();
        let want = r#"{"stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
        assert_eq!(json, want);
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
