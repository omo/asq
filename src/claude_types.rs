use serde::Serialize;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
