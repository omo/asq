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
