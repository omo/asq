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
