use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Stub: Brave Search "Answers" API client.
#[allow(dead_code)]
pub struct BraveClient {
    client: reqwest::Client,
}

/// A single search result from Brave Answers.
#[derive(Debug, Deserialize, Serialize)]
#[allow(dead_code)]
pub struct BraveAnswer {
    // TODO: populate once we inspect the real API response
}

#[allow(dead_code)]
impl BraveClient {
    /// Create a new Brave client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Search the web using Brave Answers API.
    ///
    /// # Arguments
    ///
    /// * `query` - The user's search query.
    pub async fn search(&self, query: &str) -> Result<BraveAnswer> {
        // TODO: implement actual API call
        let _ = query;
        todo!("Brave Search API call")
    }
}
