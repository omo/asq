use anyhow::Result;

/// Configuration loaded from the environment (and/or `.env` file).
pub struct Config {
    pub brave_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub claude_api_key: Option<String>,
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// Reads:
    /// - `GEMINI_API_KEY` — Google Gemini API key (optional)
    /// - `BRAVE_API_KEY` — Brave Search API key (optional)
    /// - `CLAUDE_API_KEY` — Anthropic Claude API key (optional)
    ///
    /// Keys are validated per-engine at the point of use.
    pub fn from_env() -> Result<Self> {
        // Try project-local .env first, then home ~/.env
        dotenvy::dotenv().ok();
        if std::env::var("GEMINI_API_KEY").is_err()
            && std::env::var("BRAVE_API_KEY").is_err()
            && std::env::var("CLAUDE_API_KEY").is_err()
        {
            dotenvy::from_filename(
                std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join(".env"),
            )
            .ok();
        }

        let gemini_api_key = std::env::var("GEMINI_API_KEY").ok();
        let brave_api_key = std::env::var("BRAVE_API_KEY").ok();
        let claude_api_key = std::env::var("CLAUDE_API_KEY").ok();

        Ok(Self {
            brave_api_key,
            gemini_api_key,
            claude_api_key,
        })
    }
}
