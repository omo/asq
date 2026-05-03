use anyhow::{Context, Result};

/// Configuration loaded from the environment (and/or `.env` file).
pub struct Config {
    #[allow(dead_code)]
    pub brave_api_key: Option<String>,
    pub gemini_api_key: String,
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// Expects:
    /// - `GEMINI_API_KEY` — Google Gemini API key (required)
    /// - `BRAVE_API_KEY` — Brave Search API key (optional)
    pub fn from_env() -> Result<Self> {
        // Try project-local .env first, then home ~/.env
        dotenvy::dotenv().ok();
        if std::env::var("GEMINI_API_KEY").is_err() {
            dotenvy::from_filename(
                std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join(".env"),
            )
            .ok();
        }

        let gemini_api_key = std::env::var("GEMINI_API_KEY")
            .context("GEMINI_API_KEY must be set in environment, .env, or ~/.env")?;

        let brave_api_key = std::env::var("BRAVE_API_KEY").ok();

        Ok(Self {
            brave_api_key,
            gemini_api_key,
        })
    }
}
