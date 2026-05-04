use clap::{Parser, ValueEnum};

/// ASQ — Ask a Squpid Quesiton using AI + web search.
#[derive(Parser, Debug)]
#[command(name = "asq", version, about)]
pub struct Cli {
    /// The question to answer.
    pub question: String,

    /// Search engine to use: gemini (Google-grounded) or brave (raw web search).
    #[arg(long, short, value_enum, default_value_t = Engine::Gemini)]
    pub engine: Engine,

    /// Brave Search API key (overrides BRAVE_API_KEY env var).
    #[arg(long, env = "BRAVE_API_KEY")]
    pub brave_api_key: Option<String>,

    /// Gemini API key (overrides GEMINI_API_KEY env var).
    #[arg(long, env = "GEMINI_API_KEY")]
    pub gemini_api_key: Option<String>,

    /// Claude API key (overrides CLAUDE_API_KEY env var).
    #[arg(long, env = "CLAUDE_API_KEY")]
    pub claude_api_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum Engine {
    /// Google-grounded Gemini.
    Gemini,
    /// Brave's AI chat model.
    Brave,
    /// Anthropic's Claude Sonnet.
    Claude,
}

impl Cli {
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }
}
