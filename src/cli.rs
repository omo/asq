use clap::Parser;

/// ASQ — Ask a Squpid Quesiton using AI + web search.
#[derive(Parser, Debug)]
#[command(name = "asq", version, about)]
pub struct Cli {
    /// The question to answer.
    pub question: String,

    /// Brave Search API key (overrides BRAVE_API_KEY env var).
    #[arg(long, env = "BRAVE_API_KEY")]
    pub brave_api_key: Option<String>,

    /// Gemini API key (overrides GEMINI_API_KEY env var).
    #[arg(long, env = "GEMINI_API_KEY")]
    pub gemini_api_key: Option<String>,
}

impl Cli {
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }
}
