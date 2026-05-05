mod brave;
mod brave_types;
mod claude;
mod claude_types;
mod cli;
mod config;
mod gemini;
mod gemini_types;
mod stream;

use anyhow::Result;
use cli::Engine;
use stream::{GroundingData, StreamClient, StreamEvent};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = cli::Cli::parse();
    let config = config::Config::from_env()?;

    match cli.engine {
        Engine::Gemini => {
            let key = get_api_key(
                cli.gemini_api_key,
                config.gemini_api_key,
                "gemini-api-key",
                "GEMINI_API_KEY",
            )?;
            run::<gemini::GeminiClient>(&cli.question, key).await?;
        }
        Engine::Claude => {
            let key = get_api_key(
                cli.claude_api_key,
                config.claude_api_key,
                "claude-api-key",
                "CLAUDE_API_KEY",
            )?;
            run::<claude::ClaudeClient>(&cli.question, key).await?;
        }
        Engine::Brave => {
            let key = get_api_key(
                cli.brave_api_key,
                config.brave_api_key,
                "brave-api-key",
                "BRAVE_API_KEY",
            )?;
            run::<brave::BraveClient>(&cli.question, key).await?;
        }
    }

    Ok(())
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Resolve an API key from CLI arg → config, with a descriptive error.
fn get_api_key(
    cli_key: Option<String>,
    config_key: Option<String>,
    flag: &str,
    env: &str,
) -> Result<String> {
    cli_key
        .or(config_key)
        .ok_or_else(|| anyhow::anyhow!("{env} must be set (use --{flag} or set {env})"))
}

/// Run any engine: create the client, stream the response, and print it.
async fn run<C: StreamClient>(question: &str, api_key: String) -> Result<()> {
    let client = C::new(api_key);
    let rx = client.ask_stream(question).await?;
    let grounding = print_until_done(rx).await?;

    if let Some(g) = grounding {
        if !g.web_search_queries.is_empty() {
            println!("\nSearch queries: {}", g.web_search_queries.join(", "));
        }
        for source in &g.sources {
            println!("  {}", source.uri);
        }
    }

    Ok(())
}

/// Print incoming text chunks until `Done`, then return any grounding data.
async fn print_until_done(
    mut rx: mpsc::UnboundedReceiver<Result<StreamEvent>>,
) -> Result<Option<GroundingData>> {
    use std::io::Write;

    loop {
        match rx.recv().await {
            Some(Ok(StreamEvent::Text(text))) => {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            Some(Ok(StreamEvent::Done(g))) => {
                println!();
                return Ok(g);
            }
            Some(Err(e)) => return Err(e),
            None => return Ok(None),
        }
    }
}
