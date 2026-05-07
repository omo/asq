mod brave;
mod brave_types;
mod claude;
mod claude_types;
mod cli;
mod gemini;
mod gemini_types;
mod gpt;
mod gpt_types;
mod stream;

use anyhow::Result;
use cli::Engine;
use stream::{GroundingData, StreamClient, StreamEvent};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    load_env();
    let cli = cli::Cli::parse();

    match cli.engine {
        Engine::Gemini => {
            let key = get_api_key(cli.gemini_api_key, "GEMINI_API_KEY")?;
            run::<gemini::GeminiClient>(&cli.question, key).await?;
        }
        Engine::Claude => {
            let key = get_api_key(cli.claude_api_key, "CLAUDE_API_KEY")?;
            run::<claude::ClaudeClient>(&cli.question, key).await?;
        }
        Engine::Brave => {
            let key = get_api_key(cli.brave_api_key, "BRAVE_API_KEY")?;
            run::<brave::BraveClient>(&cli.question, key).await?;
        }
        Engine::Gpt => {
            let key = get_api_key(cli.gpt_api_key, "OPENAI_API_KEY")?;
            run::<gpt::GptClient>(&cli.question, key).await?;
        }
    }

    Ok(())
}

// ─── .env Loading ─────────────────────────────────────────────────────────

/// Seed the environment from `$HOME/.env` (low priority) and project-local
/// `.env` (high priority). This must run before clap parses args so that
/// clap's `#[arg(env)]` picks up the values.
fn load_env() {
    let home = std::env::var("HOME").unwrap_or_default();
    dotenvy::from_filename(std::path::Path::new(&home).join(".env")).ok();
    dotenvy::dotenv().ok();
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Resolve an API key, returning a descriptive error if missing.
///
/// The key comes from clap, which has already checked the `--flag` arg
/// and the eponymous environment variable (thanks to `#[arg(env)]`).
fn get_api_key(key: Option<String>, env: &str) -> Result<String> {
    key.ok_or_else(|| anyhow::anyhow!("{env} must be set (use --{env} or set the {env} env var)"))
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
