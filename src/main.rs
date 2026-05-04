mod brave;
mod claude;
mod cli;
mod config;
mod gemini;

use anyhow::Result;
use cli::Engine;
use futures_util::future::join_all;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = cli::Cli::parse();
    let config = config::Config::from_env()?;

    match cli.engine {
        Engine::Gemini => run_gemini(cli, config).await?,
        Engine::Brave => run_brave(cli, config).await?,
        Engine::Claude => run_claude(cli, config).await?,
    }

    Ok(())
}

async fn run_gemini(cli: cli::Cli, _config: config::Config) -> Result<()> {
    // API key from CLI arg takes precedence over env/config
    let api_key = cli
        .gemini_api_key
        .clone()
        .or(_config.gemini_api_key)
        .ok_or_else(|| anyhow::anyhow!("GEMINI_API_KEY must be set"))?;

    let gemini = gemini::GeminiClient::new(api_key);
    let mut rx = gemini.ask_stream(&cli.question).await?;

    use gemini::GeminiStreamEvent;

    while let Some(event) = rx.recv().await {
        match event? {
            GeminiStreamEvent::Text(text) => {
                print!("{text}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            GeminiStreamEvent::Done(metadata) => {
                println!(); // newline after last text chunk

                if let Some(meta) = metadata {
                    if let Some(queries) = &meta.web_search_queries {
                        println!("\nSearch queries: {}", queries.join(", "));
                    }
                    if let Some(chunks) = &meta.grounding_chunks {
                        println!("\nSources:");
                        // Fire off all redirect resolutions concurrently
                        let futures: Vec<_> = chunks
                            .iter()
                            .map(|chunk| gemini.resolve_redirect(&chunk.web.uri))
                            .collect();
                        let urls = join_all(futures).await;
                        for url in &urls {
                            println!("  {url}");
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

async fn run_claude(cli: cli::Cli, _config: config::Config) -> Result<()> {
    let api_key = cli
        .claude_api_key
        .clone()
        .or(_config.claude_api_key)
        .ok_or_else(|| anyhow::anyhow!("CLAUDE_API_KEY must be set (use --claude-api-key or set CLAUDE_API_KEY env var)"))?;

    let claude = claude::ClaudeClient::new(api_key);
    let mut rx = claude.ask_stream(&cli.question).await?;

    use claude::ClaudeStreamEvent;

    while let Some(event) = rx.recv().await {
        match event? {
            ClaudeStreamEvent::Text(text) => {
                print!("{text}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            ClaudeStreamEvent::Done => {
                println!();
            }
        }
    }

    Ok(())
}

async fn run_brave(cli: cli::Cli, _config: config::Config) -> Result<()> {
    let api_key = cli
        .brave_api_key
        .clone()
        .or(_config.brave_api_key)
        .ok_or_else(|| anyhow::anyhow!("BRAVE_API_KEY must be set (use --brave-api-key or set BRAVE_API_KEY env var)"))?;

    let brave = brave::BraveClient::new(api_key);
    let mut rx = brave.ask_stream(&cli.question).await?;

    use brave::BraveStreamEvent;

    while let Some(event) = rx.recv().await {
        match event? {
            BraveStreamEvent::Text(text) => {
                print!("{text}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            BraveStreamEvent::Done => {
                println!();
            }
        }
    }

    Ok(())
}