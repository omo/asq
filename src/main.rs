mod brave;
mod cli;
mod config;
mod gemini;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = cli::Cli::parse();
    let config = config::Config::from_env()?;

    let _brave = brave::BraveClient::new();
    let gemini = gemini::GeminiClient::new(config.gemini_api_key);

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
                        for chunk in chunks {
                            println!("  • [{}]({})", chunk.web.title, chunk.web.uri);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}