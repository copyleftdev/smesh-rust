//! Live refinery run against the Meridian benchmark.
//!
//! ```sh
//! cargo run -p smesh-refinery
//! ```
//!
//! Requires OpenRouter credentials (`OPENROUTER_API_KEY` or
//! `~/.creds/openrouter.env`). Extraction runs on Kimi K3, verification on a
//! distinct model family, per the role model policy. Temperature is pinned
//! to 0 — the refinery wants reproducibility, not creativity.

use smesh_agent::openrouter::OpenRouterConfig;
use smesh_agent::OpenRouterClient;
use smesh_refinery::run::refine_meridian;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let Some(mut config) = OpenRouterConfig::from_env() else {
        eprintln!("no OpenRouter credentials (OPENROUTER_API_KEY or ~/.creds/openrouter.env)");
        std::process::exit(2);
    };
    config.temperature = 0.0;
    config.max_tokens = 8192;
    config.timeout_secs = 420;
    let client = OpenRouterClient::new(config);

    match refine_meridian(&client).await {
        Ok(report) => {
            print!("{}", report.render());
            if let Some(staged) = report.staged_run() {
                let path = "refinery-staged.json";
                match serde_json::to_string_pretty(&staged)
                    .map_err(|e| e.to_string())
                    .and_then(|json| std::fs::write(path, json).map_err(|e| e.to_string()))
                {
                    Ok(()) => println!("staged run written to {path} (serve with smesh-ratify)"),
                    Err(e) => eprintln!("failed to write {path}: {e}"),
                }
            }
            let gate_passed = report
                .scorecard
                .as_ref()
                .is_some_and(smesh_world::corpus::Scorecard::passes_gate);
            std::process::exit(i32::from(!gate_passed));
        }
        Err(e) => {
            eprintln!("refinery run failed: {e}");
            std::process::exit(2);
        }
    }
}
