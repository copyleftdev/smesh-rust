//! Serve the ratification bench.
//!
//! ```sh
//! cargo run -p smesh-ratify -- refinery-staged.json   # review a real run
//! cargo run -p smesh-ratify -- --demo                 # offline demo session
//! ```
//!
//! Reviewer identity comes from `SMESH_REVIEWER` (falling back to `USER`);
//! the Ed25519 keypair persists at `.ratify/reviewer.ed25519`. Decisions and
//! the signed revision are written next to the staged file.

use smesh_ratify::demo::demo_staged_run;
use smesh_ratify::signer::ReviewerKey;
use smesh_ratify::state::Session;
use smesh_ratify::web::{router, App};
use smesh_world::ReviewerId;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut demo = false;
    let mut port: u16 = 8093;
    let mut staged_path = "refinery-staged.json".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => demo = true,
            "--port" => {
                port = args.next().and_then(|p| p.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--port requires a number");
                    std::process::exit(2);
                })
            }
            other => staged_path = other.to_owned(),
        }
    }

    let (staged, stem) = if demo {
        (demo_staged_run(), PathBuf::from("demo-session"))
    } else {
        let path = staged_path;
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("cannot read staged run {path}: {e}");
                eprintln!("produce one with `cargo run -p smesh-refinery`, or use --demo");
                std::process::exit(2);
            }
        };
        let staged = match serde_json::from_slice(&bytes) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{path} is not a staged run: {e}");
                std::process::exit(2);
            }
        };
        (staged, PathBuf::from(path))
    };

    let reviewer_name = std::env::var("SMESH_REVIEWER")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "reviewer".to_owned());
    let reviewer = match ReviewerKey::load_or_generate(
        &PathBuf::from(".ratify/reviewer.ed25519"),
        ReviewerId(reviewer_name),
    ) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("reviewer key error: {e}");
            std::process::exit(2);
        }
    };

    let session = match Session::open(
        staged,
        stem.with_extension("decisions.json"),
        stem.with_extension("revision.json"),
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open session: {e}");
            std::process::exit(2);
        }
    };

    let app = App {
        session: Arc::new(Mutex::new(session)),
        reviewer: Arc::new(reviewer),
    };

    let addr = format!("127.0.0.1:{port}");
    println!("ratification bench: http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("bind ratification bench port");
    axum::serve(listener, router(app))
        .await
        .expect("serve ratification bench");
}
