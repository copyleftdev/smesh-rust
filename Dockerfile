# =============================================================================
# SMESH-Rust — Dockerfile
# =============================================================================
# Multi-stage build: compile in rust image, run in minimal alpine.
# =============================================================================

FROM rust:1.84-slim AS builder

WORKDIR /app

# Cache dependency build
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release 2>/dev/null || true
RUN rm -rf src

# Build actual source
COPY src/ ./src/
RUN cargo build --release

# --- Runtime ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/smesh-rust /usr/local/bin/smesh-rust

ENV RUST_LOG=info
ENV SMESH_CONFIG=/etc/smesh/config.toml

CMD ["smesh-rust"]
