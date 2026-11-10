//! The run's human-readable report, written next to its journal.

use serde_json::Value;

use crate::board::Finding;
use crate::run::RunConfig;

fn f(v: &Value) -> String {
    v.as_f64()
        .map(|x| format!("{x:.4}"))
        .unwrap_or_else(|| "–".into())
}

/// Render `report.md`.
pub fn render(cfg: &RunConfig, metrics: &Value, findings: &[(String, Finding)]) -> String {
    let mut out = String::new();
    out.push_str("# SMESH quant research run\n\n");
    out.push_str("Read-only: no order, account or private endpoint was called. ");
    out.push_str("Paper trades are simulated and were never sent anywhere.\n\n");

    out.push_str("## Findings\n\n");
    out.push_str("A hypothesis is a signal on the SMESH field: confirmed out-of-sample windows reinforce it, ");
    out.push_str("contradicting windows and the skeptic's doubts cut its confidence, and it decays without fresh evidence. ");
    out.push_str("Consensus means its strength ended at or above 0.30.\n\n");
    out.push_str("| Track | Hypothesis | Strength | Confirmed | Contradicted | Consensus |\n|---|---|---|---|---|---|\n");
    for (track, h) in findings {
        out.push_str(&format!(
            "| {track} | {} | {:.2} | {} | {} | {} |\n",
            h.label,
            h.strength,
            h.confirmations,
            h.contradictions,
            if h.consensus { "**yes**" } else { "no" }
        ));
    }

    if let Some(c) = metrics.get("crypto") {
        out.push_str("\n## Crypto: Kalshi hourly odds vs a volatility model\n\n");
        out.push_str(&format!(
            "{} samples, {} walk-forward windows, {} out-of-sample predictions. Settings: {:?}.\n\n",
            c["samples"], c["windows"], c["test_predictions"], cfg.crypto
        ));
        out.push_str("| | Kalshi odds | Model | Blend |\n|---|---|---|---|\n");
        out.push_str(&format!(
            "| Log loss (lower is better) | {} | {} | {} |\n",
            f(&c["logloss"]["market"]),
            f(&c["logloss"]["model"]),
            f(&c["logloss"]["blend"])
        ));
        out.push_str(&format!(
            "| Brier | {} | {} | {} |\n\n",
            f(&c["brier"]["market"]),
            f(&c["brier"]["model"]),
            f(&c["brier"]["blend"])
        ));
        out.push_str(&format!(
            "Paper strategy: {} trades, P&L {} per contract in total ({} per trade), after Kalshi taker fees.\n\n",
            c["paper"]["trades"],
            f(&c["paper"]["pnl_per_contract_sum"]),
            f(&c["paper"]["pnl_per_trade"])
        ));
        out.push_str(&format!(
            "Mean Kraken-vs-Massive spot basis at decision time: {} bps.\n",
            f(&c["kraken_basis_bps_mean"])
        ));
    }

    if let Some(m) = metrics.get("macro") {
        out.push_str("\n## Macro: Kalshi release markets\n\n");
        out.push_str(&format!(
            "{} releases ({}), {} walk-forward windows (one unseen release each).\n\n",
            m["releases"], m["by_series"], m["windows"]
        ));
        out.push_str(&format!(
            "- Mean absolute error, in threshold spacings: Kalshi {} vs last print {}.\n",
            f(&m["mean_abs_error_in_spacings"]["kalshi"]),
            f(&m["mean_abs_error_in_spacings"]["naive"])
        ));
        if let Some(by) = m["mean_abs_error_by_series"].as_object() {
            for (series, e) in by {
                out.push_str(&format!(
                    "  - {series}: Kalshi {} vs last print {} (mean absolute error, series units, {} releases)\n",
                    f(&e["kalshi"]),
                    f(&e["naive"]),
                    e["n"]
                ));
            }
        }
        out.push_str(&format!(
            "- Threshold Brier score: Kalshi {} vs climatology {} over {} rungs.\n",
            f(&m["brier"]["kalshi"]),
            f(&m["brier"]["climatology"]),
            m["brier"]["n"]
        ));
        if let Some(rates) = m["direction_hit_rate"].as_object() {
            for (asset, r) in rates {
                out.push_str(&format!(
                    "- {asset} direction after release, predicted from the surprise: {}/{} ({}).\n",
                    r["hits"],
                    r["n"],
                    f(&r["rate"])
                ));
            }
        }
    }

    out.push_str("\n## Files\n\n");
    out.push_str(
        "- `timeline.jsonl`: every agent's events merged in time order (for the visualization)\n",
    );
    out.push_str("- `agents/*.jsonl`: one journal per agent\n");
    out.push_str("- `metrics.json`: everything above, machine-readable\n\n");
    out.push_str("## Caveats\n\n");
    out.push_str("- Kalshi crypto markets settle on CF Benchmarks' index; the spot here is Massive's aggregate (cross-checked against Kraken), so small mismatches near a strike are expected.\n");
    out.push_str("- Several hypotheses are tested on the same windows; single-window results are weak evidence.\n");
    out.push_str(
        "- Macro windows hold one release each, so the reaction results rest on few events.\n",
    );
    out.push_str("- Payroll ladders have implied far from the prints in both directions. Before trusting payroll results, confirm which figure Kalshi settles on (first print, revision, or a different survey) against the `expiration_value` used here.\n");
    out
}
