//! Render adjudication results as a table, JSON, and an HTML decision report.

use super::engine::{Adjudication, Decision, Disposition, Finding};

fn decision_color(d: Decision) -> &'static str {
    match d {
        Decision::Approve => "#34d399",
        Decision::PendPriorAuth | Decision::PendStepTherapy => "#e8a030",
        Decision::EscalateHuman => "#38bdf8",
        Decision::DenyDraft => "#f87171",
    }
}

fn disp_color(d: &Disposition) -> &'static str {
    match d {
        Disposition::Approve => "#34d399",
        Disposition::Flag(_) => "#38bdf8",
        Disposition::Pend(_) => "#e8a030",
        Disposition::Deny(_) => "#f87171",
    }
}

/// Print a text summary of a batch.
pub fn print_report(results: &[Adjudication]) {
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║        SMESH × AION — Pharmaceutical Claim Adjudication            ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    for a in results {
        println!(
            "▸ {} · {} ({})   →  {}",
            a.claim.id,
            a.claim.drug,
            a.claim.atc,
            a.decision.label()
        );
        println!(
            "   policy: {} ({}) · {} · author {} · signature {} · sha256 {}",
            a.policy_name,
            a.regulation,
            a.aion_file,
            a.author_id,
            if a.verified { "verified ✓" } else { "UNVERIFIED ✗" },
            short_hash(&a.sha256),
        );
        for f in &a.findings {
            println!(
                "     [{:<11}] {:<8} — {}",
                f.agent,
                f.disposition.tag_upper(),
                f.citation
            );
        }
        if let Some(g) = &a.gating_constraint {
            println!("   ⚖ governance gate: {g}");
        }
        println!("   {}\n", a.rationale);
    }

    let approve = results.iter().filter(|a| a.decision == Decision::Approve).count();
    let escalate = results.iter().filter(|a| a.decision == Decision::EscalateHuman).count();
    let pend = results
        .iter()
        .filter(|a| matches!(a.decision, Decision::PendPriorAuth | Decision::PendStepTherapy))
        .count();
    println!(
        "Summary: {} approved · {} pended · {} escalated to human · {} claims",
        approve,
        pend,
        escalate,
        results.len()
    );
}

trait TagUpper {
    fn tag_upper(&self) -> &'static str;
}
impl TagUpper for Disposition {
    fn tag_upper(&self) -> &'static str {
        match self {
            Disposition::Approve => "APPROVE",
            Disposition::Flag(_) => "FLAG",
            Disposition::Pend(_) => "PEND",
            Disposition::Deny(_) => "DENY",
        }
    }
}

/// JSON for a single adjudication (reused by the showcase API).
pub fn adjudication_json(a: &Adjudication) -> serde_json::Value {
    serde_json::json!({
        "claim": {
            "id": a.claim.id, "drug": a.claim.drug, "atc": a.claim.atc,
            "tier": a.claim.tier, "cost_usd": a.claim.cost_usd, "diagnosis": a.claim.diagnosis,
        },
        "policy": {
            "name": a.policy_name, "regulation": a.regulation, "aion_file": a.aion_file,
            "author_id": a.author_id, "verified": a.verified, "sha256": a.sha256,
        },
        "findings": a.findings.iter().map(finding_json).collect::<Vec<_>>(),
        "consensus": a.consensus.iter().map(|(tag,n)| serde_json::json!({"tag":tag,"agents":n})).collect::<Vec<_>>(),
        "decision": a.decision.code(),
        "decision_label": a.decision.label(),
        "decision_color": decision_color(a.decision),
        "rationale": a.rationale,
        "gating_constraint": a.gating_constraint,
    })
}

fn finding_json(f: &Finding) -> serde_json::Value {
    serde_json::json!({
        "agent": f.agent,
        "disposition": f.disposition.tag_upper(),
        "color": disp_color(&f.disposition),
        "reason": f.disposition.reason(),
        "citation": f.citation,
        "detail": f.detail,
    })
}

/// JSON for a batch.
pub fn to_json(results: &[Adjudication]) -> String {
    let arr: Vec<serde_json::Value> = results.iter().map(adjudication_json).collect();
    serde_json::to_string_pretty(&serde_json::json!({ "adjudications": arr }))
        .unwrap_or_else(|_| "[]".to_string())
}

/// Render a self-contained HTML decision report for a batch.
pub fn render_html(results: &[Adjudication]) -> String {
    let mut cards = String::new();
    for a in results {
        let dcol = decision_color(a.decision);
        let mut rows = String::new();
        for f in &a.findings {
            rows.push_str(&format!(
                "<tr><td>{}</td><td style='color:{}'>{}</td><td>{}</td><td class='cite'>{}</td></tr>",
                esc(&f.agent),
                disp_color(&f.disposition),
                f.disposition.tag_upper(),
                esc(f.disposition.reason()),
                esc(&f.citation)
            ));
        }
        let gate = a
            .gating_constraint
            .as_ref()
            .map(|g| format!("<div class='gate'>⚖ governance gate — {}</div>", esc(g)))
            .unwrap_or_default();
        cards.push_str(&format!(
            r#"<div class="card">
              <div class="chead">
                <div><span class="cid">{cid}</span> <span class="drug">{drug}</span> <span class="atc">{atc}</span></div>
                <div class="badge" style="background:{dcol}22;color:{dcol};border-color:{dcol}55">{decision}</div>
              </div>
              <div class="prov">{policy} ({reg}) · {file} · author {author} · signature {sig} · sha256 {hash}</div>
              <table><thead><tr><th>Agent</th><th>Disposition</th><th>Reason</th><th>Signed rule</th></tr></thead><tbody>{rows}</tbody></table>
              {gate}
              <div class="rationale">{rationale}</div>
            </div>"#,
            cid = esc(&a.claim.id),
            drug = esc(&a.claim.drug),
            atc = esc(&a.claim.atc),
            dcol = dcol,
            decision = a.decision.label(),
            policy = esc(&a.policy_name),
            reg = esc(&a.regulation),
            file = esc(&a.aion_file),
            author = a.author_id,
            sig = if a.verified { "verified ✓" } else { "UNVERIFIED ✗" },
            hash = short_hash(&a.sha256),
            rows = rows,
            gate = gate,
            rationale = esc(&a.rationale),
        ));
    }

    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMESH × AION — Claim Adjudication</title>
<style>
  :root{{color-scheme:dark;--bg:#06060c;--card:#13131f;--border:#1e1e32;--fg:#e4e4ed;--mut:#7a7a96;--dim:#4a4a64;--mono:'SF Mono','JetBrains Mono',Consolas,monospace}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:32px}}
  .wrap{{max-width:960px;margin:0 auto}} h1{{font-size:21px;margin:0 0 4px}}
  .meta{{font-family:var(--mono);font-size:12px;color:var(--mut);margin-bottom:22px}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px}}
  .chead{{display:flex;justify-content:space-between;align-items:center;gap:12px}}
  .cid{{font-family:var(--mono);font-size:12px;color:var(--mut)}}
  .drug{{font-weight:600}} .atc{{font-family:var(--mono);font-size:12px;color:var(--dim)}}
  .badge{{font-family:var(--mono);font-size:12px;font-weight:700;padding:5px 11px;border-radius:7px;border:1px solid;letter-spacing:.04em}}
  .prov{{font-family:var(--mono);font-size:11px;color:var(--mut);margin:8px 0 12px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}}
  th{{color:var(--mut);font-size:10px;letter-spacing:.06em;text-transform:uppercase}}
  td:nth-child(2){{font-family:var(--mono);font-weight:600}}
  .cite{{font-family:var(--mono);font-size:11px;color:var(--mut)}}
  .gate{{font-family:var(--mono);font-size:12px;color:#38bdf8;background:#38bdf815;border:1px solid #38bdf833;border-radius:7px;padding:8px 10px;margin-top:12px}}
  .rationale{{color:var(--mut);font-size:13px;margin-top:10px}}
</style></head><body><div class="wrap">
  <h1>SMESH × AION — Pharmaceutical Claim Adjudication</h1>
  <div class="meta">signed formulary policies · deterministic mesh consensus · governance-gated decisions · {n} claims</div>
  {cards}
</div></body></html>"#,
        n = results.len(),
        cards = cards,
    )
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// First 12 hex chars of a content hash, for compact display.
fn short_hash(h: &str) -> String {
    h.chars().take(12).collect()
}
