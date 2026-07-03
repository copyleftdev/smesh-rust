//! Rendering the scorecard as a table, JSON, and an OWASP-style HTML chart.

use super::score::{Confusion, Scorecard};

/// Print a per-category scorecard table to stdout.
pub fn print_scorecard(sc: &Scorecard) {
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║                  SMESH × OWASP Benchmark Scorecard                 ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!(
        "Model: {}   Agents/case: {}   Cases: {}   Elapsed: {:.1}s\n",
        sc.model, sc.agents, sc.cases_scored, sc.elapsed_secs
    );

    println!(
        "{:<16} {:>4} {:>4} {:>4} {:>4} {:>7} {:>7} {:>8}",
        "Category", "TP", "FP", "TN", "FN", "TPR", "FPR", "Youden"
    );
    println!("{}", "─".repeat(64));

    for (cat, c) in &sc.per_category {
        println!(
            "{:<16} {:>4} {:>4} {:>4} {:>4} {:>6.1}% {:>6.1}% {:>+8.3}",
            cat.code(),
            c.tp,
            c.fp,
            c.tn,
            c.fn_,
            c.tpr() * 100.0,
            c.fpr() * 100.0,
            c.youden()
        );
    }

    println!("{}", "─".repeat(64));
    let o = &sc.overall;
    println!(
        "{:<16} {:>4} {:>4} {:>4} {:>4} {:>6.1}% {:>6.1}% {:>+8.3}",
        "OVERALL",
        o.tp,
        o.fp,
        o.tn,
        o.fn_,
        o.tpr() * 100.0,
        o.fpr() * 100.0,
        o.youden()
    );
    println!(
        "\nOverall Youden index: {:+.3}  (1.0 = perfect, 0.0 = random guessing)",
        o.youden()
    );
    println!(
        "Precision: {:.1}%   Recall: {:.1}%   F1: {:.3}",
        o.precision() * 100.0,
        o.tpr() * 100.0,
        o.f1()
    );
}

/// Serialize the scorecard to pretty JSON.
pub fn to_json(sc: &Scorecard) -> String {
    let cats: Vec<serde_json::Value> = sc
        .per_category
        .iter()
        .map(|(cat, c)| confusion_json(cat.code(), cat.label(), cat.cwe(), c))
        .collect();

    let value = serde_json::json!({
        "benchmark": "OWASP Benchmark v1.2",
        "model": sc.model,
        "agents_per_case": sc.agents,
        "cases_scored": sc.cases_scored,
        "elapsed_secs": sc.elapsed_secs,
        "overall": confusion_json("overall", "Overall", 0, &sc.overall),
        "categories": cats,
    });
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
}

fn confusion_json(code: &str, label: &str, cwe: u32, c: &Confusion) -> serde_json::Value {
    serde_json::json!({
        "code": code,
        "label": label,
        "cwe": cwe,
        "tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.fn_,
        "tpr": c.tpr(), "fpr": c.fpr(), "youden": c.youden(),
        "precision": c.precision(), "f1": c.f1(),
    })
}

/// Render a self-contained OWASP-style HTML scorecard: a TPR-vs-FPR scatter
/// (points above the diagonal beat random guessing) plus a per-category table.
pub fn render_html(sc: &Scorecard) -> String {
    // Plot geometry.
    let (w, h) = (560.0_f64, 560.0_f64);
    let (ml, mt) = (60.0_f64, 30.0_f64);
    let pw = w - ml - 30.0;
    let ph = h - mt - 60.0;
    let px = |fpr: f64| ml + fpr * pw;
    let py = |tpr: f64| mt + (1.0 - tpr) * ph;

    // Gridlines + axis ticks every 20%.
    let mut grid = String::new();
    for i in 0..=5 {
        let t = i as f64 / 5.0;
        let x = px(t);
        let y = py(t);
        grid.push_str(&format!(
            "<line class='grid' x1='{x:.1}' y1='{mt:.1}' x2='{x:.1}' y2='{:.1}'/>",
            mt + ph
        ));
        grid.push_str(&format!(
            "<line class='grid' x1='{ml:.1}' y1='{y:.1}' x2='{:.1}' y2='{y:.1}'/>",
            ml + pw
        ));
        grid.push_str(&format!(
            "<text class='tick' x='{x:.1}' y='{:.1}' text-anchor='middle'>{}%</text>",
            mt + ph + 18.0,
            (t * 100.0) as i32
        ));
        grid.push_str(&format!(
            "<text class='tick' x='{:.1}' y='{:.1}' text-anchor='end'>{}%</text>",
            ml - 8.0,
            y + 4.0,
            (t * 100.0) as i32
        ));
    }

    // Diagonal "random guess" line.
    let diagonal = format!(
        "<line class='diag' x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}'/>",
        px(0.0),
        py(0.0),
        px(1.0),
        py(1.0)
    );

    // Category points.
    let mut points = String::new();
    for (cat, c) in &sc.per_category {
        let x = px(c.fpr());
        let y = py(c.tpr());
        let cls = if c.youden() >= 0.0 { "good" } else { "bad" };
        // No on-chart text label: categories with near-identical TPR/FPR (e.g.
        // the injection group) would collide. The hover tooltip and the
        // per-category table below carry the identity instead.
        points.push_str(&format!(
            "<circle class='pt {cls}' cx='{x:.1}' cy='{y:.1}' r='7'><title>{} — TPR {:.0}%, FPR {:.0}%, Youden {:+.2}</title></circle>",
            cat.label(),
            c.tpr() * 100.0,
            c.fpr() * 100.0,
            c.youden()
        ));
    }

    // Overall marker (diamond).
    let ox = px(sc.overall.fpr());
    let oy = py(sc.overall.tpr());
    let overall_pt = format!(
        "<rect class='overall' x='{:.1}' y='{:.1}' width='14' height='14' transform='rotate(45 {ox:.1} {oy:.1})'><title>OVERALL — Youden {:+.2}</title></rect>",
        ox - 7.0,
        oy - 7.0,
        sc.overall.youden()
    );

    // Table rows.
    let mut rows = String::new();
    for (cat, c) in &sc.per_category {
        rows.push_str(&row_html(cat.code(), cat.cwe(), c));
    }
    let overall_row = format!(
        "<tr class='total'><td>OVERALL</td><td>—</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td class='{}'>{:+.3}</td></tr>",
        sc.overall.tp, sc.overall.fp, sc.overall.tn, sc.overall.fn_,
        sc.overall.tpr() * 100.0, sc.overall.fpr() * 100.0,
        if sc.overall.youden() >= 0.0 { "yg" } else { "yb" }, sc.overall.youden()
    );

    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMESH × OWASP Benchmark Scorecard</title>
<style>
  :root {{ color-scheme: light dark; --bg:#0f1117; --card:#171a23; --fg:#e6e8ee; --mut:#9aa3b2; --line:#2a2f3a; --good:#38b26b; --bad:#e5484d; --acc:#5b8def; }}
  @media (prefers-color-scheme: light) {{ :root {{ --bg:#f6f7f9; --card:#fff; --fg:#1a1d24; --mut:#5a6472; --line:#e3e6ec; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:32px; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .meta {{ color:var(--mut); font-size:13px; margin-bottom:24px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  @media (max-width:880px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }}
  .head {{ font-weight:600; margin-bottom:12px; }}
  .hero {{ font-size:44px; font-weight:700; line-height:1; }}
  .hero.g {{ color:var(--good); }} .hero.b {{ color:var(--bad); }}
  .sub {{ color:var(--mut); font-size:13px; margin-top:6px; }}
  svg {{ width:100%; height:auto; }}
  .grid {{ stroke:var(--line); stroke-width:1; }}
  .diag {{ stroke:var(--mut); stroke-width:1.5; stroke-dasharray:5 5; opacity:.7; }}
  .tick {{ fill:var(--mut); font-size:11px; }}
  .axis {{ fill:var(--mut); font-size:12px; font-weight:600; }}
  .pt.good {{ fill:var(--good); }} .pt.bad {{ fill:var(--bad); }}
  .pt {{ stroke:var(--card); stroke-width:1.5; }}
  .ptlabel {{ fill:var(--fg); font-size:11px; }}
  .overall {{ fill:var(--acc); stroke:var(--card); stroke-width:1.5; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:7px 8px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child,td:first-child {{ text-align:left; }}
  th {{ color:var(--mut); font-weight:600; }}
  tr.total td {{ font-weight:700; border-top:2px solid var(--line); }}
  .yg {{ color:var(--good); font-weight:600; }} .yb {{ color:var(--bad); font-weight:600; }}
  .legend {{ font-size:12px; color:var(--mut); margin-top:10px; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin:0 4px 0 12px; vertical-align:middle; }}
</style></head><body><div class="wrap">
  <h1>SMESH × OWASP Benchmark Scorecard</h1>
  <div class="meta">OWASP Benchmark v1.2 · model <b>{model}</b> · {agents} agents/case · {cases} cases · {elapsed:.1}s · signal-consensus mesh</div>

  <div class="grid2">
    <div class="card">
      <div class="head">Detection quality (TPR vs FPR)</div>
      <svg viewBox="0 0 {w:.0} {h:.0}" role="img" aria-label="TPR vs FPR scatter">
        {grid}
        {diagonal}
        {points}
        {overall_pt}
        <text class="axis" x="{axx:.1}" y="{axy:.1}" text-anchor="middle">False Positive Rate →</text>
        <text class="axis" transform="translate(16 {ayy:.1}) rotate(-90)" text-anchor="middle">True Positive Rate →</text>
      </svg>
      <div class="legend"><span class="dot" style="background:var(--good)"></span>beats guessing<span class="dot" style="background:var(--bad)"></span>below guessing<span class="dot" style="background:var(--acc)"></span>overall · dashed = random</div>
    </div>

    <div class="card">
      <div class="head">Overall Youden index</div>
      <div class="hero {herocls}">{youden:+.3}</div>
      <div class="sub">TPR {tpr:.1}% − FPR {fpr:.1}%. 1.0 is perfect; 0.0 is no better than random guessing.</div>
      <div style="height:16px"></div>
      <div class="head">Precision {prec:.1}% · Recall {rec:.1}% · F1 {f1:.3}</div>
      <div class="sub">TP {tp} · FP {fp} · TN {tn} · FN {fn_}</div>
    </div>
  </div>

  <div style="height:24px"></div>
  <div class="card">
    <div class="head">Per-category results</div>
    <table>
      <thead><tr><th>Category</th><th>CWE</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th><th>TPR</th><th>FPR</th><th>Youden</th></tr></thead>
      <tbody>{rows}{overall_row}</tbody>
    </table>
  </div>
</div></body></html>"#,
        model = html_escape(&sc.model),
        agents = sc.agents,
        cases = sc.cases_scored,
        elapsed = sc.elapsed_secs,
        w = w, h = h,
        grid = grid, diagonal = diagonal, points = points, overall_pt = overall_pt,
        axx = ml + pw / 2.0, axy = h - 20.0, ayy = mt + ph / 2.0,
        herocls = if sc.overall.youden() >= 0.0 { "g" } else { "b" },
        youden = sc.overall.youden(),
        tpr = sc.overall.tpr() * 100.0, fpr = sc.overall.fpr() * 100.0,
        prec = sc.overall.precision() * 100.0, rec = sc.overall.tpr() * 100.0, f1 = sc.overall.f1(),
        tp = sc.overall.tp, fp = sc.overall.fp, tn = sc.overall.tn, fn_ = sc.overall.fn_,
        rows = rows, overall_row = overall_row,
    )
}

fn row_html(code: &str, cwe: u32, c: &Confusion) -> String {
    let ycls = if c.youden() >= 0.0 { "yg" } else { "yb" };
    format!(
        "<tr><td>{code}</td><td>{cwe}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td class='{ycls}'>{:+.3}</td></tr>",
        c.tp, c.fp, c.tn, c.fn_, c.tpr() * 100.0, c.fpr() * 100.0, c.youden()
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
