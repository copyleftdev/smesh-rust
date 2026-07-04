//! Render the resilience scorecard as a table, JSON, and an HTML chart.

use super::scenario::Attack;
use super::ResilienceReport;

fn attack_color(a: Attack) -> &'static str {
    match a {
        Attack::NodeFailure => "#e8a030",
        Attack::Eclipse => "#f87171",
        Attack::Byzantine => "#c792ea",
        Attack::Partition => "#38bdf8",
    }
}

/// Print the scorecard to stdout.
pub fn print_report(r: &ResilienceReport) {
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║             SMESH Resilience / Byzantine-Fault Scorecard           ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!(
        "Nodes: {}   Topology: {}   Trials/point: {}   Elapsed: {:.1}s\n",
        r.nodes, r.topology, r.trials, r.elapsed_secs
    );

    for c in &r.curves {
        let bp = c
            .breaking_point
            .map(|f| format!("{:.0}%", f * 100.0))
            .unwrap_or_else(|| "survives full sweep".to_string());
        println!("▸ {} ({})", c.attack.label(), c.attack.metric());
        print!("   ");
        for (f, fid) in &c.points {
            print!("{:.0}%:{:.0}  ", f * 100.0, fid * 100.0);
        }
        println!("\n   breaking point (fidelity < 50%): {bp}\n");
    }

    let (rec, during, after) = r.partition;
    println!("▸ Partition + Heal");
    println!(
        "   far-side coverage while split: {:.0}%   after heal: {:.0}%   recovery: {} ticks\n",
        during * 100.0,
        after * 100.0,
        rec
    );
}

/// Serialize to JSON.
pub fn to_json(r: &ResilienceReport) -> String {
    let curves: Vec<serde_json::Value> = r
        .curves
        .iter()
        .map(|c| {
            serde_json::json!({
                "attack": c.attack.code(),
                "label": c.attack.label(),
                "metric": c.attack.metric(),
                "breaking_point": c.breaking_point,
                "points": c.points.iter().map(|(f,v)| serde_json::json!([f,v])).collect::<Vec<_>>(),
            })
        })
        .collect();
    let (rec, during, after) = r.partition;
    serde_json::to_string_pretty(&serde_json::json!({
        "benchmark": "SMESH resilience / Byzantine-fault",
        "nodes": r.nodes,
        "topology": r.topology,
        "trials": r.trials,
        "elapsed_secs": r.elapsed_secs,
        "curves": curves,
        "partition": { "recovery_ticks": rec, "coverage_during": during, "coverage_after": after },
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

/// Render a self-contained HTML scorecard with overlaid degradation curves.
pub fn render_html(r: &ResilienceReport) -> String {
    let (w, h) = (620.0_f64, 460.0_f64);
    let (ml, mt) = (56.0_f64, 24.0_f64);
    let pw = w - ml - 24.0;
    let ph = h - mt - 52.0;
    let maxx = r
        .curves
        .iter()
        .flat_map(|c| c.points.iter())
        .map(|(f, _)| *f)
        .fold(0.0_f64, f64::max)
        .max(0.0001);
    let px = |f: f64| ml + (f / maxx) * pw;
    let py = |v: f64| mt + (1.0 - v) * ph;

    // grid + ticks
    let mut grid = String::new();
    for i in 0..=5 {
        let t = i as f64 / 5.0;
        grid.push_str(&format!(
            "<line class='grid' x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}'/>",
            ml,
            py(t),
            ml + pw,
            py(t)
        ));
        grid.push_str(&format!(
            "<text class='tick' x='{:.1}' y='{:.1}' text-anchor='end'>{}%</text>",
            ml - 8.0,
            py(t) + 4.0,
            (t * 100.0) as i32
        ));
        let xf = maxx * t;
        grid.push_str(&format!(
            "<text class='tick' x='{:.1}' y='{:.1}' text-anchor='middle'>{}%</text>",
            px(xf),
            mt + ph + 18.0,
            (xf * 100.0) as i32
        ));
    }
    // 50% breaking line
    grid.push_str(&format!(
        "<line class='bl' x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}'/><text class='tick' x='{:.1}' y='{:.1}'>consensus breaks ↓</text>",
        ml, py(0.5), ml + pw, py(0.5), ml + 6.0, py(0.5) - 5.0
    ));

    // curves
    let mut paths = String::new();
    let mut legend = String::new();
    for c in &r.curves {
        let col = attack_color(c.attack);
        let pts: String = c
            .points
            .iter()
            .map(|(f, v)| format!("{:.1},{:.1}", px(*f), py(*v)))
            .collect::<Vec<_>>()
            .join(" ");
        paths.push_str(&format!(
            "<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2.5' stroke-linejoin='round'/>"
        ));
        for (f, v) in &c.points {
            paths.push_str(&format!(
                "<circle cx='{:.1}' cy='{:.1}' r='3' fill='{col}'><title>{} @ {:.0}% → {:.0}%</title></circle>",
                px(*f),
                py(*v),
                c.attack.label(),
                f * 100.0,
                v * 100.0
            ));
        }
        if let Some(bp) = c.breaking_point {
            paths.push_str(&format!(
                "<circle cx='{:.1}' cy='{:.1}' r='6' fill='none' stroke='{col}' stroke-width='2'/>",
                px(bp),
                py(0.5)
            ));
        }
        let bp = c
            .breaking_point
            .map(|f| format!("breaks at {:.0}%", f * 100.0))
            .unwrap_or_else(|| "unbroken".to_string());
        legend.push_str(&format!(
            "<div class='lg'><span class='sw' style='background:{col}'></span><b>{}</b> — {bp}</div>",
            c.attack.label()
        ));
    }

    let (rec, during, after) = r.partition;

    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMESH Resilience Scorecard</title>
<style>
  :root{{color-scheme:dark;--bg:#06060c;--card:#13131f;--border:#1e1e32;--fg:#e4e4ed;--mut:#7a7a96;--dim:#4a4a64;--mono:'SF Mono','JetBrains Mono',Consolas,monospace}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:32px}}
  .wrap{{max-width:900px;margin:0 auto}} h1{{font-size:21px;margin:0 0 4px}}
  .meta{{font-family:var(--mono);font-size:12px;color:var(--mut);margin-bottom:22px}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:20px}}
  .h{{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin-bottom:14px}}
  svg{{width:100%;height:auto}} .grid{{stroke:var(--border);stroke-width:1}} .bl{{stroke:#f87171;stroke-width:1;stroke-dasharray:4 4;opacity:.6}}
  .tick{{fill:var(--dim);font-size:10px;font-family:var(--mono)}}
  .axis{{fill:var(--mut);font-size:11px;font-weight:600}}
  .lg{{font-family:var(--mono);font-size:12px;color:var(--fg);margin:6px 0}} .lg b{{font-weight:600}}
  .sw{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:8px;vertical-align:middle}}
  .part{{display:flex;gap:26px;flex-wrap:wrap}} .stat .k{{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}}
  .stat .v{{font-family:var(--mono);font-size:26px;font-weight:700}}
  .note{{color:var(--mut);font-size:13px;margin-top:12px}}
</style></head><body><div class="wrap">
  <h1>SMESH Resilience / Byzantine-Fault Scorecard</h1>
  <div class="meta">{nodes} nodes · {topology} · {trials} trials/point · real engine · {elapsed:.1}s</div>

  <div class="card">
    <div class="h">Consensus fidelity vs attack intensity</div>
    <svg viewBox="0 0 {w:.0} {h:.0}">
      {grid}{paths}
      <text class="axis" x="{axx:.1}" y="{axy:.1}" text-anchor="middle">attack intensity (% of nodes) →</text>
      <text class="axis" transform="translate(14 {ayy:.1}) rotate(-90)" text-anchor="middle">consensus fidelity →</text>
    </svg>
    <div style="margin-top:12px">{legend}</div>
    <div class="note">Fidelity = fraction of honest nodes still reached by the true signal (or, for Byzantine, the honest signal remaining the stronger consensus). Below 50% the mesh has effectively lost consensus.</div>
  </div>

  <div class="card">
    <div class="h">Partition + heal (50/50 split)</div>
    <div class="part">
      <div class="stat"><div class="k">far-side coverage · split</div><div class="v" style="color:#f87171">{during:.0}%</div></div>
      <div class="stat"><div class="k">far-side coverage · healed</div><div class="v" style="color:#34d399">{after:.0}%</div></div>
      <div class="stat"><div class="k">recovery time</div><div class="v">{rec} <span style="font-size:13px;color:var(--mut)">ticks</span></div></div>
    </div>
    <div class="note">The far half is cut off while partitioned, then re-covered once the link heals — signals simply re-diffuse. No manual intervention, no state reconciliation.</div>
  </div>
</div></body></html>"#,
        nodes = r.nodes,
        topology = r.topology,
        trials = r.trials,
        elapsed = r.elapsed_secs,
        w = w, h = h,
        grid = grid, paths = paths, legend = legend,
        axx = ml + pw / 2.0, axy = h - 12.0, ayy = mt + ph / 2.0,
        during = during * 100.0, after = after * 100.0, rec = rec,
    )
}
