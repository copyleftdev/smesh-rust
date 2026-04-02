//! Deep analysis phases for web red teaming
//!
//! Tier 2 and Tier 3 specialists that run AFTER the initial dragnet:
//! - WordPress deep dive (plugins, themes, REST API, xmlrpc)
//! - API endpoint discovery and fuzzing
//! - JavaScript bundle download + static analysis (semgrep, zentinel)
//! - Cloud infrastructure recon (GCP metadata, buckets)
//! - Staging/QA environment raiding
//! - Auth/header/CORS/CSP security analysis

use std::path::Path;

use tracing::debug;

use crate::web_recon::{run_tool, Arsenal, WorkDir};

// ============================================================================
// Phase: WordPress Specialist
// ============================================================================

/// Deep WordPress analysis - plugins, themes, REST API, xmlrpc, user enum
pub async fn phase_wordpress_deep(
    target: &str,
    _arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<WebFinding> {
    println!("\n\x1b[1;36m--- WP Specialist: WordPress Deep Dive ---\x1b[0m\n");

    let mut findings: Vec<WebFinding> = Vec::new();
    let base = format!("https://{}", target);

    // 1. WordPress version + exposed info via REST API
    println!("  [curl] Probing WP REST API root...");
    if let Ok(output) = run_tool(
        "curl",
        &["-s", "-k", "--max-time", "10", &format!("{}/wp-json/", base)],
        15,
    ).await {
        if output.contains("\"name\"") {
            let _ = std::fs::write(work.path_str("wp-rest-root.json"), &output);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&output) {
                let name = v["name"].as_str().unwrap_or("?");
                let desc = v["description"].as_str().unwrap_or("");
                let gmt = v["gmt_offset"].as_f64().unwrap_or(0.0);
                println!("    Site: {} - {}", name, desc);
                println!("    GMT offset: {}", gmt);

                if let Some(namespaces) = v["namespaces"].as_array() {
                    let ns: Vec<&str> = namespaces.iter().filter_map(|n| n.as_str()).collect();
                    println!("    Namespaces: {}", ns.join(", "));

                    // Flag interesting namespaces
                    for ns in &ns {
                        if ns.contains("hubspot") || ns.contains("elementor")
                            || ns.contains("contact-form") || ns.contains("yoast")
                        {
                            findings.push(WebFinding {
                                category: "wp-plugin".into(),
                                severity: "INFO".into(),
                                title: format!("WP Plugin namespace exposed: {}", ns),
                                detail: format!("REST API namespace {} is accessible", ns),
                                url: format!("{}/wp-json/{}", base, ns),
                                evidence: ns.to_string(),
                                confidence: 0.9,
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. User enumeration via REST API
    println!("  [curl] User enumeration via /wp-json/wp/v2/users...");
    if let Ok(output) = run_tool(
        "curl",
        &["-s", "-k", "--max-time", "10", &format!("{}/wp-json/wp/v2/users", base)],
        15,
    ).await {
        if output.starts_with('[') {
            if let Ok(users) = serde_json::from_str::<Vec<serde_json::Value>>(&output) {
                for user in &users {
                    let name = user["name"].as_str().unwrap_or("?");
                    let slug = user["slug"].as_str().unwrap_or("?");
                    println!("    \x1b[33mUser: {} (slug: {})\x1b[0m", name, slug);
                }
                if !users.is_empty() {
                    findings.push(WebFinding {
                        category: "user-enum".into(),
                        severity: "MEDIUM".into(),
                        title: format!("WP REST API user enumeration: {} users exposed", users.len()),
                        detail: "WordPress REST API exposes user information without authentication".into(),
                        url: format!("{}/wp-json/wp/v2/users", base),
                        evidence: format!("{} users found", users.len()),
                        confidence: 0.95,
                    });
                }
            }
        } else if output.contains("cf_chl") || output.contains("challenge") {
            println!("    Cloudflare JS challenge blocking - WAF protected");
            findings.push(WebFinding {
                category: "waf".into(),
                severity: "INFO".into(),
                title: "Cloudflare WAF protecting WP REST API".into(),
                detail: "REST API endpoints trigger Cloudflare JS challenge".into(),
                url: format!("{}/wp-json/wp/v2/users", base),
                evidence: "CF challenge response".into(),
                confidence: 1.0,
            });
        }
    }

    // 3. Author enumeration via ?author=N
    println!("  [curl] Author ID enumeration...");
    for i in 1..=10 {
        if let Ok(output) = run_tool(
            "curl",
            &["-s", "-k", "-o", "/dev/null", "-w", "%{redirect_url}|%{http_code}",
              "--max-time", "5", "-L",
              &format!("{}/?author={}", base, i)],
            8,
        ).await {
            let parts: Vec<&str> = output.split('|').collect();
            if parts.len() >= 2 {
                let redirect = parts[0];
                let code = parts[1];
                if redirect.contains("/author/") {
                    let author = redirect.split("/author/").last().unwrap_or("?").trim_end_matches('/');
                    println!("    Author {}: {} ({})", i, author, code);
                    findings.push(WebFinding {
                        category: "user-enum".into(),
                        severity: "LOW".into(),
                        title: format!("WP author enumeration: ID {} = {}", i, author),
                        detail: "Author ID enumeration via /?author=N redirect".into(),
                        url: format!("{}/?author={}", base, i),
                        evidence: format!("Redirects to /author/{}/", author),
                        confidence: 0.9,
                    });
                }
            }
        }
    }

    // 4. XML-RPC probing
    println!("  [curl] XML-RPC endpoint check...");
    if let Ok(output) = run_tool(
        "curl",
        &["-s", "-k", "--max-time", "10",
          "-X", "POST",
          "-H", "Content-Type: text/xml",
          "-d", "<?xml version='1.0'?><methodCall><methodName>system.listMethods</methodName></methodCall>",
          &format!("{}/xmlrpc.php", base)],
        15,
    ).await {
        if output.contains("methodResponse") && output.contains("listMethods") {
            let method_count = output.matches("<value><string>").count();
            println!("    \x1b[33mXML-RPC enabled! {} methods exposed\x1b[0m", method_count);
            findings.push(WebFinding {
                category: "xmlrpc".into(),
                severity: "MEDIUM".into(),
                title: format!("XML-RPC enabled with {} methods", method_count),
                detail: "XML-RPC is enabled and exposes methods. Can be used for brute force (wp.getUsersBlogs), pingback DDoS, and SSRF.".into(),
                url: format!("{}/xmlrpc.php", base),
                evidence: format!("{} methods available", method_count),
                confidence: 0.95,
            });

            // Check for multicall (amplification attacks)
            if output.contains("system.multicall") {
                findings.push(WebFinding {
                    category: "xmlrpc".into(),
                    severity: "HIGH".into(),
                    title: "XML-RPC system.multicall available (brute force amplification)".into(),
                    detail: "system.multicall allows hundreds of login attempts in a single request, bypassing rate limiting".into(),
                    url: format!("{}/xmlrpc.php", base),
                    evidence: "system.multicall in method list".into(),
                    confidence: 0.95,
                });
            }
        } else if output.contains("XML-RPC server accepts POST requests only") {
            println!("    XML-RPC endpoint exists but needs POST");
        } else {
            println!("    XML-RPC not accessible or blocked");
        }
    }

    // 5. WP-Cron check (can be abused for DoS)
    println!("  [curl] WP-Cron exposure check...");
    if let Ok(output) = run_tool(
        "curl",
        &["-s", "-k", "-o", "/dev/null", "-w", "%{http_code}",
          "--max-time", "10",
          &format!("{}/wp-cron.php", base)],
        15,
    ).await {
        if output.trim() == "200" {
            println!("    \x1b[33mwp-cron.php accessible (200)\x1b[0m");
            findings.push(WebFinding {
                category: "wp-config".into(),
                severity: "LOW".into(),
                title: "WP-Cron publicly accessible".into(),
                detail: "wp-cron.php responds with 200. Can be used for DoS via repeated triggering.".into(),
                url: format!("{}/wp-cron.php", base),
                evidence: "HTTP 200".into(),
                confidence: 0.8,
            });
        }
    }

    // 6. Debug/readme exposure
    println!("  [curl] Checking debug & info exposure...");
    let check_paths = vec![
        ("readme.html", "WP readme.html exposed (version disclosure)"),
        ("wp-config.php.bak", "WP config backup exposed"),
        ("wp-config.php~", "WP config editor backup"),
        (".wp-config.php.swp", "WP config vim swap file"),
        ("wp-includes/version.php", "WP version.php accessible"),
        ("debug.log", "Debug log exposed"),
        ("wp-content/debug.log", "WP debug log exposed"),
        (".git/HEAD", "Git repository exposed"),
        (".env", "Environment file exposed"),
        ("wp-content/uploads/", "Uploads directory listing"),
    ];

    for (path, desc) in check_paths {
        if let Ok(output) = run_tool(
            "curl",
            &["-s", "-k", "-o", "/dev/null", "-w", "%{http_code}|%{size_download}",
              "--max-time", "5", &format!("{}/{}", base, path)],
            8,
        ).await {
            let parts: Vec<&str> = output.split('|').collect();
            if parts.len() >= 2 {
                let code = parts[0].trim();
                let size: u64 = parts[1].trim().parse().unwrap_or(0);
                if code == "200" && size > 0 {
                    let severity = if path.contains("config") || path.contains(".env") || path.contains(".git") || path.contains("debug") {
                        "CRITICAL"
                    } else {
                        "LOW"
                    };
                    println!("    \x1b[33m{} -> {} ({}b)\x1b[0m", path, code, size);
                    findings.push(WebFinding {
                        category: "exposure".into(),
                        severity: severity.into(),
                        title: desc.into(),
                        detail: format!("Path /{} returns HTTP 200 with {} bytes", path, size),
                        url: format!("{}/{}", base, path),
                        evidence: format!("HTTP {} {}b", code, size),
                        confidence: 0.9,
                    });
                }
            }
        }
    }

    println!("  {} findings from WordPress analysis\n", findings.len());
    findings
}

// ============================================================================
// Phase: API Analyst
// ============================================================================

/// Discover and probe API endpoints across all subdomains
pub async fn phase_api_analysis(
    subdomains: &[String],
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<WebFinding> {
    println!("\x1b[1;36m--- API Analyst: Endpoint Discovery ---\x1b[0m\n");

    let mut findings: Vec<WebFinding> = Vec::new();

    // 1. Probe common API paths on all subdomains
    let api_paths = vec![
        "/api", "/api/v1", "/api/v2", "/api/health", "/api/status",
        "/swagger.json", "/openapi.json", "/api-docs", "/swagger-ui.html",
        "/graphql", "/graphiql", "/.well-known/openid-configuration",
        "/oauth/token", "/auth/login", "/auth/register",
        "/health", "/healthz", "/ready", "/metrics", "/debug/vars",
        "/actuator", "/actuator/health", "/actuator/env",
        "/v1/docs", "/v2/docs", "/redoc",
    ];

    // Focus on non-cloudflare subdomains (direct GCP IPs)
    let api_subs: Vec<&String> = subdomains.iter()
        .filter(|s| s.contains("api-") || s.contains("developer") || s.contains("auth") || s.contains("tech-ops"))
        .collect();

    println!("  Probing {} API subdomains with {} paths...", api_subs.len(), api_paths.len());

    for sub in &api_subs {
        for proto in &["https", "http"] {
            let base = format!("{}://{}", proto, sub);

            for path in &api_paths {
                if let Ok(output) = run_tool(
                    "curl",
                    &["-s", "-k", "-o", "/dev/null",
                      "-w", "%{http_code}|%{content_type}|%{size_download}",
                      "--max-time", "5",
                      "--connect-timeout", "3",
                      &format!("{}{}", base, path)],
                    8,
                ).await {
                    let parts: Vec<&str> = output.split('|').collect();
                    if parts.len() >= 3 {
                        let code = parts[0].trim();
                        let content_type = parts[1].trim();
                        let size: u64 = parts[2].trim().parse().unwrap_or(0);

                        // Interesting responses: 200, 401, 403, 405 (method not allowed = endpoint exists)
                        if ["200", "401", "403", "405", "301", "302"].contains(&code) && size > 0 {
                            let severity = match (code, *path) {
                                ("200", p) if p.contains("swagger") || p.contains("openapi") || p.contains("graphi") => "HIGH",
                                ("200", p) if p.contains("actuator") || p.contains("debug") || p.contains("metrics") => "HIGH",
                                ("200", p) if p.contains("health") || p.contains("status") => "LOW",
                                ("401", _) => "INFO", // Auth required = endpoint exists
                                ("403", _) => "INFO",
                                _ => "LOW",
                            };

                            println!("    [{:>3}] {}{} [{}] ({}b)",
                                code, sub, path, content_type, size);

                            findings.push(WebFinding {
                                category: "api-endpoint".into(),
                                severity: severity.into(),
                                title: format!("API endpoint: {}{} ({})", sub, path, code),
                                detail: format!("Content-Type: {}, Size: {}b", content_type, size),
                                url: format!("{}{}", base, path),
                                evidence: format!("HTTP {} {} {}b", code, content_type, size),
                                confidence: if code == "200" { 0.95 } else { 0.7 },
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. Nuclei API-specific templates on discovered endpoints
    if arsenal.has("nuclei") && !findings.is_empty() {
        let api_targets = work.path_str("api-targets.txt");
        let api_urls: Vec<String> = findings.iter().map(|f| f.url.clone()).collect();
        let _ = std::fs::write(&api_targets, api_urls.join("\n"));

        println!("\n  [nuclei] API-specific template scan...");
        let api_nuclei = work.path_str("nuclei-api.json");
        match run_tool(
            "nuclei",
            &["-list", &api_targets,
              "-tags", "api,swagger,graphql,exposure,token,default-login",
              "-severity", "critical,high,medium",
              "-json-export", &api_nuclei,
              "-silent", "-rate-limit", "30", "-timeout", "10"],
            300,
        ).await {
            Ok(_) => {
                if let Ok(content) = std::fs::read_to_string(&api_nuclei) {
                    let count = content.lines().filter(|l| !l.trim().is_empty()).count();
                    if count > 0 {
                        println!("    {} nuclei API findings", count);
                    }
                }
            }
            Err(e) => debug!("nuclei api scan: {}", e),
        }
    }

    println!("  {} findings from API analysis\n", findings.len());
    findings
}

// ============================================================================
// Phase: JavaScript Decompiler + Static Analysis
// ============================================================================

/// Download JS bundles and run semgrep + zentinel
pub async fn phase_js_analysis(
    target: &str,
    endpoints: &[String],
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<WebFinding> {
    println!("\x1b[1;36m--- JS Decompiler: Static Analysis on Downloaded JS ---\x1b[0m\n");

    let mut findings: Vec<WebFinding> = Vec::new();
    let js_dir = work.root.join("js_bundles");
    let _ = std::fs::create_dir_all(&js_dir);

    // 1. Collect JS URLs from crawled endpoints
    let js_urls: Vec<&String> = endpoints.iter()
        .filter(|u| {
            let lower = u.to_lowercase();
            lower.ends_with(".js") && !lower.contains("jquery")
                && !lower.contains("bootstrap") && !lower.contains("twemoji")
                && !lower.contains("popper") && !lower.contains("gtm.js")
                && !lower.contains("wp-emoji")
        })
        .collect();

    println!("  {} interesting JS files to analyze (filtered from {})", js_urls.len(), endpoints.len());

    // 2. Download JS files
    let mut downloaded = Vec::new();
    for (i, url) in js_urls.iter().take(30).enumerate() {
        let filename = format!("bundle_{}.js", i);
        let filepath = js_dir.join(&filename);

        if let Ok(_) = run_tool(
            "curl",
            &["-s", "-k", "--max-time", "10", "-o", &filepath.to_string_lossy(), url],
            15,
        ).await {
            if filepath.exists() {
                let size = std::fs::metadata(&filepath).map(|m| m.len()).unwrap_or(0);
                if size > 100 {
                    downloaded.push((filepath, url.to_string()));
                }
            }
        }
    }
    println!("  Downloaded {} JS bundles\n", downloaded.len());

    if downloaded.is_empty() {
        return findings;
    }

    // 3. Semgrep scan on downloaded JS
    if arsenal.has("semgrep") {
        println!("  [semgrep] Scanning JS bundles for vulnerabilities...");
        let semgrep_out = work.path_str("semgrep-js.json");
        match run_tool(
            "semgrep",
            &["scan",
              "--config", "p/javascript",
              "--config", "p/security-audit",
              "--json", "--output", &semgrep_out,
              "--quiet",
              &js_dir.to_string_lossy()],
            120,
        ).await {
            Ok(_) => {
                if let Ok(content) = std::fs::read_to_string(&semgrep_out) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(results) = v["results"].as_array() {
                            println!("    {} semgrep findings", results.len());
                            for r in results.iter().take(20) {
                                let check = r["check_id"].as_str().unwrap_or("?");
                                let msg = r["extra"]["message"].as_str().unwrap_or("");
                                let sev = r["extra"]["severity"].as_str().unwrap_or("WARNING");
                                let path = r["path"].as_str().unwrap_or("?");
                                let line = r["start"]["line"].as_u64().unwrap_or(0);

                                println!("    [{}] {} - {} ({}:{})", sev, check, msg, path, line);

                                findings.push(WebFinding {
                                    category: "js-vuln".into(),
                                    severity: semgrep_severity(sev),
                                    title: format!("{}: {}", check, truncate(msg, 80)),
                                    detail: msg.to_string(),
                                    url: format!("JS from {}", target),
                                    evidence: format!("{}:{}", path, line),
                                    confidence: 0.7,
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => debug!("semgrep js: {}", e),
        }
    }

    // 4. Zentinel scan on downloaded JS
    if arsenal.has("zent") {
        println!("  [zentinel] Fast static analysis on JS bundles...");
        let zent_rules = find_zentinel_rules().await;
        if let Some(rules) = zent_rules {
            let zent_out = work.path_str("zentinel-js.json");
            match run_tool(
                "zent",
                &["scan", &js_dir.to_string_lossy(), "--config", &rules, "--format", "agent"],
                60,
            ).await {
                Ok(output) => {
                    let _ = std::fs::write(&zent_out, &output);
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&output) {
                        if let Some(zfindings) = v["findings"].as_array() {
                            println!("    {} zentinel findings", zfindings.len());
                            for f in zfindings.iter().take(20) {
                                let msg = f["message"].as_str().unwrap_or("?");
                                let sev = f["severity"].as_str().unwrap_or("WARNING");
                                let file = f["file"].as_str().unwrap_or("?");
                                let line = f["line"].as_u64().unwrap_or(0);
                                let cat = f["category"].as_str().unwrap_or("unknown");

                                println!("    [{}] {} - {} ({}:{})", sev, cat, msg, file, line);

                                findings.push(WebFinding {
                                    category: format!("zentinel-{}", cat),
                                    severity: zentinel_severity(sev),
                                    title: format!("{}: {}", cat, truncate(msg, 80)),
                                    detail: msg.to_string(),
                                    url: format!("JS from {}", target),
                                    evidence: format!("{}:{}", file, line),
                                    confidence: f["confidence"].as_f64().unwrap_or(0.7),
                                });
                            }
                        }
                    }
                }
                Err(e) => debug!("zentinel js: {}", e),
            }
        } else {
            println!("    No zentinel rules found, skipping");
        }
    }

    // 5. Grep for hardcoded secrets in JS
    println!("  [grep] Scanning JS for hardcoded secrets...");
    let secret_patterns = [
        (r#"(?i)(api[_-]?key|apikey)\s*[:=]\s*['"][a-zA-Z0-9_\-]{16,}['"]"#, "API key"),
        (r#"(?i)(secret|password|passwd|token)\s*[:=]\s*['"][^'"]{8,}['"]"#, "Secret/Password"),
        (r#"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}"#, "Bearer token"),
        (r#"(?i)aws[_-]?(access|secret|key)"#, "AWS credential"),
        (r#"(?i)(sk-|pk_live_|sk_live_|rk_live_)"#, "Stripe/API live key"),
        (r#"(?i)ghp_[a-zA-Z0-9]{36}"#, "GitHub PAT"),
        (r#"(?i)eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}"#, "JWT token"),
    ];

    for (pattern, label) in &secret_patterns {
        if let Ok(output) = run_tool(
            "bash",
            &["-c", &format!("grep -rPn '{}' {} 2>/dev/null | head -5", pattern, js_dir.display())],
            10,
        ).await {
            for line in output.lines() {
                if !line.trim().is_empty() {
                    println!("    \x1b[1;31m[SECRET] {} -> {}\x1b[0m", label, truncate(line, 100));
                    findings.push(WebFinding {
                        category: "secret-exposure".into(),
                        severity: "CRITICAL".into(),
                        title: format!("Hardcoded {} in JavaScript", label),
                        detail: truncate(line, 200),
                        url: format!("JS from {}", target),
                        evidence: truncate(line, 100),
                        confidence: 0.85,
                    });
                }
            }
        }
    }

    println!("  {} findings from JS analysis\n", findings.len());
    findings
}

// ============================================================================
// Phase: Cloud Infrastructure Recon
// ============================================================================

/// GCP-specific recon on direct-IP API backends
pub async fn phase_cloud_recon(
    subdomains: &[String],
    _arsenal: &Arsenal,
    _work: &WorkDir,
) -> Vec<WebFinding> {
    println!("\x1b[1;36m--- Cloud Recon: GCP Infrastructure ---\x1b[0m\n");

    let mut findings: Vec<WebFinding> = Vec::new();

    // Identify GCP-hosted subdomains (34.x.x.x, 35.x.x.x ranges)
    let gcp_subs: Vec<&String> = subdomains.iter()
        .filter(|s| s.contains("api-") || s.contains("developer") || s.contains("tech-ops"))
        .collect();

    println!("  {} GCP-hosted subdomains identified\n", gcp_subs.len());

    // 1. Check for exposed GCP metadata on direct IP endpoints
    for sub in &gcp_subs {
        // Check common cloud debug/health endpoints
        let cloud_paths = vec![
            ("/server-status", "Apache server-status"),
            ("/nginx_status", "Nginx status"),
            ("/_debug", "Debug endpoint"),
            ("/_health", "Health endpoint"),
            ("/env", "Environment variables"),
            ("/config", "Configuration endpoint"),
            ("/info", "Info endpoint"),
            ("/.env", "Environment file"),
            ("/robots.txt", "Robots.txt"),
        ];

        for (path, desc) in &cloud_paths {
            for proto in &["https", "http"] {
                if let Ok(output) = run_tool(
                    "curl",
                    &["-s", "-k", "-o", "/dev/null",
                      "-w", "%{http_code}|%{size_download}",
                      "--max-time", "3", "--connect-timeout", "2",
                      &format!("{}://{}{}", proto, sub, path)],
                    6,
                ).await {
                    let parts: Vec<&str> = output.split('|').collect();
                    if parts.len() >= 2 {
                        let code = parts[0].trim();
                        let size: u64 = parts[1].trim().parse().unwrap_or(0);
                        if code == "200" && size > 50 {
                            println!("    \x1b[33m[{}] {}://{}{} ({}b)\x1b[0m",
                                code, proto, sub, path, size);
                            findings.push(WebFinding {
                                category: "cloud-exposure".into(),
                                severity: if path.contains("env") || path.contains("config") || path.contains("debug") { "HIGH" } else { "LOW" }.into(),
                                title: format!("{} on {}", desc, sub),
                                detail: format!("{}://{}{} responds with HTTP {} ({}b)", proto, sub, path, code, size),
                                url: format!("{}://{}{}", proto, sub, path),
                                evidence: format!("HTTP {} {}b", code, size),
                                confidence: 0.8,
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. GCP bucket enumeration based on naming patterns
    println!("  [curl] Checking for exposed GCP storage buckets...");
    let bucket_patterns = vec![
        "zuub", "zuub-prod", "zuub-staging", "zuub-dev", "zuub-backup",
        "zuub-uploads", "zuub-data", "zuub-assets", "zuub-api",
        "zuub-dental", "zuub-patient", "zuub-hipaa",
    ];

    for bucket in &bucket_patterns {
        if let Ok(output) = run_tool(
            "curl",
            &["-s", "-k", "-o", "/dev/null",
              "-w", "%{http_code}",
              "--max-time", "5",
              &format!("https://storage.googleapis.com/{}/", bucket)],
            8,
        ).await {
            let code = output.trim();
            match code {
                "200" => {
                    println!("    \x1b[1;31m[OPEN] gs://{} - PUBLIC READ\x1b[0m", bucket);
                    findings.push(WebFinding {
                        category: "bucket-exposure".into(),
                        severity: "CRITICAL".into(),
                        title: format!("GCP bucket publicly readable: gs://{}", bucket),
                        detail: "Bucket allows unauthenticated listing/read".into(),
                        url: format!("https://storage.googleapis.com/{}/", bucket),
                        evidence: "HTTP 200".into(),
                        confidence: 0.95,
                    });
                }
                "403" => {
                    println!("    [EXISTS] gs://{} - access denied (bucket exists)", bucket);
                    findings.push(WebFinding {
                        category: "bucket-enum".into(),
                        severity: "INFO".into(),
                        title: format!("GCP bucket exists: gs://{}", bucket),
                        detail: "Bucket returns 403 (exists but access denied)".into(),
                        url: format!("https://storage.googleapis.com/{}/", bucket),
                        evidence: "HTTP 403".into(),
                        confidence: 0.8,
                    });
                }
                _ => {} // 404 = doesn't exist
            }
        }
    }

    println!("  {} findings from cloud recon\n", findings.len());
    findings
}

// ============================================================================
// Phase: Staging Raider
// ============================================================================

/// Specifically target non-production environments
pub async fn phase_staging_raid(
    subdomains: &[String],
    arsenal: &Arsenal,
    _work: &WorkDir,
) -> Vec<WebFinding> {
    println!("\x1b[1;36m--- Staging Raider: Non-Production Environments ---\x1b[0m\n");

    let mut findings: Vec<WebFinding> = Vec::new();

    let staging_subs: Vec<&String> = subdomains.iter()
        .filter(|s| {
            s.contains("staging") || s.contains("qa") || s.contains("sandbox")
                || s.contains("-dev") || s.contains("test")
        })
        .collect();

    println!("  {} non-production subdomains found\n", staging_subs.len());

    for sub in &staging_subs {
        println!("  Raiding {}...", sub);

        // httpx detailed probe
        if arsenal.has("httpx") {
            if let Ok(output) = run_tool(
                "bash",
                &["-c", &format!(
                    "echo '{}' | httpx -silent -json -title -tech-detect -status-code -content-length -follow-redirects 2>/dev/null",
                    sub
                )],
                30,
            ).await {
                for line in output.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        let url = v["url"].as_str().unwrap_or("?");
                        let status = v["status_code"].as_i64().unwrap_or(0);
                        let title = v["title"].as_str().unwrap_or("");
                        let tech = v["tech"].as_array()
                            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
                            .unwrap_or_default();

                        println!("    [{:>3}] {} \"{}\" [{}]", status, url, title, tech);

                        if status == 200 {
                            findings.push(WebFinding {
                                category: "staging-exposure".into(),
                                severity: "HIGH".into(),
                                title: format!("Staging/QA environment accessible: {}", sub),
                                detail: format!("Title: {}, Tech: {}", title, tech),
                                url: url.to_string(),
                                evidence: format!("HTTP {} - {}", status, title),
                                confidence: 0.9,
                            });
                        }
                    }
                }
            }
        }

        // Check for debug/swagger on staging
        let staging_paths = vec![
            "/swagger-ui.html", "/api-docs", "/graphql", "/graphiql",
            "/debug", "/debug/vars", "/metrics", "/actuator",
            "/health", "/admin", "/console",
        ];

        for path in &staging_paths {
            for proto in &["https", "http"] {
                if let Ok(output) = run_tool(
                    "curl",
                    &["-s", "-k", "-o", "/dev/null",
                      "-w", "%{http_code}|%{size_download}",
                      "--max-time", "3", "--connect-timeout", "2",
                      &format!("{}://{}{}", proto, sub, path)],
                    6,
                ).await {
                    let parts: Vec<&str> = output.split('|').collect();
                    if parts.len() >= 2 {
                        let code = parts[0].trim();
                        let size: u64 = parts[1].trim().parse().unwrap_or(0);
                        if (code == "200" || code == "401" || code == "403") && size > 50 {
                            let severity = if code == "200" && (path.contains("swagger") || path.contains("debug") || path.contains("admin") || path.contains("graphi")) {
                                "CRITICAL"
                            } else if code == "200" {
                                "HIGH"
                            } else {
                                "MEDIUM"
                            };

                            println!("    \x1b[33m[{}] {}://{}{} ({}b)\x1b[0m", code, proto, sub, path, size);
                            findings.push(WebFinding {
                                category: "staging-endpoint".into(),
                                severity: severity.into(),
                                title: format!("{} on staging: {}{}", path, sub, path),
                                detail: format!("Non-production endpoint accessible"),
                                url: format!("{}://{}{}", proto, sub, path),
                                evidence: format!("HTTP {} {}b", code, size),
                                confidence: 0.85,
                            });
                        }
                    }
                }
            }
        }
    }

    println!("  {} findings from staging raid\n", findings.len());
    findings
}

// ============================================================================
// Phase: Auth & Security Headers
// ============================================================================

/// Analyze authentication mechanisms, security headers, CORS, CSP
pub async fn phase_auth_and_headers(
    target: &str,
    subdomains: &[String],
    _arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<WebFinding> {
    println!("\x1b[1;36m--- Auth Tester: Security Headers & CORS ---\x1b[0m\n");

    let mut findings: Vec<WebFinding> = Vec::new();

    // Test main target + key subdomains
    let mut test_hosts = vec![target.to_string()];
    test_hosts.extend(
        subdomains.iter()
            .filter(|s| s.contains("dental") || s.contains("auth") || s.contains("patient") || s.contains("api-production"))
            .take(5)
            .cloned()
    );

    for host in &test_hosts {
        println!("  Analyzing headers for {}...", host);

        if let Ok(output) = run_tool(
            "curl",
            &["-s", "-k", "-I", "--max-time", "10",
              &format!("https://{}", host)],
            15,
        ).await {
            let headers: Vec<&str> = output.lines().collect();
            let _ = std::fs::write(
                work.path_str(&format!("headers-{}.txt", host.replace('.', "_"))),
                &output,
            );

            let has_header = |name: &str| -> bool {
                headers.iter().any(|h| h.to_lowercase().starts_with(&name.to_lowercase()))
            };

            let get_header = |name: &str| -> Option<String> {
                headers.iter()
                    .find(|h| h.to_lowercase().starts_with(&name.to_lowercase()))
                    .map(|h| h.to_string())
            };

            // Missing security headers
            let required = vec![
                ("Strict-Transport-Security", "HSTS", "MEDIUM"),
                ("Content-Security-Policy", "CSP", "MEDIUM"),
                ("X-Content-Type-Options", "X-Content-Type-Options", "LOW"),
                ("X-Frame-Options", "X-Frame-Options (clickjacking)", "LOW"),
                ("X-XSS-Protection", "X-XSS-Protection", "INFO"),
                ("Referrer-Policy", "Referrer-Policy", "LOW"),
                ("Permissions-Policy", "Permissions-Policy", "LOW"),
            ];

            for (header, label, severity) in &required {
                if !has_header(header) {
                    println!("    \x1b[33mMissing: {}\x1b[0m", label);
                    findings.push(WebFinding {
                        category: "missing-header".into(),
                        severity: severity.to_string(),
                        title: format!("Missing {} header on {}", label, host),
                        detail: format!("The {} response header is not set", header),
                        url: format!("https://{}", host),
                        evidence: format!("Header {} absent", header),
                        confidence: 0.95,
                    });
                }
            }

            // Check for information leakage in headers
            if let Some(server) = get_header("Server") {
                println!("    Server: {}", server.trim());
                if server.contains("Apache") || server.contains("nginx") {
                    findings.push(WebFinding {
                        category: "info-disclosure".into(),
                        severity: "LOW".into(),
                        title: format!("Server version disclosed: {}", server.trim()),
                        detail: "Server header reveals software version".into(),
                        url: format!("https://{}", host),
                        evidence: server.trim().to_string(),
                        confidence: 0.9,
                    });
                }
            }

            if let Some(powered) = get_header("X-Powered-By") {
                println!("    \x1b[33mX-Powered-By: {}\x1b[0m", powered.trim());
                findings.push(WebFinding {
                    category: "info-disclosure".into(),
                    severity: "LOW".into(),
                    title: format!("X-Powered-By disclosed: {}", powered.trim()),
                    detail: "X-Powered-By header reveals technology".into(),
                    url: format!("https://{}", host),
                    evidence: powered.trim().to_string(),
                    confidence: 0.95,
                });
            }
        }

        // CORS testing
        println!("  Testing CORS on {}...", host);
        if let Ok(output) = run_tool(
            "curl",
            &["-s", "-k", "-I", "--max-time", "10",
              "-H", "Origin: https://evil.com",
              &format!("https://{}", host)],
            15,
        ).await {
            let lower = output.to_lowercase();
            if lower.contains("access-control-allow-origin: https://evil.com")
                || lower.contains("access-control-allow-origin: *")
            {
                let wildcard = lower.contains("access-control-allow-origin: *");
                println!("    \x1b[1;31mCORS misconfiguration! {}\x1b[0m",
                    if wildcard { "Wildcard origin (*)" } else { "Reflects arbitrary origin" });

                findings.push(WebFinding {
                    category: "cors".into(),
                    severity: if wildcard { "MEDIUM" } else { "HIGH" }.into(),
                    title: format!("CORS misconfiguration on {}", host),
                    detail: if wildcard {
                        "Access-Control-Allow-Origin: * allows any origin".into()
                    } else {
                        "Server reflects arbitrary Origin header, allowing cross-origin requests from any domain".into()
                    },
                    url: format!("https://{}", host),
                    evidence: "Tested with Origin: https://evil.com".into(),
                    confidence: 0.95,
                });
            }
        }
    }

    println!("  {} findings from auth/header analysis\n", findings.len());
    findings
}

// ============================================================================
// Types & Helpers
// ============================================================================

/// A finding from web deep analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebFinding {
    pub category: String,
    pub severity: String,
    pub title: String,
    pub detail: String,
    pub url: String,
    pub evidence: String,
    pub confidence: f64,
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max.saturating_sub(3)]) }
}

fn semgrep_severity(sev: &str) -> String {
    match sev.to_uppercase().as_str() {
        "ERROR" => "HIGH".into(),
        "WARNING" => "MEDIUM".into(),
        "INFO" => "LOW".into(),
        _ => "MEDIUM".into(),
    }
}

fn zentinel_severity(sev: &str) -> String {
    match sev.to_uppercase().as_str() {
        "ERROR" => "HIGH".into(),
        "WARNING" => "MEDIUM".into(),
        "INFO" => "LOW".into(),
        _ => "MEDIUM".into(),
    }
}

/// Find zentinel rules directory
async fn find_zentinel_rules() -> Option<String> {
    let candidates = vec![
        "/home/ops/.local/share/zentinel/rules",
        "/usr/local/share/zentinel/rules",
        "/opt/zentinel/rules",
    ];
    for path in candidates {
        if Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    // Try to find via zent binary location
    if let Ok(output) = run_tool("bash", &["-c", "dirname $(which zent) 2>/dev/null"], 5).await {
        let rules = format!("{}/rules", output.trim());
        if Path::new(&rules).exists() {
            return Some(rules);
        }
    }
    None
}
