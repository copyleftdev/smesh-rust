//! Embedded visualization server
//!
//! Serves the SMESH signal field visualization from a single binary.
//! `smesh viz` → serves on :8080, zero dependencies.

use std::io::Write;
use std::net::TcpListener;

/// The visualization HTML, embedded at compile time
const VIZ_HTML: &str = include_str!("assets/viz.html");

/// Serve the visualization on the given port
pub fn serve(port: u16) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr)?;

    println!("SMESH Signal Field Visualization");
    println!("  Serving on http://localhost:{}", port);
    println!("  Press Ctrl+C to stop\n");

    // Try to open browser
    let url = format!("http://localhost:{}", port);
    let _ = open_browser(&url);

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Read request (we don't care about the contents — serve the same page for everything)
        let mut buf = [0u8; 1024];
        let _ = std::io::Read::read(&mut stream, &mut buf);

        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Cache-Control: no-cache\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            VIZ_HTML.len(),
            VIZ_HTML
        );

        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    }

    Ok(())
}

/// Try to open the URL in the default browser
fn open_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", url])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
