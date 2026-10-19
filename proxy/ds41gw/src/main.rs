//! ds41gw の起動。使い方: `ds41gw --config /etc/ds41gw/ds41gw.toml`

use std::sync::atomic::Ordering;
use std::time::Duration;

use ds41gw::config::Config;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg_path: Option<String> = None;
    let mut check = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                cfg_path = args.get(i + 1).cloned();
                i += 1;
            }
            "--check" => check = true,
            "--probe" => {
                // docker の HEALTHCHECK 用 (curl が無いイメージで使う)
                let url = args.get(i + 1).cloned().unwrap_or_else(|| "http://127.0.0.1:30001/health".into());
                std::process::exit(if probe(&url) { 0 } else { 1 });
            }
            "--version" | "-V" => {
                println!("ds41gw {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            "--help" | "-h" => {
                println!("usage: ds41gw [--config FILE] [--check] [--probe URL]");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let cfg = match cfg_path {
        Some(p) => match Config::load(std::path::Path::new(&p)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ds41gw: {e}");
                std::process::exit(2);
            }
        },
        None => Config::default(),
    };
    if check {
        println!("{cfg:#?}");
        return;
    }
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    rt.block_on(run(cfg));
}

async fn run(cfg: Config) {
    let listen = cfg.server.listen.clone();
    let grace = cfg.server.shutdown_grace_s;
    let state = ds41gw::build_state(cfg);
    ds41gw::models::spawn_pollers(state.clone());
    let router = ds41gw::router(state.clone());
    let (tx, _) = tokio::sync::broadcast::channel::<()>(1);
    eprintln!(
        "ds41gw {} upstream={} tokenizer={} listen={:?}",
        env!("CARGO_PKG_VERSION"),
        state.cfg.upstream.url,
        state.counter.has_tokenizer(),
        listen
    );
    let serve_opts = ds41gw::ServeOpts::from_config(&state.cfg.server);
    for addr in listen {
        let router = router.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            // bind に失敗したら (tailscaled が未起動など) 5 秒ごとにやり直す
            let listener = loop {
                match tokio::net::TcpListener::bind(&addr).await {
                    Ok(l) => break l,
                    Err(e) => {
                        eprintln!("ds41gw: bind {addr} に失敗 ({e})。5 秒後に再試行");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            };
            eprintln!("ds41gw: listening on {addr}");
            let mut rx = tx.subscribe();
            let shutdown = async move {
                let _ = rx.recv().await;
            };
            if let Err(e) = ds41gw::serve_listener_with(listener, router, serve_opts, shutdown).await {
                eprintln!("ds41gw: serve {addr}: {e}");
            }
        });
    }
    wait_signal().await;
    eprintln!("ds41gw: 停止の合図を受けた。新しい要求を断り、最大 {grace} 秒待つ");
    state.up.shutting_down.store(true, Ordering::Relaxed);
    let _ = tx.send(());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(grace);
    while tokio::time::Instant::now() < deadline && state.up.inflight.load(Ordering::Relaxed) > 0 {
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // 要求ログを書き切る
    tokio::time::sleep(Duration::from_millis(300)).await;
    std::process::exit(0);
}

fn probe(url: &str) -> bool {
    use std::io::{Read, Write};
    let rest = url.trim_start_matches("http://");
    let (hostport, path) = match rest.find('/') {
        Some(p) => (&rest[..p], &rest[p..]),
        None => (rest, "/"),
    };
    let Ok(mut addrs) = std::net::ToSocketAddrs::to_socket_addrs(hostport) else { return false };
    let Some(addr) = addrs.next() else { return false };
    let Ok(mut s) = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_secs(3)));
    if s.write_all(format!("GET {path} HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\n\r\n").as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let n = s.read(&mut buf).unwrap_or(0);
    String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200")
}

async fn wait_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("sigterm");
        let mut int = signal(SignalKind::interrupt()).expect("sigint");
        tokio::select! {
            _ = term.recv() => {},
            _ = int.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
