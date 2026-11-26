use clap::Parser;
use ocak_engine::BackendKind;
use ocak_server::{build_router, AppState, ServerConfig};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ocakd", about = "Ocak ground-up LLM server daemon")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8090)]
    port: u16,
    #[arg(long)]
    compat_port: Option<u16>,
    #[arg(long, env = "OCAK_MODELS_DIR")]
    models_dir: Option<PathBuf>,
    #[arg(long)]
    extra_models_dir: Option<PathBuf>,
    #[arg(long)]
    presets: Option<PathBuf>,
    #[arg(long, default_value_t = 120)]
    sleep_idle_seconds: u64,
    #[arg(long, default_value_t = 2048)]
    max_ctx: usize,
    #[arg(long, default_value = "vk")]
    backend: String,
    #[arg(long, env = "OCAK_API_KEY")]
    api_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let backend = if args.backend == "cpu" { BackendKind::CpuAvx2 } else { BackendKind::Vulkan };
    let allow_remote_host = !matches!(args.host.as_str(), "127.0.0.1" | "localhost" | "::1")
        || std::env::var("OCAK_ALLOW_REMOTE_HOST").ok().as_deref() == Some("1");
    let models_dir = args.models_dir.unwrap_or_else(|| {
        std::env::var("HOME")
            .map(|h| PathBuf::from(h).join("models"))
            .unwrap_or_else(|_| PathBuf::from("./models"))
    });
    let presets_path = args.presets.unwrap_or_else(|| models_dir.join("models-preset.ini"));
    let mut models_dirs = vec![models_dir];
    if let Some(extra) = args.extra_models_dir {
        models_dirs.push(extra);
    }
    let cfg = ServerConfig {
        models_dirs,
        presets_path,
        sleep_idle_seconds: args.sleep_idle_seconds,
        backend,
        max_ctx: args.max_ctx,
        api_key: args.api_key.filter(|s| !s.is_empty()),
        allow_remote_host,
    };
    let state = AppState::new(cfg);
    state.spawn_watchdog();
    tracing::info!(models = state.catalog.len(), sleep_idle_seconds = args.sleep_idle_seconds, "ocakd catalog ready");

    if let Some(cp) = args.compat_port {
        let addr: SocketAddr = format!("{}:{cp}", args.host).parse()?;
        let router = build_router(state.clone());
        tokio::spawn(async move {
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    tracing::info!(%addr, "ocakd compat listener online");
                    let _ = axum::serve(listener, router).await;
                }
                Err(e) => {
                    tracing::warn!(%addr, error = %e, "ocakd compat listener failed to bind; continuing on primary port");
                }
            }
        });
    }

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, ui = %format!("http://{addr}/ui"), "ocakd listening");
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
