//! Server tests (end-to-end ones need models in ~/models and a GPU).
use axum::http::StatusCode;
use axum::Router;
use ocak_engine::{GenRequest, SamplerParams};
use ocak_sysfs::ModelResidencyState;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use crate::*;
#[test]
fn loopback_host_and_origin_are_strict() {
    for h in ["localhost", "localhost:8080", "127.0.0.1:1", "[::1]", "[::1]:443", "LocalHost:80"] {
        assert!(crate::is_loopback_host(h), "{h}");
    }
    for h in ["evil.example", "localhost.evil", "localhost:80@evil.example", "[::1]junk", "[::1]:", "127.0.0.1:x", "::1"] {
        assert!(!crate::is_loopback_host(h), "{h}");
    }
    assert!(crate::is_loopback_origin("http://localhost:3000"));
    assert!(!crate::is_loopback_origin("http://localhost:80@evil.example"));
    assert!(!crate::is_loopback_origin("null"));
    assert_eq!(crate::query_decode("a%2Bb%3D%2F+c%"), "a+b=/ c%");
}

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, String) {
    let req = match body {
        Some(v) => Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => Request::builder().method(method).uri(uri).body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let st = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (st, String::from_utf8_lossy(&bytes).into_owned())
}

fn find_test_tinyllama() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let candidates = [
        PathBuf::from(&home).join("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        PathBuf::from(&home).join("Projeler/mergen/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[tokio::test]
async fn end_to_end_openai_anthropic_embeddings_radix_kv_and_warm_sleep() {
    let Some(model_path) = find_test_tinyllama() else {
        return;
    };
    let mut cfg = ServerConfig { sleep_idle_seconds: 2, ..Default::default() };
    if let Some(dir) = model_path.parent() {
        if !cfg.models_dirs.iter().any(|d| d == dir) {
            cfg.models_dirs.push(dir.to_path_buf());
        }
    }
    let state = AppState::new(cfg);
    state.spawn_watchdog();
    let app = build_router(state.clone());

    // 1. /v1/models & /ui & /metrics
    let (st, body) = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(st, StatusCode::OK);
    let models: Value = serde_json::from_str(&body).unwrap();
    assert!(models["data"].as_array().unwrap().len() >= 2);
    let (st_ui, ui_html) = call(&app, "GET", "/ui", None).await;
    assert_eq!(st_ui, StatusCode::OK);
    assert!(ui_html.contains("Message Ocak") && ui_html.contains("/v1/chat/completions"));

    // 2. Turn 1: OpenAI chat completion
    let sys_msg = "You are a helpful, concise computer architecture assistant answering questions accurately in English.";
    let (st1, b1) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [
                { "role": "system", "content": sys_msg },
                { "role": "user", "content": "What is the capital of France?" }
            ]
        })),
    )
    .await;
    assert_eq!(st1, StatusCode::OK, "{b1}");
    let j1: Value = serde_json::from_str(&b1).unwrap();
    let reply1 = j1["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    println!("Turn 1: {:?} | timings: {}", reply1, j1["timings"]);

    // 3. Turn 2: Same prefix -> Radix KV prefix cache hit (cache_n >= 16)
    let (st2, b2) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "max_tokens": 16,
            "temperature": 0.0,
            "messages": [
                { "role": "system", "content": sys_msg },
                { "role": "user", "content": "What is the capital of France?" },
                { "role": "assistant", "content": reply1 },
                { "role": "user", "content": "And Spain?" }
            ]
        })),
    )
    .await;
    assert_eq!(st2, StatusCode::OK, "{b2}");
    let j2: Value = serde_json::from_str(&b2).unwrap();
    let cache_n = j2["timings"]["cache_n"].as_u64().unwrap();
    println!("Turn 2: {:?} | cache_n={cache_n}", j2["choices"][0]["message"]["content"]);
    assert!(cache_n >= 16, "expected Radix KV prefix hit >= 16, got {cache_n}");

    // 4. OpenAI streaming SSE
    let (st_sse, sse_body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "stream": true,
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Hi" }]
        })),
    )
    .await;
    assert_eq!(st_sse, StatusCode::OK);
    assert!(sse_body.contains("chat.completion.chunk") && sse_body.contains("data: [DONE]"));

    // 5. Anthropic /v1/messages streaming SSE
    let (st_ant, ant_body) = call(
        &app,
        "POST",
        "/v1/messages",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "stream": true,
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Say hello." }]
        })),
    )
    .await;
    assert_eq!(st_ant, StatusCode::OK);
    assert!(ant_body.contains("event: message_start"));
    assert!(ant_body.contains("event: content_block_delta"));
    assert!(ant_body.contains("event: message_stop"));

    // 5b. Security: Host-header DNS rebinding guard + optional API key check
    let bad_host = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("host", "evil.example.com:8090")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(bad_host).await.unwrap().status(), StatusCode::FORBIDDEN);

    let bad_origin = Request::builder()
        .method("POST")
        .uri("/v1/ocak/models/foo/sleep")
        .header("origin", "https://evil.example.com")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(bad_origin).await.unwrap().status(), StatusCode::FORBIDDEN);

    let auth_cfg = ServerConfig { api_key: Some("secret-test-key".into()), ..Default::default() };
    let auth_app = build_router(AppState::new(auth_cfg));
    let no_key = Request::builder().method("GET").uri("/v1/models").body(Body::empty()).unwrap();
    assert_eq!(auth_app.clone().oneshot(no_key).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    let ok_key = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer secret-test-key")
        .body(Body::empty())
        .unwrap();
    assert_eq!(auth_app.clone().oneshot(ok_key).await.unwrap().status(), StatusCode::OK);

    // 6. /v1/embeddings L2 norm == 1.0
    let (st_emb, emb_body) = call(
        &app,
        "POST",
        "/v1/embeddings",
        Some(json!({ "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M", "input": "paged attention" })),
    )
    .await;
    assert_eq!(st_emb, StatusCode::OK);
    let jemb: Value = serde_json::from_str(&emb_body).unwrap();
    let vec = jemb["data"][0]["embedding"].as_array().unwrap();
    let norm = vec.iter().map(|v| v.as_f64().unwrap().powi(2)).sum::<f64>().sqrt();
    println!("Embeddings: dim={}, L2 norm={norm:.6}", vec.len());
    assert!((norm - 1.0).abs() < 1e-4);

    // 6b. Continuous multi-slot batching: 3 concurrent /v1/chat/completions decoded together!
    let r_a = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "max_tokens": 16,
            "temperature": 0.0,
            "messages": [{ "role": "user", "content": "What is 2 + 2?" }]
        })),
    );
    let r_b = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "max_tokens": 16,
            "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Name one planet." }]
        })),
    );
    let r_c = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M",
            "max_tokens": 16,
            "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Say hello." }]
        })),
    );
    let ((sa, ba), (sb, bb), (sc, bc)) = tokio::join!(r_a, r_b, r_c);
    assert_eq!(sa, StatusCode::OK, "{ba}");
    assert_eq!(sb, StatusCode::OK, "{bb}");
    assert_eq!(sc, StatusCode::OK, "{bc}");
    let ja: Value = serde_json::from_str(&ba).unwrap();
    let jb: Value = serde_json::from_str(&bb).unwrap();
    let jc: Value = serde_json::from_str(&bc).unwrap();
    assert!(!ja["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty());
    assert!(!jb["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty());
    assert!(!jc["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty());

    // 7. WarmSleep idle teardown & RX 6700 XT power/VRAM release + automatic wake
    let vram_active = state.status_snapshot().await.gpu.map(|g| g.vram_used_bytes).unwrap_or(0);
    tokio::time::sleep(Duration::from_secs(4)).await;
    let snap_sleep = state.status_snapshot().await;
    let mstate = snap_sleep
        .models
        .iter()
        .find(|m| m.id == "tinyllama-1.1b-chat-v1.0.Q4_K_M")
        .unwrap()
        .residency;
    let vram_sleep = snap_sleep.gpu.map(|g| g.vram_used_bytes).unwrap_or(0);
    let pwr_sleep = snap_sleep.gpu.map(|g| g.power_watts).unwrap_or(0.0);
    println!(
        "WarmSleep: state={mstate:?}, VRAM {:.2} -> {:.2} GiB, Power={pwr_sleep:.1} W",
        vram_active as f64 / (1u64 << 30) as f64,
        vram_sleep as f64 / (1u64 << 30) as f64
    );
    assert_eq!(mstate, ModelResidencyState::WarmSleep);
    assert!(vram_sleep < vram_active, "expected VRAM to drop after WarmSleep");

    // Wake automatically on next request
    let t_w = Instant::now();
    let (st_w, bw) = call(
        &app,
        "POST",
        "/v1/completions",
        Some(json!({ "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M", "prompt": "Hello", "max_tokens": 4, "temperature": 0.0 })),
    )
    .await;
    assert_eq!(st_w, StatusCode::OK, "{bw}");
    println!("Auto-wake completed in {:.0} ms", t_w.elapsed().as_secs_f64() * 1000.0);
}

#[tokio::test]
async fn slow_sse_does_not_stall_batch_and_swap_queues_cleanly() {
    let Some(tiny) = find_test_tinyllama() else {
        return;
    };
    let mut cfg = ServerConfig::default();
    if let Some(dir) = tiny.parent() {
        if !cfg.models_dirs.iter().any(|d| d == dir) {
            cfg.models_dirs.push(dir.to_path_buf());
        }
    }
    let state = AppState::new(cfg);
    let entry_tiny = state.find_model("tinyllama-1.1b-chat-v1.0.Q4_K_M").unwrap().clone();
    state.ensure_native(&entry_tiny).await.unwrap();

    // 1. Slow SSE client (capacity 1, never read) + healthy batched client
    let (slow_tx, _slow_rx_never_read) = mpsc::channel::<String>(1);
    let (slow_done_tx, slow_done_rx) = tokio::sync::oneshot::channel();
    let (fast_done_tx, fast_done_rx) = tokio::sync::oneshot::channel();
    state.batcher.submit(
        Arc::clone(&state),
        batch::BatchJob {
            model_id: entry_tiny.id.clone(),
            greq: GenRequest {
                prompt: "Count to twenty: 1, 2, 3,".into(),
                parse_special: false,
                max_tokens: 24,
                sampler: SamplerParams { temperature: 0.0, ..Default::default() },
                stop: Vec::new(),
                json_mode: false,
                thinking: None,
            },
            token_tx: Some(slow_tx),
            done_tx: slow_done_tx,
        },
    );
    state.batcher.submit(
        Arc::clone(&state),
        batch::BatchJob {
            model_id: entry_tiny.id.clone(),
            greq: GenRequest {
                prompt: "The capital of France is".into(),
                parse_special: false,
                max_tokens: 8,
                sampler: SamplerParams { temperature: 0.0, ..Default::default() },
                stop: Vec::new(),
                json_mode: false,
                thinking: None,
            },
            token_tx: None,
            done_tx: fast_done_tx,
        },
    );
    let fast_res = tokio::time::timeout(Duration::from_secs(15), fast_done_rx)
        .await
        .expect("healthy slot stalled by slow SSE client")
        .unwrap()
        .unwrap();
    assert!(!fast_res.text.is_empty());
    let _ = slow_done_rx.await;

    // 2. Graceful model swap when queued job targets a different model
    if let Some(lfm) = state.find_model("LFM3-1.7B").cloned() {
        let (swap_done_tx, swap_done_rx) = tokio::sync::oneshot::channel();
        state.batcher.submit(
            Arc::clone(&state),
            batch::BatchJob {
                model_id: lfm.id.clone(),
                greq: GenRequest {
                    prompt: "Hello".into(),
                    parse_special: false,
                    max_tokens: 6,
                    sampler: SamplerParams { temperature: 0.0, ..Default::default() },
                    stop: Vec::new(),
                    json_mode: false,
                    thinking: None,
                },
                token_tx: None,
                done_tx: swap_done_tx,
            },
        );
        let swap_res = tokio::time::timeout(Duration::from_secs(30), swap_done_rx)
            .await
            .expect("model swap timed out")
            .unwrap()
            .expect("graceful model swap failed");
        assert!(!swap_res.text.is_empty());
    }
}
