//! Loopback Host / Origin checks and API-key authentication middleware.
use axum::extract::{Request as AxRequest, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::sync::Arc;
use crate::*;

/// `host[:port]` naming the loopback interface, and nothing else (no userinfo, path or junk).
pub(crate) fn is_loopback_host(host: &str) -> bool {
    let h = host.trim();
    let port_ok = |p: &str| p.is_empty() || p.strip_prefix(':').is_some_and(|d| !d.is_empty() && d.len() <= 5 && d.bytes().all(|b| b.is_ascii_digit()));
    if let Some(rest) = h.strip_prefix('[') {
        return rest.strip_prefix("::1]").is_some_and(port_ok);
    }
    let (bare, port) = h.find(':').map_or((h, ""), |i| (&h[..i], &h[i..]));
    matches!(bare.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1") && port_ok(port)
}

/// Percent-decodes a query value (`+` is a space, as in form encoding).
pub(crate) fn query_decode(v: &str) -> String {
    let b = v.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        match b[i] {
            b'%' if i + 2 < b.len() => {
                if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                    continue;
                }
                out.push(b'%');
            }
            b'+' => out.push(b' '),
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub(crate) fn is_loopback_origin(origin: &str) -> bool {
    let s = origin.trim();
    let rest = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or("");
    if rest.is_empty() {
        return false;
    }
    let host_port = rest.split('/').next().unwrap_or("");
    is_loopback_host(host_port)
}

pub(crate) fn ct_eq(a: &str, b: &str) -> bool {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    if ab.len() != bb.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in ab.iter().zip(bb.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

pub(crate) async fn security_middleware(
    State(st): State<Arc<AppState>>,
    req: AxRequest,
    next: Next,
) -> Response {
    if !st.cfg.allow_remote_host {
        // A header that is present but not valid UTF-8 fails the check (fail closed).
        if let Some(host) = req.headers().get("host") {
            if !host.to_str().is_ok_and(is_loopback_host) {
                return (StatusCode::FORBIDDEN, Json(json!({ "error": "invalid Host header" }))).into_response();
            }
        }
        if let Some(origin) = req.headers().get("origin") {
            if !origin.to_str().is_ok_and(is_loopback_origin) {
                return (StatusCode::FORBIDDEN, Json(json!({ "error": "cross-origin request blocked" }))).into_response();
            }
        }
    }
    if let Some(ref expected) = st.cfg.api_key {
        let path = req.uri().path();
        if path.starts_with("/v1/") {
            let bearer = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            let x_api = req.headers().get("x-api-key").and_then(|v| v.to_str().ok());
            // Query param api_key is accepted ONLY on SSE /v1/ocak/events (EventSource cannot set headers)
            let query_key = if path == "/v1/ocak/events" {
                req.uri().query().and_then(|q| {
                    q.split('&')
                        .find_map(|pair| pair.strip_prefix("api_key="))
                        .map(query_decode)
                })
            } else {
                None
            };
            let ok = bearer.is_some_and(|k| ct_eq(k, expected))
                || x_api.is_some_and(|k| ct_eq(k, expected))
                || query_key.is_some_and(|k| ct_eq(&k, expected));
            if !ok {
                return (StatusCode::UNAUTHORIZED, Json(json!({ "error": "unauthorized" }))).into_response();
            }
        }
    }
    next.run(req).await
}
