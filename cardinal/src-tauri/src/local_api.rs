//! Local HTTP API that lets other apps on this Mac (launchers such as Tuna or
//! Raycast) search with Cardinal.
//!
//! Cardinal listens on 127.0.0.1 with a random port, and writes the port and a
//! fresh access token to `local-api.json` next to its database on every launch.
//! Only the current user can read that file, so web pages and other accounts
//! cannot use the API.
//!
//! Every request needs `Authorization: Bearer <token>`:
//! - `GET /v1/status`
//! - `GET /v1/search?q=<query>&limit=<1-200>&caseSensitive=<true|false>`
//! - `POST /v1/show?q=<query>` brings up the window with the query filled in.

use crate::{
    background::indexed_files,
    external_search::ExternalSearchJob,
    lifecycle::{AppLifecycleState, load_app_state},
    window_controls::activate_main_window_impl,
};
use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Sender, bounded};
use fswalk::NodeFileType;
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_foundation::{NSActivityOptions, NSProcessInfo, ns_string};
use search_cache::SearchResultNode;
use search_cancel::CancellationToken;
use serde::Serialize;
use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::Arc,
    time::Duration,
};
use tauri::{AppHandle, Emitter};
use tiny_http::{Header, Method, Request, Response, Server};
use tracing::{error, info, warn};

pub const CONNECTION_FILE_NAME: &str = "local-api.json";
const CONNECTION_FILE_VERSION: u32 = 1;
const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 200;
const SEARCH_TIMEOUT: Duration = Duration::from_secs(10);

struct ApiContext {
    token: String,
    app_handle: AppHandle,
    external_search_tx: Sender<ExternalSearchJob>,
}

/// Starts the API on a background thread and publishes its connection file.
pub fn start(
    app_handle: AppHandle,
    external_search_tx: Sender<ExternalSearchJob>,
    connection_file: &Path,
) -> Result<()> {
    let server =
        Server::http("127.0.0.1:0").map_err(|e| anyhow!("Failed to bind local API: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("Local API is not bound to an IP address")?
        .port();
    let token = random_token()?;
    write_connection_file(connection_file, port, &token)?;
    info!("Local API listening on 127.0.0.1:{port}");

    let context = Arc::new(ApiContext {
        token,
        app_handle,
        external_search_tx,
    });
    std::thread::Builder::new()
        .name("local-api".into())
        .spawn(move || {
            for request in server.incoming_requests() {
                let context = Arc::clone(&context);
                // One thread per request, so a newer search can cancel an older one.
                std::thread::spawn(move || handle_request(&context, request));
            }
        })?;
    Ok(())
}

pub fn remove_connection_file(connection_file: &Path) {
    if let Err(err) = fs::remove_file(connection_file)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        warn!(?err, "Failed to remove local API connection file");
    }
}

#[derive(Serialize)]
struct ConnectionInfo<'a> {
    version: u32,
    port: u16,
    token: &'a str,
    pid: u32,
}

fn write_connection_file(path: &Path, port: u16, token: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    // Write to a private temp file first so readers never see a partial file.
    let temp_path = path.with_extension("json.tmp");
    let _ = fs::remove_file(&temp_path);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp_path)
        .with_context(|| format!("Failed to create {temp_path:?}"))?;
    let info = ConnectionInfo {
        version: CONNECTION_FILE_VERSION,
        port,
        token,
        pid: std::process::id(),
    };
    file.write_all(&serde_json::to_vec_pretty(&info)?)?;
    file.sync_all()?;
    fs::rename(&temp_path, path).with_context(|| format!("Failed to publish {path:?}"))?;
    Ok(())
}

fn random_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

type ApiResponse = Response<Cursor<Vec<u8>>>;

fn handle_request(context: &ApiContext, request: Request) {
    let (path, params) = split_url(request.url());
    let response = if !is_authorized(&request, &context.token) {
        json_error(401, "Missing or wrong access token")
    } else {
        match (request.method(), path.as_str()) {
            (Method::Get, "/v1/status") => status_response(),
            (Method::Get, "/v1/search") => search_response(context, &params),
            (Method::Post, "/v1/show") => show_response(context, &params),
            (_, "/v1/status" | "/v1/search" | "/v1/show") => json_error(405, "Method not allowed"),
            _ => json_error(404, "Not found"),
        }
    };
    if let Err(err) = request.respond(response) {
        warn!(?err, "Failed to answer local API request");
    }
}

fn split_url(url: &str) -> (String, HashMap<String, String>) {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let params = form_urlencoded::parse(query.as_bytes())
        .into_owned()
        .collect();
    (path.to_string(), params)
}

fn is_authorized(request: &Request, token: &str) -> bool {
    let expected = format!("Bearer {token}");
    request
        .headers()
        .iter()
        .any(|header| header.field.equiv("Authorization") && header.value.as_str() == expected)
}

fn state_name(state: AppLifecycleState) -> &'static str {
    match state {
        AppLifecycleState::Initializing => "indexing",
        AppLifecycleState::Updating => "updating",
        AppLifecycleState::Ready => "ready",
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusBody {
    state: &'static str,
    indexed_files: usize,
    version: &'static str,
}

fn status_response() -> ApiResponse {
    json_ok(&StatusBody {
        state: state_name(load_app_state()),
        indexed_files: indexed_files(),
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchBody {
    state: &'static str,
    indexed_files: usize,
    /// Number of matches before the list was cut to `limit`.
    total: usize,
    /// A newer search replaced this one; the caller should ignore it.
    cancelled: bool,
    results: Vec<ResultItem>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResultItem {
    name: String,
    path: String,
    /// `file`, `folder`, `symlink` or `unknown`.
    kind: &'static str,
    /// Size in bytes, only for files.
    size: Option<i64>,
    /// Unix timestamps in seconds.
    modified: Option<u32>,
    created: Option<u32>,
}

impl From<&SearchResultNode> for ResultItem {
    fn from(node: &SearchResultNode) -> Self {
        let path = node.path.to_string_lossy().into_owned();
        let name = node
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        let kind = match node.metadata.file_type_hint() {
            NodeFileType::File => "file",
            NodeFileType::Dir => "folder",
            NodeFileType::Symlink => "symlink",
            NodeFileType::Unknown => "unknown",
        };
        let metadata = node.metadata.as_ref();
        Self {
            name,
            path,
            kind,
            size: metadata
                .as_ref()
                .filter(|_| kind == "file")
                .map(|metadata| metadata.size()),
            modified: metadata
                .as_ref()
                .and_then(|metadata| metadata.mtime())
                .map(|time| time.get()),
            created: metadata
                .as_ref()
                .and_then(|metadata| metadata.ctime())
                .map(|time| time.get()),
        }
    }
}

fn search_response(context: &ApiContext, params: &HashMap<String, String>) -> ApiResponse {
    let query = params.get("q").cloned().unwrap_or_default();
    let limit = match params.get("limit").map(|value| value.parse::<usize>()) {
        None => DEFAULT_LIMIT,
        Some(Ok(limit)) => limit.clamp(1, MAX_LIMIT),
        Some(Err(_)) => return json_error(400, "limit must be a number"),
    };
    let case_sensitive = match params.get("caseSensitive").map(String::as_str) {
        None | Some("false") | Some("0") => false,
        Some("true") | Some("1") => true,
        Some(_) => return json_error(400, "caseSensitive must be true or false"),
    };

    let state = load_app_state();
    let body = |total, cancelled, results| SearchBody {
        state: state_name(state),
        indexed_files: indexed_files(),
        total,
        cancelled,
        results,
    };
    // The first scan holds the index, so searches would wait until it finishes.
    if state == AppLifecycleState::Initializing {
        return json_ok(&body(0, false, Vec::new()));
    }

    let _activity = UserInitiatedActivity::begin();
    let (result_tx, result_rx) = bounded(1);
    let job = ExternalSearchJob {
        query,
        case_insensitive: !case_sensitive,
        limit,
        cancellation_token: CancellationToken::new_external_search(),
        result_tx,
    };
    if context.external_search_tx.send(job).is_err() {
        return json_error(503, "Cardinal is not ready yet");
    }
    match result_rx.recv_timeout(SEARCH_TIMEOUT) {
        Ok(Ok(Some(results))) => json_ok(&body(
            results.total,
            false,
            results.nodes.iter().map(ResultItem::from).collect(),
        )),
        Ok(Ok(None)) => json_ok(&body(0, true, Vec::new())),
        Ok(Err(err)) => json_error(400, &format!("Invalid query: {err}")),
        Err(_) => json_error(503, "Cardinal is busy, try again"),
    }
}

/// Keeps macOS from throttling Cardinal with App Nap while it answers a request.
/// Launchers call while Cardinal sits in the background, where a throttled search
/// runs about three times slower.
struct UserInitiatedActivity(Retained<ProtocolObject<dyn NSObjectProtocol>>);

impl UserInitiatedActivity {
    fn begin() -> Self {
        Self(
            NSProcessInfo::processInfo().beginActivityWithOptions_reason(
                NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
                ns_string!("Answering a search from another app"),
            ),
        )
    }
}

impl Drop for UserInitiatedActivity {
    fn drop(&mut self) {
        // SAFETY: the activity came from `beginActivityWithOptions_reason` and ends once.
        unsafe { NSProcessInfo::processInfo().endActivity(&self.0) };
    }
}

fn show_response(context: &ApiContext, params: &HashMap<String, String>) -> ApiResponse {
    let query = params.get("q").cloned().unwrap_or_default();
    activate_main_window_impl(&context.app_handle);
    if let Err(err) = context.app_handle.emit("external_search", query) {
        error!(?err, "Failed to forward external search to the window");
        return json_error(500, "Failed to open Cardinal");
    }
    json_ok(&serde_json::json!({ "ok": true }))
}

fn json_ok<T: Serialize>(body: &T) -> ApiResponse {
    json_response(200, serde_json::to_vec(body).unwrap_or_default())
}

fn json_error(status: u16, message: &str) -> ApiResponse {
    json_response(
        status,
        serde_json::to_vec(&serde_json::json!({ "error": message })).unwrap_or_default(),
    )
}

fn json_response(status: u16, body: Vec<u8>) -> ApiResponse {
    let content_type = Header::from_bytes("Content-Type", "application/json; charset=utf-8")
        .expect("static header is valid");
    Response::from_data(body)
        .with_status_code(status)
        .with_header(content_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_url_decodes_query_parameters() {
        let (path, params) = split_url("/v1/search?q=annual%20report+ext%3Apdf&limit=5");
        assert_eq!(path, "/v1/search");
        assert_eq!(params["q"], "annual report ext:pdf");
        assert_eq!(params["limit"], "5");

        let (path, params) = split_url("/v1/status");
        assert_eq!(path, "/v1/status");
        assert!(params.is_empty());
    }

    #[test]
    fn connection_file_is_private_and_complete() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CONNECTION_FILE_NAME);
        write_connection_file(&path, 4321, "secret").unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let info: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(info["port"], 4321);
        assert_eq!(info["token"], "secret");
        assert_eq!(info["version"], CONNECTION_FILE_VERSION);

        // Rewriting on the next launch replaces the old file.
        write_connection_file(&path, 1234, "other").unwrap();
        let info: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(info["port"], 1234);

        remove_connection_file(&path);
        assert!(!path.exists());
    }

    #[test]
    fn random_tokens_are_long_and_unique() {
        let first = random_token().unwrap();
        let second = random_token().unwrap();
        assert_eq!(first.len(), 64);
        assert_ne!(first, second);
    }
}
