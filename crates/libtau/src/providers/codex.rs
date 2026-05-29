use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, oneshot};

use crate::providers::{CodexProviderConfig, ModelMetadata, openai};

pub const PROVIDER_NAME: &str = "openai-codex";
pub const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const CODEX_SCOPE: &str = "openid profile email offline_access";

pub fn default_models() -> Vec<ModelMetadata> {
    openai::default_models()
}

pub async fn authenticate() -> Result<CodexProviderConfig, Box<dyn std::error::Error>> {
    let code_verifier = random_base64url(32);
    let code_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(code_verifier.as_bytes()));
    let oauth_state = random_hex(16);

    let (code_tx, code_rx) = oneshot::channel();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let app_state = CallbackState {
        expected_state: oauth_state.clone(),
        code_tx: Arc::new(Mutex::new(Some(code_tx))),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:1455").await?;
    let app = Router::new()
        .route("/auth/callback", get(oauth_callback))
        .with_state(app_state);
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
    });

    let auth_url = authorization_url(&code_challenge, &oauth_state);
    println!("Open this URL in your browser to authenticate with OpenAI Codex:\n{auth_url}\n");
    println!("Waiting up to 120 seconds for browser callback...");

    let code = match tokio::time::timeout(Duration::from_secs(120), code_rx).await {
        Ok(result) => {
            result.map_err(|_| "OAuth callback server stopped before receiving a code")?
        }
        Err(_) => {
            println!("No callback received. Paste the redirect URL or authorization code:");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            parse_manual_code(input.trim(), &oauth_state)?
        }
    };

    let _ = shutdown_tx.send(());
    server.await??;

    let token = exchange_code(&code, &code_verifier).await?;
    let account_id = account_id_from_access_token(&token.access_token)?;
    let expires = unix_timestamp_millis()? + token.expires_in * 1000;

    Ok(CodexProviderConfig {
        access: token.access_token,
        refresh: token.refresh_token,
        expires,
        account_id,
    })
}

#[derive(Clone)]
struct CallbackState {
    expected_state: String,
    code_tx: Arc<Mutex<Option<oneshot::Sender<String>>>>,
}

async fn oauth_callback(
    State(state): State<CallbackState>,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    match validate_callback(&query, &state.expected_state) {
        Ok(code) => {
            if let Some(sender) = state.code_tx.lock().await.take() {
                let _ = sender.send(code);
            }
            (
                StatusCode::OK,
                Html(
                    "<h1>Tau authentication complete</h1><p>You can close this tab.</p>"
                        .to_string(),
                ),
            )
        }
        Err(message) => (
            StatusCode::BAD_REQUEST,
            Html(format!(
                "<h1>Tau authentication failed</h1><p>{message}</p>"
            )),
        ),
    }
}

fn validate_callback(
    query: &HashMap<String, String>,
    expected_state: &str,
) -> Result<String, String> {
    if query.get("state").map(String::as_str) != Some(expected_state) {
        return Err("OAuth state did not match".to_string());
    }
    query
        .get("code")
        .filter(|code| !code.is_empty())
        .cloned()
        .ok_or_else(|| "OAuth code was missing".to_string())
}

fn authorization_url(code_challenge: &str, state: &str) -> String {
    let mut url = url::Url::parse(CODEX_AUTHORIZE_URL).expect("static URL parses");
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CODEX_CLIENT_ID)
        .append_pair("redirect_uri", CODEX_REDIRECT_URI)
        .append_pair("scope", CODEX_SCOPE)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", "pi");
    url.to_string()
}

fn parse_manual_code(
    input: &str,
    expected_state: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if input.is_empty() {
        return Err("no OAuth callback was received and no code was pasted".into());
    }

    let (code, pasted_state) = if input.starts_with("http://") || input.starts_with("https://") {
        let url = url::Url::parse(input)?;
        let params: HashMap<_, _> = url.query_pairs().into_owned().collect();
        (
            params
                .get("code")
                .cloned()
                .ok_or("pasted URL did not contain a code")?,
            params.get("state").cloned(),
        )
    } else if let Some((code, state)) = input.split_once('#') {
        (code.to_string(), Some(state.to_string()))
    } else if input.contains("code=") {
        let params: HashMap<_, _> = url::form_urlencoded::parse(input.as_bytes())
            .into_owned()
            .collect();
        (
            params
                .get("code")
                .cloned()
                .ok_or("pasted parameters did not contain a code")?,
            params.get("state").cloned(),
        )
    } else {
        (input.to_string(), None)
    };

    if pasted_state
        .as_deref()
        .is_some_and(|state| state != expected_state)
    {
        return Err("pasted OAuth state did not match".into());
    }
    Ok(code)
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: i64,
}

async fn exchange_code(
    code: &str,
    code_verifier: &str,
) -> Result<TokenResponse, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let response = client
        .post(CODEX_TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", CODEX_CLIENT_ID),
            ("code", code),
            ("code_verifier", code_verifier),
            ("redirect_uri", CODEX_REDIRECT_URI),
        ])
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(format!("OpenAI OAuth token exchange failed ({status}): {body}").into());
    }
    Ok(serde_json::from_str(&body)?)
}

fn account_id_from_access_token(access_token: &str) -> Result<String, Box<dyn std::error::Error>> {
    let payload = access_token
        .split('.')
        .nth(1)
        .ok_or("access token was not a JWT")?;
    let decoded = URL_SAFE_NO_PAD.decode(payload)?;
    let value: Value = serde_json::from_slice(&decoded)?;
    value
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| "access token did not contain chatgpt_account_id".into())
}

fn random_base64url(bytes: usize) -> String {
    let mut data = vec![0_u8; bytes];
    rand::rng().fill_bytes(&mut data);
    URL_SAFE_NO_PAD.encode(data)
}

fn random_hex(bytes: usize) -> String {
    let mut data = vec![0_u8; bytes];
    rand::rng().fill_bytes(&mut data);
    data.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unix_timestamp_millis() -> Result<i64, Box<dyn std::error::Error>> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
