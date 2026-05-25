# OpenAI Codex OAuth Specification

This document describes the HTTP APIs and runtime behavior required to independently implement OpenAI Codex OAuth for ChatGPT Plus/Pro subscription access.

## Constants

```text
Provider ID:      openai-codex
Client ID:        app_EMoamEEZ73f0CkXaXp7hrann
Authorize URL:    https://auth.openai.com/oauth/authorize
Token URL:        https://auth.openai.com/oauth/token
Redirect URI:     http://localhost:1455/auth/callback
Scope:            openid profile email offline_access
Callback port:    1455
Default bind host: 127.0.0.1
```

The client ID is the public OAuth client identifier for the Codex-style native/CLI flow. It is not a secret. Use this exact value for compatibility with this flow unless OpenAI provides a separate registered client ID for your application. Treat it as configurable in your implementation because OpenAI can change or revoke public client registrations.

The callback server may bind to a custom host if desired, but the OAuth `redirect_uri` must remain:

```text
http://localhost:1455/auth/callback
```

## Login Flow

OpenAI Codex OAuth uses authorization-code OAuth with PKCE.

### 1. Generate PKCE and state

Generate a PKCE verifier:

```text
code_verifier = BASE64URL_NO_PAD(32 random bytes)
```

Generate the challenge:

```text
code_challenge = BASE64URL_NO_PAD(SHA256(code_verifier))
code_challenge_method = S256
```

Generate a CSRF state value:

```text
state = hex(16 random bytes)
```

### 2. Start local callback server

Start an HTTP server on port `1455`.

Expected callback:

```http
GET /auth/callback?code=<authorization_code>&state=<state>
```

Required behavior:

- If the path is not `/auth/callback`, return `404`.
- If `state` does not match the generated state, return `400`.
- If `code` is missing, return `400`.
- On success, return `200` and capture the authorization code.
- Close the callback server after the login attempt completes.

A simple HTML success/error page is sufficient for browser responses.

### 3. Build the authorization URL

Send the user to:

```http
GET https://auth.openai.com/oauth/authorize
```

Query parameters:

```text
response_type=code
client_id=app_EMoamEEZ73f0CkXaXp7hrann
redirect_uri=http://localhost:1455/auth/callback
scope=openid profile email offline_access
code_challenge=<code_challenge>
code_challenge_method=S256
state=<state>
id_token_add_organizations=true
codex_cli_simplified_flow=true
originator=pi
```

`originator` identifies the client. Existing behavior uses `pi`.

### 4. Manual fallback

If the callback server cannot be used, support manual input. The user may paste any of:

```text
<full redirect URL>
code=<code>&state=<state>
<code>#<state>
<code>
```

Parsing behavior:

- If input is a URL, read `code` and `state` from its query string.
- If input contains `#`, treat it as `code#state`.
- If input contains `code=`, parse it as URL-encoded parameters.
- Otherwise treat the full input as the authorization code.

If a pasted state is present, it must match the generated state.

## Authorization Code Exchange

Exchange the authorization code for tokens.

Request:

```http
POST https://auth.openai.com/oauth/token
Content-Type: application/x-www-form-urlencoded
```

Form body:

```text
grant_type=authorization_code
client_id=app_EMoamEEZ73f0CkXaXp7hrann
code=<authorization_code>
code_verifier=<code_verifier>
redirect_uri=http://localhost:1455/auth/callback
```

Expected success response:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "expires_in": 3600
}
```

All three fields are required. `expires_in` must be numeric.

Store expiry as milliseconds since Unix epoch:

```text
expires = current_time_ms + expires_in * 1000
```

On non-2xx response, read the response body and surface it as an authentication error.

## Refresh Flow

Refresh when:

```text
current_time_ms >= expires
```

Request:

```http
POST https://auth.openai.com/oauth/token
Content-Type: application/x-www-form-urlencoded
```

Form body:

```text
grant_type=refresh_token
refresh_token=<stored_refresh_token>
client_id=app_EMoamEEZ73f0CkXaXp7hrann
```

Expected success response:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "expires_in": 3600
}
```

The refresh response is expected to include a new refresh token. Replace the stored access token, refresh token, and expiry.

On failure, preserve the old credentials so the user can reauthenticate or retry later.

## Access Token Claims

The access token is a JWT. Decode the JWT payload and read:

```text
payload["https://api.openai.com/auth"].chatgpt_account_id
```

This account ID is required for Codex backend API calls.

A stored credential record should contain at least:

```json
{
  "access": "<jwt access token>",
  "refresh": "<refresh token>",
  "expires": 1710000000000,
  "accountId": "<chatgpt account id>"
}
```

## Using the Token with Codex APIs

The OAuth access token is used as a bearer token.

Default Codex backend base URL:

```text
https://chatgpt.com/backend-api
```

Responses endpoint resolution:

- If the base URL ends with `/codex/responses`, use it unchanged.
- Else if it ends with `/codex`, append `/responses`.
- Otherwise append `/codex/responses`.

Default endpoint:

```http
POST https://chatgpt.com/backend-api/codex/responses
```

### Required HTTP/SSE headers

```text
Authorization: Bearer <access_token>
chatgpt-account-id: <accountId>
originator: pi
User-Agent: <client user agent>
OpenAI-Beta: responses=experimental
Accept: text/event-stream
Content-Type: application/json
```

For session-affinity and prompt caching, when a session ID is available, also send:

```text
session_id: <session_id>
x-client-request-id: <session_id>
```

## Retry Behavior for Codex Responses

Retry transient failures up to 3 times.

Retryable HTTP statuses:

```text
429, 500, 502, 503, 504
```

Also retry network failures and response bodies matching transient conditions such as:

```text
rate limit
overloaded
service unavailable
upstream connect
connection refused
```

Default backoff:

```text
1000ms, 2000ms, 4000ms
```

If the response includes `retry-after-ms`, use that value in milliseconds.

Otherwise, if the response includes `retry-after`, parse it as either:

- seconds, or
- an HTTP date.

## WebSocket Transport

A WebSocket transport may be used for the same Codex Responses endpoint.

Convert the resolved HTTP URL scheme:

```text
https -> wss
http  -> ws
```

Required WebSocket headers:

```text
Authorization: Bearer <access_token>
chatgpt-account-id: <accountId>
originator: pi
User-Agent: <client user agent>
OpenAI-Beta: responses_websockets=2026-02-06
x-client-request-id: <request_or_session_id>
session_id: <request_or_session_id>
```

Do not send SSE-specific headers such as:

```text
Accept: text/event-stream
Content-Type: application/json
OpenAI-Beta: responses=experimental
```

If WebSocket setup fails before streaming starts, fall back to HTTP/SSE. If it fails after streaming has started, treat it as a stream failure.

## Implementation Checklist

1. Generate PKCE verifier and challenge.
2. Generate random OAuth state.
3. Start a local callback server on port `1455`.
4. Build and open or print the authorization URL.
5. Accept `/auth/callback`, validate state, and extract code.
6. Support pasted redirect URL or authorization code fallback.
7. Exchange the authorization code for tokens.
8. Decode the access-token JWT and extract `chatgpt_account_id`.
9. Persist access token, refresh token, expiry, and account ID.
10. Refresh credentials when expired.
11. Use `Authorization: Bearer <access_token>` and `chatgpt-account-id` for Codex backend requests.