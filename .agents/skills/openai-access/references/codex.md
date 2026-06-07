# OpenAI Codex OAuth and API Specification

This document describes the HTTP APIs and runtime behavior needed to implement OpenAI Codex access for ChatGPT subscription accounts.

Codex authentication can be implemented with either of these OAuth patterns:

1. Authorization code + PKCE, using a browser redirect to a local callback server.
2. Device authorization grant, using a user code, verification URL, and polling.

Both flows produce the same credential shape: an access token, refresh token, expiry, and ChatGPT account ID. The resulting access token is used the same way for Codex backend requests.

## Shared Constants

```text
Provider ID:      openai-codex
Client ID:        app_EMoamEEZ73f0CkXaXp7hrann
Authorize URL:    https://auth.openai.com/oauth/authorize
Device Code URL:  https://auth.openai.com/oauth/device/code
Token URL:        https://auth.openai.com/oauth/token
Redirect URI:     http://localhost:1455/auth/callback
Scope:            openid profile email offline_access
Callback port:    1455
Default bind host: 127.0.0.1
Default API base: https://chatgpt.com/backend-api
```

The client ID is a public OAuth client identifier for the Codex-style native/CLI flow. It is not a secret. Treat it as configurable because OpenAI can change or revoke public client registrations.

## Flow 1: Authorization Code + PKCE Grant

Use this flow when a browser can redirect back to a local callback server.

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

Keep the verifier and state until the authorization code has been exchanged.

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
originator=<client name>
```

`originator` identifies the client.

Expected behavior:

- User signs in through ChatGPT in the browser.
- The selected account must have Codex entitlement through ChatGPT Plus/Pro or another eligible plan.
- On success, the browser redirects to the local callback with `code` and `state`.

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

### 5. Exchange authorization code for tokens

Request:

```http
POST https://auth.openai.com/oauth/token
Content-Type: application/x-www-form-urlencoded
Accept: application/json
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

## Flow 2: Device Authorization Grant

Use this flow when a local browser callback is undesirable or unavailable, such as SSH/headless environments. The client requests a device code, displays a user code and verification URL, then polls the token endpoint until the user completes login in a browser.

### 1. Request a device code

Request:

```http
POST https://auth.openai.com/oauth/device/code
Content-Type: application/x-www-form-urlencoded
Accept: application/json
```

Form body:

```text
client_id=app_EMoamEEZ73f0CkXaXp7hrann
scope=openid profile email offline_access
id_token_add_organizations=true
codex_cli_simplified_flow=true
originator=<client name>
```

Expected success response:

```json
{
  "device_code": "...",
  "user_code": "ABCD-EFGH",
  "verification_uri": "https://auth.openai.com/activate",
  "verification_uri_complete": "https://auth.openai.com/activate?user_code=ABCD-EFGH",
  "expires_in": 900,
  "interval": 5
}
```

Required fields:

- `device_code`: opaque code used by the client while polling.
- `user_code`: code shown to the user.
- `verification_uri`: URL where the user enters the code.
- `expires_in`: lifetime of the device code in seconds.

Optional fields:

- `verification_uri_complete`: URL that may pre-fill the user code.
- `interval`: polling interval in seconds. If omitted, use 5 seconds.

### 2. Show the user code

Display:

```text
Open <verification_uri> and enter code <user_code>
```

If `verification_uri_complete` is present, open or display it as the preferred URL while still showing the `user_code` for manual entry.

The user signs in through ChatGPT in the browser. The selected account must have Codex entitlement through ChatGPT Plus/Pro or another eligible plan.

### 3. Poll for tokens

Poll only after waiting the server-provided interval. Do not poll immediately unless the server explicitly instructs a zero interval.

Request:

```http
POST https://auth.openai.com/oauth/token
Content-Type: application/x-www-form-urlencoded
Accept: application/json
```

Form body:

```text
grant_type=urn:ietf:params:oauth:grant-type:device_code
client_id=app_EMoamEEZ73f0CkXaXp7hrann
device_code=<device_code>
```

Successful response:

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

### 4. Polling error behavior

While login is pending, the token endpoint returns JSON errors rather than tokens.

Pending response:

```json
{
  "error": "authorization_pending",
  "error_description": "..."
}
```

Behavior: keep polling at the current interval.

Slow-down response:

```json
{
  "error": "slow_down",
  "error_description": "..."
}
```

Behavior: increase the polling interval by 5 seconds for this and all later polls.

Expired response:

```json
{
  "error": "expired_token",
  "error_description": "..."
}
```

Behavior: stop polling and restart the device-code flow if the user wants to retry.

Denied response:

```json
{
  "error": "access_denied",
  "error_description": "..."
}
```

Behavior: stop polling and report that login was denied or cancelled.

Other error response:

```json
{
  "error": "<error>",
  "error_description": "<description>"
}
```

Behavior: stop polling and surface the error.

Timeout behavior:

- Stop polling when the device-code lifetime expires.
- If one or more `slow_down` responses occurred before timeout, include a clock-drift hint for WSL/VM environments.
- Support cancellation via the caller/UI and report cancellation distinctly from timeout.

## Refresh Flow

Both login flows use the same refresh flow.

Refresh before dispatching a Codex API request when:

```text
current_time_ms >= expires
```

Request:

```http
POST https://auth.openai.com/oauth/token
Content-Type: application/x-www-form-urlencoded
Accept: application/json
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

Behavior:

- Resolve credentials per request, not only at process startup.
- Refresh is based on local stored expiry before request dispatch.
- On refresh failure, preserve the old credentials so the user can reauthenticate or retry later.

Common refresh failure:

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json
```

Example body:

```json
{
  "error": {
    "message": "Could not validate your token. Please try signing in again.",
    "type": "invalid_request_error"
  }
}
```

Recommended surfaced message:

```text
OpenAI Codex token refresh failed (401): {response body or status text}
```

After a failed refresh, prompt the user to log in again.

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

## Codex Responses HTTP API

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
originator: <client name>
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

### Request body

The request body is OpenAI Responses-style JSON.

Typical fields:

```json
{
  "model": "gpt-5.1-codex",
  "store": false,
  "stream": true,
  "instructions": "You are a helpful assistant.",
  "input": [],
  "tools": [],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "text": {
    "verbosity": "low"
  },
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  },
  "include": ["reasoning.encrypted_content"],
  "prompt_cache_key": "<session id or cache key>"
}
```

Notes:

- `store` should be `false` for ChatGPT Codex Responses.
- `stream` should be `true` for SSE streaming.
- `input` follows the OpenAI Responses input item format.
- `tools` follows the OpenAI Responses tool format.
- `reasoning` is optional and depends on model support.
- `prompt_cache_key` can be derived from a stable session identifier.

### Streaming response behavior

The HTTP response is an SSE stream with `data:` JSON events. Treat `[DONE]` as a terminal sentinel if present.

Important events:

- Standard OpenAI Responses stream events should be handled normally.
- `response.done`, `response.completed`, and `response.incomplete` are terminal response events.
- `response.failed` contains a response error and should be surfaced as a provider error.
- `error` events should be surfaced as provider errors.

## WebSocket Transport

A WebSocket transport may be used for the same Codex Responses endpoint.

Convert the resolved HTTP URL scheme:

```text
https -> wss
http  -> ws
```

Default WebSocket endpoint:

```text
wss://chatgpt.com/backend-api/codex/responses
```

Required WebSocket headers:

```text
Authorization: Bearer <access_token>
chatgpt-account-id: <accountId>
originator: <client name>
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

Initial message:

```json
{
  "type": "response.create",
  "model": "gpt-5.1-codex",
  "store": false,
  "stream": true,
  "input": []
}
```

Behavior:

- If WebSocket setup fails before streaming starts, fall back to HTTP/SSE.
- If WebSocket fails after streaming has started, treat it as a stream failure.
- WebSocket sessions may be reused for a short period when a stable session ID is available.
- If cached continuation is implemented, send `previous_response_id` and only the input delta when the server-side connection context is still valid.
- Idle cached WebSocket sessions should expire after a short inactivity window, such as five minutes.

## Error Behavior

### Authentication errors

Codex API authentication errors commonly return `401` with an OpenAI-style error object.

Example:

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json
```

```json
{
  "error": {
    "message": "Could not validate your token. Please try signing in again.",
    "type": "invalid_request_error"
  }
}
```

Recommended behavior:

- If local credentials are expired before dispatch, refresh before making the request.
- If the API returns `401`, surface the auth failure and prompt re-login.
- Do not assume every `401` is recoverable by refreshing.
- If implementing refresh-on-401, do at most one refresh-and-retry attempt and avoid infinite loops.

### Usage and rate-limit errors

Usage/rate-limit errors commonly return `429` with an OpenAI-style error object. The error may include fields such as:

```json
{
  "error": {
    "code": "usage_limit_reached",
    "message": "...",
    "plan_type": "PLUS",
    "resets_at": 1710000000
  }
}
```

Recommended behavior:

- Surface usage-limit errors distinctly from auth errors.
- If `resets_at` is present, convert it to an approximate retry time.
- Do not prompt re-login for usage-limit errors.

### Transient errors and retries

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

## Implementation Checklist

1. Choose authorization code + PKCE or device authorization grant.
2. Complete the selected login flow and obtain tokens.
3. Decode the access-token JWT and extract `chatgpt_account_id`.
4. Persist access token, refresh token, expiry, and account ID.
5. Resolve credentials on every Codex request.
6. Refresh credentials before request dispatch when expired.
7. Use `Authorization: Bearer <access_token>` and `chatgpt-account-id` for Codex backend requests.
8. Surface refresh failures and request `401` responses as re-login-required authentication failures.
9. Implement SSE streaming and optionally WebSocket with SSE fallback.
