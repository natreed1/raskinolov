# Private telemetry dashboard deploy (Railway-first)

This sets up a **private, project-wide telemetry dashboard** for usage, data, and documentation tracking. It stores incoming telemetry in **SQLite** and serves a private page. This is for **project members only**. If you still want Vercel, keep it for a frontend and point it to this private API, but Railway is the fastest secure path for now.

## Purpose

The dashboard is for **project members only**. It stores incoming telemetry in **SQLite** and serves a private page. If you still want Vercel, keep it for a frontend and point it to this private API, but Railway is the fastest secure path for now.

## Architecture

- **Server**: `scripts/private_dashboard_server.py` runs on Railway.
  - `GET /` private HTML dashboard (HTTP Basic auth)
  - `GET /api/summary` private JSON
  - `GET /api/events?limit=100` private JSON
  - `POST /api/ingest` bearer-token ingest endpoint
  - `GET /healthz` public health probe
- **Cursor hooks** can forward events to remote ingest endpoint:
  - `.cursor/hooks/lab_hook_after_shell_autodoc.py`
  - `.cursor/hooks/lab_hook_stop_append.py`
- **Local artifacts** are written to `docs/run_history.jsonl` and `docs/specialized_run_history.md`.
- **Remote ingest** is to `/api/ingest` with a bearer token.

## Deploy (Railway)

1. **Create a new Railway project** from this repo.
2. **Add environment variables**:

```bash
FE_TELEMETRY_USER=your_username
FE_TELEMETRY_PASSWORD=your_long_random_password
FE_TELEMETRY_INGEST_TOKEN=your_long_random_ingest_token
FE_TELEMETRY_DB_PATH=/data/telemetry.sqlite3
```

3. **Add a persistent volume and mount at `/data`**.
4. **Deploy** (Railway uses `railway.json` start command automatically).
5. **Open your service URL** and log in via HTTP Basic auth.

## Local Hook Wiring

**Cursor hooks** can forward events to remote ingest endpoint:

- `.cursor/hooks/lab_hook_after_shell_autodoc.py` (run after shell commands)
- `.cursor/hooks/lab_hook_stop_append.py` (run when you stop a run)

**Local artifacts** are written to `docs/run_history.jsonl` and `docs/specialized_run_history.md`. **Remote ingest** is to `/api/ingest` with a bearer token.

## Security

- Use strong random values for `FE_TELEMETRY_PASSWORD` and `FE_TELEMETRY_INGEST_TOKEN`.
- Treat ingest token like an API secret.
- Rotate both secrets periodically.
- If needed, add an IP allowlist at Railway/proxy layer.

## Operations Checklist

- **Verify end-to-end**:
  1. Run a tracked command in Cursor terminal:

  ```bash
  python -m unittest tests.test_documentation_rag -v
  ```

  2. Visit dashboard URL and confirm:
     - `events_total` increased
     - recent row shows your command + exit code
     - docs snapshot sections render
- **Rotate secrets** periodically.
- **Add an IP allowlist** at Railway/proxy layer
