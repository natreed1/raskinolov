# Private dashboard deploy (Railway-first)

This sets up a **private, personal dashboard** for project usage/data/documentation tracking.

## Why Railway over Vercel for this use case

- You wanted a site that is only for you.
- This dashboard stores incoming telemetry in **SQLite** and serves one private page.
- Railway keeps a long-running process + persistent disk path more naturally than Vercel serverless.

If you still want Vercel, keep it for a frontend and point it to this private API, but Railway is the fastest secure path for now.

## What was added

- Server: `scripts/private_dashboard_server.py`
  - `GET /` private HTML dashboard (HTTP Basic auth)
  - `GET /api/summary` private JSON
  - `GET /api/events?limit=100` private JSON
  - `GET /api/scoring` private JSON (Cursor/Codex vs open-source vs specialized scoring snapshot)
  - `POST /api/ingest` bearer-token ingest endpoint
  - `GET /healthz` public health probe
- Deployment config: `railway.json`
- Cursor hooks can forward events to remote ingest endpoint:
  - `.cursor/hooks/lab_hook_after_shell_autodoc.py`
  - `.cursor/hooks/lab_hook_stop_append.py`

## Railway deploy steps

1. Create a new Railway project from this repo.
2. Add environment variables:

```bash
FE_DASHBOARD_USER=your_username
FE_DASHBOARD_PASSWORD=your_long_random_password
FE_DASHBOARD_INGEST_TOKEN=your_long_random_ingest_token
FE_DASHBOARD_DB_PATH=/data/private_dashboard.sqlite3
```

3. Add a persistent volume and mount at `/data`.
4. Deploy (Railway uses `railway.json` start command automatically).
5. Open your service URL and log in via HTTP Basic auth.

## Connect your local Cursor workflow to remote ingest

Set these in your shell profile (`~/.zshrc`) on your machine:

```bash
export FE_LAB_REMOTE_INGEST_URL="https://<your-railway-domain>/api/ingest"
export FE_LAB_REMOTE_INGEST_TOKEN="<same-as-FE_DASHBOARD_INGEST_TOKEN>"
```

Your existing hooks will then:

- keep writing local ledgers (`lab_dashboard/*.jsonl`, `docs/SPECIALIZED_RUN_HISTORY.md`)
- also POST each captured event to your private dashboard.

## Verify end-to-end

1. Run a tracked command in Cursor terminal:

```bash
python -m unittest tests.test_documentation_rag -v
```

2. Visit dashboard URL and confirm:
   - `events_total` increased
   - recent row shows your command + exit code
   - **Scoring vs Cursor Work** tab renders (when `docs/generated/*.comparison.json` exists)
   - docs snapshot sections render

## Security notes

- Use strong random values for `FE_DASHBOARD_PASSWORD` and `FE_DASHBOARD_INGEST_TOKEN`.
- Treat ingest token like an API secret.
- Rotate both secrets periodically.
- If needed, add an IP allowlist at Railway/proxy layer.
