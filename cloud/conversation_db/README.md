# Conversation DB — always-on cloud home for booked debates

A tiny FastAPI + SQLite website that stores Model Chat debates pushed from the
local arena. The chat (and the open-source MLX models) stay on your computer;
this service is just the always-on database + viewer.

## Pages

- `/` — searchable table of conversations (topic, speakers, winner, turns, date)
- `/c/{id}` — full transcript + judge verdicts
- `/api/conversations` (GET list / POST ingest), `/api/conversations/{id}`, `/health`

## Deploy to Railway

```bash
railway login
cd cloud/conversation_db
railway init            # create a project
railway volume add --mount-path /data     # persist SQLite across deploys
railway variables --set FE_CONVERSATION_DB_TOKEN=<pick-a-secret>
railway up              # build + deploy (uses the Dockerfile)
railway domain          # get the public URL
```

## Point the local arena at it

```bash
export FE_CONVERSATION_DB_URL="https://<your-app>.up.railway.app"
export FE_CONVERSATION_DB_TOKEN="<same-secret>"
python scripts/game_task_arena.py ui --port 7868
```

With those set, every judged debate is pushed to the cloud automatically
(in addition to the local `benchmarks/results/model_chat_debates/debates.jsonl`).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 8090
```
