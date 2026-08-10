# Scheduling examples

Two paths:

| Example | Purpose |
| --- | --- |
| [`schedule_agent.py`](schedule_agent.py) + [`agent.yaml`](agent.yaml) | **Deployable** HTTP agent (`registry()` → Nomad) |
| [`scheduled_report_example.py`](scheduled_report_example.py) | Local asyncio demo of `execute_job` (not deploy) |

## Deployable agent (`schedule_agent.py`)

`ReportScheduler` class — same idea as the local demo, but the job lives in
SQLite/Postgres and the process is an HTTP assistant.

**Where is `interval_seconds`?** On the `Job` object, written in
`ReportScheduler.default_job()` (default `3600`, override
`SCHEDULE_INTERVAL_SECONDS` in `agent.yaml`). On startup the agent **seeds**
that job into `CAT_AGENT_SCHEDULER_DSN`. Chat tools (`create_schedule`) add
more jobs by converting `every_hours * 3600` → `interval_seconds` internally.

This process does **not** run the tick loop — pair with
`cat-agent schedule run-due` (K8s CronJob) or APScheduler in dev.

### Setup

```bash
pip install 'cat-agent[serve,platform,scheduler]'
```

Secrets in repo **`.env`** (see [`examples/multi_agent/README.md`](../multi_agent/README.md)):

```bash
OLLAMA_API_BASE=https://ollama.com/v1
OLLAMA_API_KEY=...
```

Models / scheduler DSN in **`agent.yaml`** (`model.alias`, `env.CAT_AGENT_SCHEDULER_DSN`).

### Local serve

```bash
python examples/scheduling/schedule_agent.py
curl -sS http://127.0.0.1:8080/agents/report-scheduler/run \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Create a schedule for user alice: topic AI news, every 24 hours, webhook http://127.0.0.1:9999/hook"}]}'
```

### Nomad deploy

```bash
cd ../cat-agent-stack && cat-agent stack bootstrap   # once
cd ../cat-agent
cat-agent deploy --dir examples/scheduling
```

Then hit the printed URL (or Traefik host) with the same JSON body as above.

Production: point `CAT_AGENT_SCHEDULER_DSN` at Postgres (see main README
“Scheduled reports”) so all replicas share job state.

---

## Local loop demo (`scheduled_report_example.py`)

Minimal **non-deploy** script: SQLite store, tick every 1 minute, print report,
exit after 5 minutes.

```bash
pip install 'cat-agent[scheduler]'
python examples/scheduling/scheduled_report_example.py
```

See repo root README — **report jobs** vs **Nomad deploy** are different planes.
