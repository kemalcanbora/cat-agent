"""Simple scheduler demo: run every 1 minute, exit after 5 minutes.

No email. Each tick seeds one source and prints a short LLM report, then
shuts down cleanly when the wall-clock budget is up.

Config from repo ``.env`` (same pattern as ``examples/multi_agent/team_example.py``)::

    CAT_AGENT_OFFLINE=0
    OLLAMA_API_KEY=...
    LLM_MODEL=minimax-m2.7:cloud
    OLLAMA_API_BASE=https://ollama.com/v1

    python examples/scheduling/scheduled_report_example.py

Requires ``pip install 'cat-agent[scheduler]'``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Must load BEFORE importing cat_agent (offline guards read CAT_AGENT_OFFLINE at import).
load_dotenv(REPO_ROOT / '.env', override=True)

from cat_agent.scheduling.graph import build_report_markdown
from cat_agent.scheduling.models import Job
from cat_agent.scheduling.runner import execute_job
from cat_agent.scheduling.store import JobStore

INTERVAL_SECONDS = 60
RUN_FOR_SECONDS = 5 * 60
USER_ID = 'alice'
JOB_ID = 'report:alice:heartbeat'
TOPIC = 'heartbeat demo'


def build_llm_cfg() -> Dict:
    """Ollama Cloud / local via the existing OpenAI-compatible ``oai`` backend."""
    api_key = (
        os.getenv('OLLAMA_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or 'EMPTY'
    )
    model = os.getenv('LLM_MODEL', 'minimax-m2.7:cloud')
    base_url = (os.getenv('OLLAMA_API_BASE') or 'https://ollama.com/v1').rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = base_url + '/v1'
    return {
        'model': model,
        'model_type': 'oai',
        'model_server': base_url,
        'api_key': api_key,
        'generate_cfg': {
            'temperature': 0.2,
            'top_p': 0.8,
            'max_tokens': 256,
        },
    }


def ensure_job(store: JobStore, *, now: float) -> Job:
    job = Job(
        id=JOB_ID,
        user_id=USER_ID,
        kind='report',
        topic=TOPIC,
        interval_seconds=INTERVAL_SECONDS,
        channel='webhook',
        target='http://127.0.0.1/unused',  # unused — we print instead of delivering
        enabled=True,
        next_run_at=now - 1,
        created_at=now,
        updated_at=now,
    )
    store.upsert_job(job)
    return job


async def run_once(store: JobStore, llm_cfg: Dict, *, tick: int) -> None:
    now = time.time()
    stamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
    store.save_source(
        user_id=USER_ID,
        url=f'https://example.local/tick/{tick}',
        title=f'Tick #{tick} at {stamp}',
        summary=f'Scheduler heartbeat tick {tick} ({stamp} UTC).',
        tags='demo,heartbeat',
        job_id=JOB_ID,
        collected_at=now,
    )
    # Make the job due for this tick.
    job = store.get_job(JOB_ID)
    assert job is not None
    job.next_run_at = now - 1
    store.upsert_job(job)

    async def report_fn(job, sources, store_):
        return await build_report_markdown(job, sources, llm=llm_cfg)

    async def deliver_fn(job, body, sources, store_):
        print()
        print(f'--- tick {tick} report ({len(sources)} source(s)) ---')
        print(body)
        print('---')

    run = await execute_job(
        JOB_ID,
        store=store,
        owner='example-host',
        report_fn=report_fn,
        deliver_fn=deliver_fn,
    )
    print(
        f'[{stamp}] status={run.status} sources={run.sources_count} '
        f'failures={store.get_job(JOB_ID).consecutive_failures}'
    )


async def run_demo() -> None:
    llm_cfg = build_llm_cfg()
    print(f'LLM: model={llm_cfg["model"]} server={llm_cfg["model_server"]}')
    print(f'Cadence: every {INTERVAL_SECONDS}s for {RUN_FOR_SECONDS}s, then exit')
    print()

    example_dir = Path(__file__).resolve().parent
    db_path = example_dir / 'scheduling_example.sqlite'
    if db_path.exists():
        db_path.unlink()
    store = JobStore(dsn=f'sqlite:///{db_path}')
    ensure_job(store, now=time.time())

    deadline = time.monotonic() + RUN_FOR_SECONDS
    tick = 0
    while True:
        tick += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(f'\n=== tick {tick} ({remaining:.0f}s left) ===')
        try:
            await run_once(store, llm_cfg, tick=tick)
        except Exception as exc:
            print(f'tick {tick} failed: {type(exc).__name__}: {exc}')

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_for = min(INTERVAL_SECONDS, remaining)
        print(f'sleeping {sleep_for:.0f}s …')
        await asyncio.sleep(sleep_for)

    print()
    print(f'Done after {tick} tick(s). Store: {db_path}')
    print('Exiting.')


def main() -> None:
    if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
        print(f'Missing OLLAMA_API_KEY in {REPO_ROOT / ".env"} — see .env.example.')
        sys.exit(1)
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        print("Install the scheduler extra: pip install 'cat-agent[scheduler]'")
        sys.exit(1)
    asyncio.run(run_demo())


if __name__ == '__main__':
    main()
