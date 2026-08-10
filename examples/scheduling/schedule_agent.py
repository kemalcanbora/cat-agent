"""Deployable scheduling assistant (HTTP + Nomad).

Same ``Job(interval_seconds=...)`` shape as ``scheduled_report_example.py``,
but persisted in ``CAT_AGENT_SCHEDULER_DSN`` and served over HTTP. A separate
``cat-agent schedule run-due`` worker executes jobs when ``next_run_at`` is due.

Local::

    pip install 'cat-agent[serve,platform,scheduler]'
    python examples/scheduling/schedule_agent.py

Deploy::

    cat-agent deploy --dir examples/scheduling
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Registers scheduling tools before deploy-time tool validation.
import cat_agent.scheduling  # noqa: F401
from cat_agent.agents import Assistant
from cat_agent.llm.env_config import apply_agent_yaml_env, llm_config_from_env
from cat_agent.scheduling.models import Job
from cat_agent.scheduling.store import JobStore, default_scheduler_dsn, make_job_id
from cat_agent.serve import AgentRegistry, create_app, run_app

_YAML = Path(__file__).resolve().parent / 'agent.yaml'


class ReportScheduler:
    """Seed one recurring job, then expose an HTTP assistant to manage more."""

    # Cadence and delivery — override via agent.yaml ``env:`` (see SCHEDULE_*).
    INTERVAL_SECONDS = 3600
    USER_ID = 'demo'
    TOPIC = 'AI news digest'
    CHANNEL = 'webhook'
    TARGET = 'http://127.0.0.1:9999/hook'

    def __init__(self) -> None:
        apply_agent_yaml_env(_YAML)
        self.interval_seconds = int(
            os.getenv('SCHEDULE_INTERVAL_SECONDS', str(self.INTERVAL_SECONDS))
        )
        self.user_id = os.getenv('SCHEDULE_USER_ID', self.USER_ID)
        self.topic = os.getenv('SCHEDULE_TOPIC', self.TOPIC)
        self.channel = os.getenv('SCHEDULE_CHANNEL', self.CHANNEL)
        self.target = os.getenv('SCHEDULE_WEBHOOK', self.TARGET)
        dsn = os.getenv('CAT_AGENT_SCHEDULER_DSN') or default_scheduler_dsn()
        self.store = JobStore(dsn=dsn)

    @property
    def job_id(self) -> str:
        return make_job_id(self.user_id, self.topic)

    def default_job(self) -> Job:
        now = time.time()
        return Job(
            id=self.job_id,
            user_id=self.user_id,
            kind='collect_and_report',
            topic=self.topic,
            interval_seconds=self.interval_seconds,
            channel=self.channel,
            target=self.target,
            enabled=True,
            next_run_at=now + self.interval_seconds,
            created_at=now,
            updated_at=now,
        )

    def seed_default_job(self) -> Job:
        job = self.default_job()
        self.store.upsert_job(job)
        return job

    def build(self) -> Assistant:
        llm = llm_config_from_env(agent_yaml=_YAML, model_type='oai')
        return Assistant(
            llm=llm,
            name='ReportScheduler',
            description='Creates and manages recurring report jobs.',
            system_message=(
                f'Default job {self.job_id!r} runs every {self.interval_seconds}s '
                f'on topic {self.topic!r}. Use create_schedule, list_schedules, '
                'cancel_schedule to manage jobs. Always call tools; do not invent ids.'
            ),
            function_list=['create_schedule', 'list_schedules', 'cancel_schedule'],
        )

    def registry(self) -> AgentRegistry:
        job = self.seed_default_job()
        print(
            f'Seeded {job.id!r}  interval_seconds={job.interval_seconds}  '
            f'channel={job.channel}  target={job.target}'
        )
        reg = AgentRegistry()
        reg.register(self.build(), name='report-scheduler')
        return reg


def registry() -> AgentRegistry:
    """Zero-arg entrypoint for ``cat-agent serve`` / Nomad deploy."""
    return ReportScheduler().registry()


def main() -> None:
    app = create_app(registry())
    print('Serving report-scheduler')
    print('  POST /agents/report-scheduler/run')
    run_app(app)


if __name__ == '__main__':
    main()
