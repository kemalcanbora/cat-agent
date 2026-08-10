# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQLAlchemy Core persistence for scheduled jobs and collected sources.

Requires the ``scheduler`` extra (``pip install 'cat-agent[scheduler]'``).
SQLAlchemy is imported lazily so the base install stays dependency-free.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cat_agent.scheduling.models import Job, JobRun, Source
from cat_agent.settings import DEFAULT_WORKSPACE, SCHEDULER_DSN

_TRACKING_PARAMS = frozenset({
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'utm_id', 'utm_reader', 'utm_name', 'utm_referrer',
    'fbclid', 'gclid', 'mc_cid', 'mc_eid', 'ref',
})

_JOB_COLUMNS = (
    'id', 'user_id', 'kind', 'topic', 'interval_seconds', 'cron_expr',
    'timezone', 'channel', 'target', 'enabled', 'next_run_at', 'last_run_at',
    'lease_owner', 'lease_until', 'consecutive_failures', 'created_at', 'updated_at',
)

_RUN_COLUMNS = (
    'id', 'job_id', 'started_at', 'finished_at', 'status',
    'sources_count', 'error', 'trace_id',
)

_SOURCE_COLUMNS = (
    'id', 'user_id', 'job_id', 'url', 'title', 'summary', 'tags',
    'collected_at', 'delivered_at', 'content_hash',
)


def _require_sqlalchemy():
    try:
        import sqlalchemy
        from sqlalchemy import (
            Boolean,
            Column,
            Float,
            Integer,
            MetaData,
            String,
            Table,
            Text,
            and_,
            create_engine,
            or_,
            select,
            text,
            update,
        )
        from sqlalchemy.engine import Engine
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Scheduling persistence requires the 'scheduler' extra. "
            "Install with: pip install 'cat-agent[scheduler]'"
        ) from exc
    return {
        'sqlalchemy': sqlalchemy,
        'Boolean': Boolean,
        'Column': Column,
        'Float': Float,
        'Integer': Integer,
        'MetaData': MetaData,
        'String': String,
        'Table': Table,
        'Text': Text,
        'and_': and_,
        'create_engine': create_engine,
        'or_': or_,
        'select': select,
        'text': text,
        'update': update,
        'Engine': Engine,
    }


def default_scheduler_dsn() -> str:
    """Resolve ``CAT_AGENT_SCHEDULER_DSN`` or the workspace SQLite default."""
    if SCHEDULER_DSN:
        return SCHEDULER_DSN
    root = os.path.join(DEFAULT_WORKSPACE, 'scheduling')
    os.makedirs(root, exist_ok=True)
    path = os.path.abspath(os.path.join(root, 'scheduling.sqlite'))
    return f'sqlite:///{path}'


def normalize_url(url: str) -> str:
    """Normalize a URL for stable dedupe hashing.

    Lowercases the host, strips ``utm_*`` / ``fbclid`` (and similar) query
    params, and removes trailing slash and fragment.
    """
    raw = (url or '').strip()
    if not raw:
        return ''
    parts = urlsplit(raw)
    scheme = (parts.scheme or 'https').lower()
    netloc = parts.netloc.lower()
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    path = parts.path or ''
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ''))


def source_id_for(user_id: str, url: str) -> str:
    """Stable primary key: ``sha256(user_id + '|' + normalized_url)[:16]``."""
    normalized = normalize_url(url)
    digest = hashlib.sha256(f'{user_id}|{normalized}'.encode('utf-8')).hexdigest()
    return digest[:16]


def content_hash_for(title: str, summary: str) -> str:
    payload = f'{title or ""}\n{summary or ""}'.encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:32]


def _row_to_job(row: Any) -> Job:
    mapping = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
    return Job(
        id=mapping['id'],
        user_id=mapping['user_id'],
        kind=mapping['kind'],
        topic=mapping['topic'],
        interval_seconds=mapping.get('interval_seconds'),
        cron_expr=mapping.get('cron_expr'),
        timezone=mapping.get('timezone') or 'UTC',
        channel=mapping['channel'],
        target=mapping['target'],
        enabled=bool(mapping.get('enabled', True)),
        next_run_at=float(mapping.get('next_run_at') or 0.0),
        last_run_at=mapping.get('last_run_at'),
        lease_owner=mapping.get('lease_owner'),
        lease_until=mapping.get('lease_until'),
        consecutive_failures=int(mapping.get('consecutive_failures') or 0),
        created_at=float(mapping.get('created_at') or 0.0),
        updated_at=float(mapping.get('updated_at') or 0.0),
    )


def _row_to_run(row: Any) -> JobRun:
    mapping = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
    return JobRun(
        id=mapping['id'],
        job_id=mapping['job_id'],
        started_at=float(mapping['started_at']),
        finished_at=mapping.get('finished_at'),
        status=mapping['status'],
        sources_count=int(mapping.get('sources_count') or 0),
        error=mapping.get('error'),
        trace_id=mapping.get('trace_id'),
    )


def _row_to_source(row: Any) -> Source:
    mapping = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
    return Source(
        id=mapping['id'],
        user_id=mapping['user_id'],
        job_id=mapping.get('job_id'),
        url=mapping['url'],
        title=mapping.get('title') or '',
        summary=mapping.get('summary') or '',
        tags=mapping.get('tags') or '',
        collected_at=float(mapping['collected_at']),
        delivered_at=mapping.get('delivered_at'),
        content_hash=mapping.get('content_hash'),
    )


class JobStore:
    """Jobs + job_runs persistence (SQLAlchemy Core)."""

    def __init__(self, dsn: Optional[str] = None):
        sa = _require_sqlalchemy()
        self._sa = sa
        self.dsn = dsn or default_scheduler_dsn()
        connect_args = {}
        if self.dsn.startswith('sqlite'):
            connect_args['check_same_thread'] = False
        self.engine = sa['create_engine'](
            self.dsn,
            future=True,
            connect_args=connect_args,
        )
        self.metadata = sa['MetaData']()
        self._build_tables()
        self.metadata.create_all(self.engine)
        self._ensure_indexes()

    @property
    def dialect_name(self) -> str:
        return self.engine.dialect.name

    def _build_tables(self) -> None:
        sa = self._sa
        Column, String, Text, Integer, Float, Boolean, Table = (
            sa['Column'], sa['String'], sa['Text'], sa['Integer'],
            sa['Float'], sa['Boolean'], sa['Table'],
        )
        self.jobs = Table(
            'jobs',
            self.metadata,
            Column('id', String(128), primary_key=True),
            Column('user_id', String(128), nullable=False, index=True),
            Column('kind', String(64), nullable=False),
            Column('topic', Text, nullable=False),
            Column('interval_seconds', Integer, nullable=True),
            Column('cron_expr', String(128), nullable=True),
            Column('timezone', String(64), nullable=False, server_default='UTC'),
            Column('channel', String(32), nullable=False),
            Column('target', Text, nullable=False),
            Column('enabled', Boolean, nullable=False, server_default='1'),
            Column('next_run_at', Float, nullable=False, index=True),
            Column('last_run_at', Float, nullable=True),
            Column('lease_owner', String(256), nullable=True),
            Column('lease_until', Float, nullable=True),
            Column('consecutive_failures', Integer, nullable=False, server_default='0'),
            Column('created_at', Float, nullable=False),
            Column('updated_at', Float, nullable=False),
        )
        self.job_runs = Table(
            'job_runs',
            self.metadata,
            Column('id', String(64), primary_key=True),
            Column('job_id', String(128), nullable=False, index=True),
            Column('started_at', Float, nullable=False),
            Column('finished_at', Float, nullable=True),
            Column('status', String(32), nullable=False),
            Column('sources_count', Integer, nullable=False, server_default='0'),
            Column('error', Text, nullable=True),
            Column('trace_id', String(128), nullable=True),
        )
        self.sources = Table(
            'sources',
            self.metadata,
            Column('id', String(32), primary_key=True),
            Column('user_id', String(128), nullable=False),
            Column('job_id', String(128), nullable=True),
            Column('url', Text, nullable=False),
            Column('title', Text, nullable=False, server_default=''),
            Column('summary', Text, nullable=False, server_default=''),
            Column('tags', Text, nullable=False, server_default=''),
            Column('collected_at', Float, nullable=False),
            Column('delivered_at', Float, nullable=True),
            Column('content_hash', String(64), nullable=True),
        )

    def _ensure_indexes(self) -> None:
        # Composite indexes used by watermark / lease queries.
        stmts = [
            'CREATE INDEX IF NOT EXISTS idx_sources_user_collected '
            'ON sources (user_id, collected_at)',
            'CREATE INDEX IF NOT EXISTS idx_sources_user_undelivered '
            'ON sources (user_id, delivered_at)',
            'CREATE INDEX IF NOT EXISTS idx_jobs_due '
            'ON jobs (enabled, next_run_at)',
        ]
        with self.engine.begin() as conn:
            for stmt in stmts:
                conn.execute(self._sa['text'](stmt))

    # --- jobs CRUD ---------------------------------------------------------

    def upsert_job(self, job: Job) -> Job:
        now = time.time()
        if not job.created_at:
            job.created_at = now
        job.updated_at = now
        values = {col: getattr(job, col) for col in _JOB_COLUMNS}
        with self.engine.begin() as conn:
            existing = conn.execute(
                self._sa['select'](self.jobs.c.id).where(self.jobs.c.id == job.id)
            ).first()
            if existing:
                update_vals = {k: v for k, v in values.items() if k != 'id'}
                conn.execute(
                    self._sa['update'](self.jobs)
                    .where(self.jobs.c.id == job.id)
                    .values(**update_vals)
                )
            else:
                conn.execute(self.jobs.insert().values(**values))
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self.engine.connect() as conn:
            row = conn.execute(
                self._sa['select'](self.jobs).where(self.jobs.c.id == job_id)
            ).first()
        return _row_to_job(row) if row else None

    def list_jobs(
        self,
        *,
        user_id: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[Job]:
        sa = self._sa
        stmt = sa['select'](self.jobs)
        if user_id is not None:
            stmt = stmt.where(self.jobs.c.user_id == user_id)
        if enabled_only:
            stmt = stmt.where(self.jobs.c.enabled.is_(True))
        stmt = stmt.order_by(self.jobs.c.created_at.asc())
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_job(r) for r in rows]

    def count_jobs_for_user(self, user_id: str) -> int:
        sa = self._sa
        sqlalchemy = sa['sqlalchemy']
        stmt = (
            sa['select'](sqlalchemy.func.count())
            .select_from(self.jobs)
            .where(self.jobs.c.user_id == user_id)
        )
        with self.engine.connect() as conn:
            return int(conn.execute(stmt).scalar() or 0)

    def delete_job(self, job_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(self.jobs.delete().where(self.jobs.c.id == job_id))
        return (result.rowcount or 0) > 0

    def set_enabled(self, job_id: str, enabled: bool) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self._sa['update'](self.jobs)
                .where(self.jobs.c.id == job_id)
                .values(enabled=enabled, updated_at=time.time())
            )

    def update_schedule_state(
        self,
        job_id: str,
        *,
        next_run_at: float,
        last_run_at: Optional[float] = None,
        consecutive_failures: Optional[int] = None,
        clear_lease: bool = False,
    ) -> None:
        values: dict = {
            'next_run_at': next_run_at,
            'updated_at': time.time(),
        }
        if last_run_at is not None:
            values['last_run_at'] = last_run_at
        if consecutive_failures is not None:
            values['consecutive_failures'] = consecutive_failures
        if clear_lease:
            values['lease_owner'] = None
            values['lease_until'] = None
        with self.engine.begin() as conn:
            conn.execute(
                self._sa['update'](self.jobs)
                .where(self.jobs.c.id == job_id)
                .values(**values)
            )

    def force_lease(
        self,
        job_id: str,
        *,
        owner: str,
        lease_until: float,
        now: Optional[float] = None,
    ) -> bool:
        """Assign a lease unconditionally (manual / APScheduler runs)."""
        when = time.time() if now is None else now
        with self.engine.begin() as conn:
            result = conn.execute(
                self._sa['update'](self.jobs)
                .where(self.jobs.c.id == job_id)
                .values(
                    lease_owner=owner,
                    lease_until=lease_until,
                    updated_at=when,
                )
            )
        return (result.rowcount or 0) > 0

    def renew_lease(self, job_id: str, *, owner: str, lease_until: float) -> bool:
        sa = self._sa
        with self.engine.begin() as conn:
            result = conn.execute(
                sa['update'](self.jobs)
                .where(
                    sa['and_'](
                        self.jobs.c.id == job_id,
                        self.jobs.c.lease_owner == owner,
                    )
                )
                .values(lease_until=lease_until, updated_at=time.time())
            )
        return (result.rowcount or 0) > 0

    def release_lease(self, job_id: str, *, owner: str) -> None:
        sa = self._sa
        with self.engine.begin() as conn:
            conn.execute(
                sa['update'](self.jobs)
                .where(
                    sa['and_'](
                        self.jobs.c.id == job_id,
                        self.jobs.c.lease_owner == owner,
                    )
                )
                .values(lease_owner=None, lease_until=None, updated_at=time.time())
            )

    def release_all_leases(self, owner: str) -> int:
        sa = self._sa
        with self.engine.begin() as conn:
            result = conn.execute(
                sa['update'](self.jobs)
                .where(self.jobs.c.lease_owner == owner)
                .values(lease_owner=None, lease_until=None, updated_at=time.time())
            )
        return int(result.rowcount or 0)

    def claim_due_jobs(
        self,
        *,
        limit: int,
        lease_seconds: int,
        owner: str,
        now: Optional[float] = None,
    ) -> List[Job]:
        """Atomically lease due jobs. Multi-process safe.

        Postgres: ``SELECT ... FOR UPDATE SKIP LOCKED`` then conditional UPDATE.
        SQLite: ``BEGIN IMMEDIATE`` + conditional UPDATE confirming ``rowcount``.
        """
        now = time.time() if now is None else now
        lease_until = now + lease_seconds
        if self.dialect_name == 'postgresql':
            return self._claim_due_jobs_postgres(
                limit=limit, owner=owner, now=now, lease_until=lease_until,
            )
        return self._claim_due_jobs_sqlite(
            limit=limit, owner=owner, now=now, lease_until=lease_until,
        )

    def _claim_due_jobs_sqlite(
        self,
        *,
        limit: int,
        owner: str,
        now: float,
        lease_until: float,
    ) -> List[Job]:
        """Claim with an exclusive SQLite write lock (BEGIN IMMEDIATE)."""
        sa = self._sa
        claimed: List[Job] = []
        # AUTOCOMMIT so we can issue BEGIN IMMEDIATE ourselves.
        with self.engine.connect().execution_options(
            isolation_level='AUTOCOMMIT',
        ) as conn:
            conn.execute(sa['text']('BEGIN IMMEDIATE'))
            try:
                due_stmt = (
                    sa['select'](self.jobs.c.id)
                    .where(
                        sa['and_'](
                            self.jobs.c.enabled.is_(True),
                            self.jobs.c.next_run_at <= now,
                            sa['or_'](
                                self.jobs.c.lease_until.is_(None),
                                self.jobs.c.lease_until < now,
                            ),
                        )
                    )
                    .order_by(self.jobs.c.next_run_at.asc())
                    .limit(limit)
                )
                candidate_ids = [r[0] for r in conn.execute(due_stmt).fetchall()]
                for job_id in candidate_ids:
                    result = conn.execute(
                        sa['update'](self.jobs)
                        .where(
                            sa['and_'](
                                self.jobs.c.id == job_id,
                                self.jobs.c.enabled.is_(True),
                                self.jobs.c.next_run_at <= now,
                                sa['or_'](
                                    self.jobs.c.lease_until.is_(None),
                                    self.jobs.c.lease_until < now,
                                ),
                            )
                        )
                        .values(
                            lease_owner=owner,
                            lease_until=lease_until,
                            updated_at=now,
                        )
                    )
                    if (result.rowcount or 0) == 1:
                        row = conn.execute(
                            sa['select'](self.jobs).where(self.jobs.c.id == job_id)
                        ).first()
                        if row:
                            claimed.append(_row_to_job(row))
                conn.execute(sa['text']('COMMIT'))
            except Exception:
                conn.execute(sa['text']('ROLLBACK'))
                raise
        return claimed

    def _claim_due_jobs_postgres(
        self,
        *,
        limit: int,
        owner: str,
        now: float,
        lease_until: float,
    ) -> List[Job]:
        sa = self._sa
        claimed: List[Job] = []
        with self.engine.begin() as conn:
            due_stmt = (
                sa['select'](self.jobs.c.id)
                .where(
                    sa['and_'](
                        self.jobs.c.enabled.is_(True),
                        self.jobs.c.next_run_at <= now,
                        sa['or_'](
                            self.jobs.c.lease_until.is_(None),
                            self.jobs.c.lease_until < now,
                        ),
                    )
                )
                .order_by(self.jobs.c.next_run_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            candidate_ids = [r[0] for r in conn.execute(due_stmt).fetchall()]
            for job_id in candidate_ids:
                result = conn.execute(
                    sa['update'](self.jobs)
                    .where(
                        sa['and_'](
                            self.jobs.c.id == job_id,
                            sa['or_'](
                                self.jobs.c.lease_until.is_(None),
                                self.jobs.c.lease_until < now,
                            ),
                        )
                    )
                    .values(
                        lease_owner=owner,
                        lease_until=lease_until,
                        updated_at=now,
                    )
                )
                if (result.rowcount or 0) == 1:
                    row = conn.execute(
                        sa['select'](self.jobs).where(self.jobs.c.id == job_id)
                    ).first()
                    if row:
                        claimed.append(_row_to_job(row))
        return claimed

    # --- job_runs ----------------------------------------------------------

    def insert_run(self, run: JobRun) -> JobRun:
        values = {col: getattr(run, col) for col in _RUN_COLUMNS}
        with self.engine.begin() as conn:
            conn.execute(self.job_runs.insert().values(**values))
        return run

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: Optional[float] = None,
        sources_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self._sa['update'](self.job_runs)
                .where(self.job_runs.c.id == run_id)
                .values(
                    status=status,
                    finished_at=finished_at if finished_at is not None else time.time(),
                    sources_count=sources_count,
                    error=error,
                )
            )

    def get_run(self, run_id: str) -> Optional[JobRun]:
        with self.engine.connect() as conn:
            row = conn.execute(
                self._sa['select'](self.job_runs).where(self.job_runs.c.id == run_id)
            ).first()
        return _row_to_run(row) if row else None

    def list_runs(self, job_id: str, *, limit: int = 50) -> List[JobRun]:
        sa = self._sa
        stmt = (
            sa['select'](self.job_runs)
            .where(self.job_runs.c.job_id == job_id)
            .order_by(self.job_runs.c.started_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_run(r) for r in rows]

    # --- sources -----------------------------------------------------------

    def save_source(
        self,
        *,
        user_id: str,
        url: str,
        title: str,
        summary: str,
        tags: str = '',
        job_id: Optional[str] = None,
        collected_at: Optional[float] = None,
        content_hash: Optional[str] = None,
    ) -> Tuple[Source, bool]:
        """Insert a source. Returns ``(source, created)``.

        Identical normalized URLs for the same user collide on the primary key
        and are treated as a no-op (dedupe).
        """
        sid = source_id_for(user_id, url)
        normalized = normalize_url(url)
        now = collected_at if collected_at is not None else time.time()
        chash = content_hash or content_hash_for(title, summary)
        source = Source(
            id=sid,
            user_id=user_id,
            job_id=job_id,
            url=normalized or url,
            title=title or '',
            summary=summary or '',
            tags=tags or '',
            collected_at=now,
            delivered_at=None,
            content_hash=chash,
        )
        with self.engine.begin() as conn:
            existing = conn.execute(
                self._sa['select'](self.sources.c.id).where(self.sources.c.id == sid)
            ).first()
            if existing:
                row = conn.execute(
                    self._sa['select'](self.sources).where(self.sources.c.id == sid)
                ).first()
                return _row_to_source(row), False
            conn.execute(
                self.sources.insert().values(
                    **{col: getattr(source, col) for col in _SOURCE_COLUMNS}
                )
            )
        return source, True

    def get_source(self, source_id: str) -> Optional[Source]:
        with self.engine.connect() as conn:
            row = conn.execute(
                self._sa['select'](self.sources).where(self.sources.c.id == source_id)
            ).first()
        return _row_to_source(row) if row else None

    def list_undelivered(
        self,
        user_id: str,
        *,
        max_items: int = 50,
        job_id: Optional[str] = None,
    ) -> List[Source]:
        """Watermark query: only rows where ``delivered_at IS NULL``."""
        sa = self._sa
        conditions = [
            self.sources.c.user_id == user_id,
            self.sources.c.delivered_at.is_(None),
        ]
        if job_id is not None:
            conditions.append(self.sources.c.job_id == job_id)
        stmt = (
            sa['select'](self.sources)
            .where(sa['and_'](*conditions))
            .order_by(self.sources.c.collected_at.asc())
            .limit(max_items)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_source(r) for r in rows]

    def mark_delivered(
        self,
        source_ids: Sequence[str],
        *,
        delivered_at: Optional[float] = None,
    ) -> int:
        if not source_ids:
            return 0
        when = delivered_at if delivered_at is not None else time.time()
        with self.engine.begin() as conn:
            result = conn.execute(
                self._sa['update'](self.sources)
                .where(self.sources.c.id.in_(list(source_ids)))
                .values(delivered_at=when)
            )
        return int(result.rowcount or 0)

    def list_sources_for_user(
        self,
        user_id: str,
        *,
        limit: int = 100,
        undelivered_only: bool = False,
    ) -> List[Source]:
        sa = self._sa
        stmt = sa['select'](self.sources).where(self.sources.c.user_id == user_id)
        if undelivered_only:
            stmt = stmt.where(self.sources.c.delivered_at.is_(None))
        stmt = stmt.order_by(self.sources.c.collected_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_source(r) for r in rows]


# Alias used by the prompt / docs.
SourceStore = JobStore


def new_run_id() -> str:
    return uuid.uuid4().hex


def make_job_id(user_id: str, topic: str, kind: str = 'collect_and_report') -> str:
    slug = ''.join(c if c.isalnum() or c in '-_' else '-' for c in topic.lower())
    slug = '-'.join(filter(None, slug.split('-')))[:48] or 'topic'
    return f'report:{user_id}:{slug}'
