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

"""FastAPI application factory for on-demand agent invoke."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from cat_agent.serve.errors import error_body, log_run_exception
from cat_agent.serve.jobs import InlineJobTable, JobNotFound, JobTableFull
from cat_agent.serve.middleware import install_request_id_middleware, log_access
from cat_agent.serve.models import (
    AgentInfoOut,
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
    RunRequest,
    RunResponse,
)
from cat_agent.serve.registry import AgentRegistry, CapacityFull, normalize_agent_name
from cat_agent.serve.stream import final_content, messages_to_dicts, sse_event

try:
    from fastapi import Request
except ImportError:  # pragma: no cover
    Request = Any  # type: ignore[misc, assignment]


def _require_fastapi():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cat-agent serve requires FastAPI. Install with: pip install 'cat-agent[serve]'"
        ) from exc
    return Depends, FastAPI, Header, HTTPException, JSONResponse, StreamingResponse


def _retry_after_seconds() -> int:
    from cat_agent.settings import SERVE_RETRY_AFTER_SECONDS
    return max(1, int(SERVE_RETRY_AFTER_SECONDS))


def create_app(
    registry: AgentRegistry,
    *,
    title: str = 'Cat-Agent Serve',
    bearer_token: Optional[str] = None,
) -> Any:
    """Build a FastAPI app that exposes *registry* agents over HTTP.

    Endpoints:
      * ``GET /health`` — backwards-compatible liveness-ish status
      * ``GET /healthz`` — liveness (never touches agents / LLMs)
      * ``GET /readyz`` — readiness (200 only when every agent is ready)
      * ``GET /agents``
      * ``POST /agents/{name}/run`` — JSON or SSE (``stream=true``)
      * ``POST /agents/{name}/jobs`` — enqueue an inline async job (202)
      * ``GET /agents/{name}/jobs/{job_id}`` — poll job status
      * ``DELETE /agents/{name}/jobs/{job_id}`` — cancel a job

    Deferred agents registered via :meth:`AgentRegistry.register_factory` are
    built sequentially during lifespan startup (off-thread). Construction
    failures leave the process bound so ``/readyz`` can report the reason.

    Capacity: when an agent is at ``max_concurrency`` and its waiter count
    would exceed ``max_queue``, the request gets HTTP 429 with ``Retry-After``
    (waiter-counter only — no ``queue_timeout``).
    """
    Depends, FastAPI, Header, HTTPException, JSONResponse, StreamingResponse = (
        _require_fastapi()
    )

    if not isinstance(registry, AgentRegistry):
        raise TypeError(f'Expected AgentRegistry, got {type(registry)!r}')
    if len(registry) == 0:
        raise ValueError('AgentRegistry is empty; register at least one agent')

    expected_token = (bearer_token or '').strip() or None

    from cat_agent.settings import SERVE_JOB_MAX, SERVE_JOB_TTL_SECONDS

    jobs = InlineJobTable(
        max_jobs=SERVE_JOB_MAX,
        finished_ttl_seconds=SERVE_JOB_TTL_SECONDS,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await registry.build_deferred()
        yield
        await jobs.shutdown()
        for agent in list(registry.ready_agents()):
            aclose = getattr(agent, 'aclose', None)
            if aclose is not None:
                await aclose()

    app = FastAPI(title=title, lifespan=lifespan)
    app.state.registry = registry
    app.state.jobs = jobs
    install_request_id_middleware(app)

    async def require_auth(
        authorization: Optional[str] = Header(default=None),
    ) -> None:
        if expected_token is None:
            return
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail='Missing Bearer token')
        token = authorization[len('Bearer '):].strip()
        if token != expected_token:
            raise HTTPException(status_code=403, detail='Invalid Bearer token')

    def _resolve_key(name: str) -> str:
        try:
            key = normalize_agent_name(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if key not in registry:
            raise HTTPException(status_code=404, detail=f'Unknown agent: {key!r}')
        return key

    def _resolve_ready_agent(name: str):
        key = _resolve_key(name)
        state = registry.get_slot_state(key)
        if state != 'ready':
            _, breakdown = registry.readiness_payload()
            raise HTTPException(
                status_code=503,
                detail={
                    'status': 'not_ready',
                    'agent': key,
                    'state': state,
                    'agents': {key: breakdown.get(key, {'state': state})},
                },
            )
        try:
            return key, registry.get(key)
        except KeyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _request_id(request: Request) -> str:
        return getattr(request.state, 'request_id', None) or 'unknown'

    def _capacity_response(exc: CapacityFull, request_id: str, *, queue_wait_ms: float, started: float) -> Any:
        log_access(
            request_id=request_id,
            agent=exc.agent,
            outcome='capacity_full',
            status=429,
            duration_ms=(time.monotonic() - started) * 1000,
            queue_wait_ms=queue_wait_ms,
            stream=False,
        )
        return JSONResponse(
            status_code=429,
            content={
                'agent': exc.agent,
                'error_type': 'CapacityFull',
                'error': str(exc),
                'request_id': request_id,
            },
            headers={'Retry-After': str(_retry_after_seconds())},
        )

    def _run_error_response(
        agent: str,
        exc: BaseException,
        request_id: str,
        *,
        queue_wait_ms: float,
        started: float,
        stream: bool,
    ) -> Any:
        log_run_exception(exc, agent=agent, request_id=request_id)
        log_access(
            request_id=request_id,
            agent=agent,
            outcome='error',
            status=500,
            duration_ms=(time.monotonic() - started) * 1000,
            queue_wait_ms=queue_wait_ms,
            stream=stream,
        )
        return JSONResponse(
            status_code=500,
            content=error_body(agent, exc, request_id=request_id),
        )

    @app.get('/health')
    async def health() -> Dict[str, Any]:
        return {'status': 'ok', 'agents': len(registry)}

    @app.get('/healthz')
    async def healthz() -> Dict[str, str]:
        """Liveness: process is up. Never touches agents or LLMs."""
        return {'status': 'ok'}

    @app.get('/readyz')
    async def readyz():
        all_ready, agents = registry.readiness_payload()
        if all_ready:
            return JSONResponse(
                status_code=200,
                content={'status': 'ready', 'agents': agents},
            )
        return JSONResponse(
            status_code=503,
            content={'status': 'not_ready', 'agents': agents},
        )

    @app.get('/agents', response_model=List[AgentInfoOut], dependencies=[Depends(require_auth)])
    async def list_agents() -> List[AgentInfoOut]:
        return [
            AgentInfoOut(
                name=info.name,
                description=info.description,
                agent_class=info.agent_class,
                max_concurrency=info.max_concurrency,
            )
            for info in registry.list_info()
        ]

    @app.post(
        '/agents/{name}/run',
        response_model=None,
        dependencies=[Depends(require_auth)],
    )
    async def run_agent(name: str, body: RunRequest, request: Request):
        request_id = _request_id(request)
        started = time.monotonic()
        queue_wait_ms = 0.0
        key, agent = _resolve_ready_agent(name)
        messages = [m.to_dict() for m in body.messages]
        kwargs: Dict[str, Any] = {}
        if body.run_timeout is not None:
            kwargs['run_timeout'] = body.run_timeout

        if body.stream:
            acquire_started = time.monotonic()
            try:
                await registry.acquire_run_slot(key)
            except CapacityFull as exc:
                queue_wait_ms = (time.monotonic() - acquire_started) * 1000
                return _capacity_response(
                    exc, request_id, queue_wait_ms=queue_wait_ms, started=started,
                )
            queue_wait_ms = (time.monotonic() - acquire_started) * 1000

            stream = agent.arun(messages, **kwargs)
            aiter = stream.__aiter__()
            try:
                first_turn = await aiter.__anext__()
            except StopAsyncIteration:
                registry.release_run_slot(key)
                log_access(
                    request_id=request_id,
                    agent=key,
                    outcome='ok',
                    status=200,
                    duration_ms=(time.monotonic() - started) * 1000,
                    queue_wait_ms=queue_wait_ms,
                    stream=True,
                )
                return JSONResponse(
                    status_code=200,
                    content={'agent': key, 'messages': [], 'content': None},
                )
            except Exception as exc:
                registry.release_run_slot(key)
                return _run_error_response(
                    key, exc, request_id,
                    queue_wait_ms=queue_wait_ms, started=started, stream=True,
                )

            async def event_gen() -> AsyncIterator[str]:
                outcome = 'ok'
                try:
                    yield sse_event({
                        'type': 'turn',
                        'agent': key,
                        'messages': messages_to_dicts(first_turn),
                        'content': final_content(first_turn),
                    })
                    try:
                        async for turn in aiter:
                            yield sse_event({
                                'type': 'turn',
                                'agent': key,
                                'messages': messages_to_dicts(turn),
                                'content': final_content(turn),
                            })
                        yield sse_event({'type': 'done', 'agent': key})
                    except Exception as exc:
                        outcome = 'error'
                        log_run_exception(exc, agent=key, request_id=request_id)
                        body_err = error_body(key, exc, request_id=request_id)
                        yield sse_event({
                            'type': 'error',
                            'agent': body_err['agent'],
                            'error_type': body_err['error_type'],
                            'error': body_err['error'],
                            'request_id': request_id,
                        })
                finally:
                    registry.release_run_slot(key)
                    log_access(
                        request_id=request_id,
                        agent=key,
                        outcome=outcome,
                        status=200,
                        duration_ms=(time.monotonic() - started) * 1000,
                        queue_wait_ms=queue_wait_ms,
                        stream=True,
                    )

            return StreamingResponse(
                event_gen(),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                },
            )

        acquire_started = time.monotonic()
        try:
            await registry.acquire_run_slot(key)
        except CapacityFull as exc:
            queue_wait_ms = (time.monotonic() - acquire_started) * 1000
            return _capacity_response(
                exc, request_id, queue_wait_ms=queue_wait_ms, started=started,
            )
        queue_wait_ms = (time.monotonic() - acquire_started) * 1000

        try:
            try:
                last = await agent.arun_nonstream(messages, **kwargs)
            except Exception as exc:
                return _run_error_response(
                    key, exc, request_id,
                    queue_wait_ms=queue_wait_ms, started=started, stream=False,
                )
        finally:
            registry.release_run_slot(key)

        dicts = messages_to_dicts(last)
        log_access(
            request_id=request_id,
            agent=key,
            outcome='ok',
            status=200,
            duration_ms=(time.monotonic() - started) * 1000,
            queue_wait_ms=queue_wait_ms,
            stream=False,
        )
        return JSONResponse(
            status_code=200,
            content=RunResponse(
                agent=key, messages=dicts, content=final_content(dicts),
            ).model_dump(),
        )

    @app.post(
        '/agents/{name}/jobs',
        response_model=JobCreateResponse,
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    async def create_job(name: str, body: JobCreateRequest, request: Request):
        request_id = _request_id(request)
        key, agent = _resolve_ready_agent(name)
        messages = [m.to_dict() for m in body.messages]
        kwargs: Dict[str, Any] = {}
        if body.run_timeout is not None:
            kwargs['run_timeout'] = body.run_timeout

        async def runner():
            await registry.acquire_run_slot(key)
            try:
                last = await agent.arun_nonstream(messages, **kwargs)
                dicts = messages_to_dicts(last)
                return {
                    'messages': dicts,
                    'content': final_content(dicts),
                }
            finally:
                registry.release_run_slot(key)

        try:
            job_id = await jobs.submit(key, runner)
        except JobTableFull as exc:
            log_access(
                request_id=request_id,
                agent=key,
                outcome='job_table_full',
                status=429,
                duration_ms=0.0,
                queue_wait_ms=0.0,
                stream=False,
            )
            return JSONResponse(
                status_code=429,
                content={
                    'agent': key,
                    'error_type': 'JobTableFull',
                    'error': str(exc),
                    'request_id': request_id,
                },
                headers={'Retry-After': str(_retry_after_seconds())},
            )
        return JobCreateResponse(job_id=job_id)

    @app.get(
        '/agents/{name}/jobs/{job_id}',
        response_model=JobStatusResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_job(name: str, job_id: str):
        key = _resolve_key(name)
        try:
            rec = await jobs.get(key, job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f'Unknown job: {job_id!r}') from exc
        return JobStatusResponse(**rec.to_dict())

    @app.delete(
        '/agents/{name}/jobs/{job_id}',
        response_model=JobStatusResponse,
        dependencies=[Depends(require_auth)],
    )
    async def delete_job(name: str, job_id: str):
        key = _resolve_key(name)
        try:
            rec = await jobs.cancel(key, job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f'Unknown job: {job_id!r}') from exc
        return JobStatusResponse(**rec.to_dict())

    return app
