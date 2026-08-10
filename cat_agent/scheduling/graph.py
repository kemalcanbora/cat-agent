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

"""Report StateGraph: fetch → dedupe → summarize → redact → deliver."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

from cat_agent.agents.assistant import Assistant
from cat_agent.graph import END, AgentNode, FunctionNode, StateGraph
from cat_agent.graph.state import GraphState
from cat_agent.llm.schema import Message
from cat_agent.scheduling.channels.base import (
    get_channel,
    markdown_to_html,
    send_with_retry,
)
from cat_agent.scheduling.models import Job, Source
from cat_agent.scheduling.store import JobStore
from cat_agent.scheduling.tools import scheduling_context
from cat_agent.security.pii import redact_text
from cat_agent.tools.base import enable_optional_tools
from cat_agent.utils.rate_limit import RateLimiter

_COLLECTOR_SYSTEM = (
    'You are a source collector. Search the web for recent, high-quality sources '
    'on the given topic. For each distinct source, call save_source exactly once '
    'with user_id, url, title, summary, and optional tags. '
    'Never summarize sources in chat — only call save_source. '
    'Prefer primary sources and reputable outlets. Stop after a reasonable set '
    '(roughly 5–15 sources).'
)

_REPORTER_SYSTEM = (
    'You are a report writer. You receive a list of pre-fetched, pre-deduped '
    'sources. Produce a concise Markdown report with: a short executive summary, '
    'then a bullet list of sources (title, one-line takeaway, URL). '
    'Do not invent sources. Do not call tools — you have none.'
)


def _default_rate_limiter() -> RateLimiter:
    return RateLimiter(requests_per_interval=2, interval_seconds=1.0, max_concurrency=2)


def build_collector(
    *,
    llm=None,
    rate_limiter: Optional[RateLimiter] = None,
    handlers=None,
) -> Assistant:
    """Assistant with web_search / web_extractor / save_source."""
    return Assistant(
        function_list=['web_search', 'web_extractor', 'save_source'],
        llm=llm,
        system_message=_COLLECTOR_SYSTEM,
        name='source_collector',
        rate_limiter=rate_limiter or _default_rate_limiter(),
        handlers=handlers,
    )


def build_reporter(
    *,
    llm=None,
    rate_limiter: Optional[RateLimiter] = None,
    handlers=None,
) -> Assistant:
    """Assistant with no tools — summarizes pre-fetched sources."""
    return Assistant(
        function_list=[],
        llm=llm,
        system_message=_REPORTER_SYSTEM,
        name='source_reporter',
        rate_limiter=rate_limiter or _default_rate_limiter(),
        handlers=handlers,
    )


def _near_dup_key(source: Source) -> str:
    if source.content_hash:
        return source.content_hash
    blob = f'{source.title}\n{source.summary}'.strip().lower()
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:32]


def dedupe_sources(sources: Sequence[Source]) -> List[Source]:
    """Drop near-duplicates by content_hash (URL dedupe already happened at save)."""
    seen = set()
    out: List[Source] = []
    for src in sources:
        key = _near_dup_key(src)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def sources_to_prompt(job: Job, sources: Sequence[Source]) -> str:
    lines = [
        f'Topic: {job.topic}',
        f'User: {job.user_id}',
        f'Sources ({len(sources)}):',
        '',
    ]
    for i, s in enumerate(sources, 1):
        lines.append(f'{i}. {s.title}')
        lines.append(f'   URL: {s.url}')
        lines.append(f'   Summary: {s.summary}')
        if s.tags:
            lines.append(f'   Tags: {s.tags}')
        lines.append('')
    return '\n'.join(lines)


def build_report_graph(
    *,
    reporter: Optional[Assistant] = None,
    llm=None,
    rate_limiter: Optional[RateLimiter] = None,
) -> 'object':
    """Compile fetch → (empty?) → dedupe → summarize → redact → deliver."""
    agent = reporter or build_reporter(llm=llm, rate_limiter=rate_limiter)

    def fetch(state: GraphState) -> GraphState:
        store: JobStore = state.scratch['store']
        job: Job = state.scratch['job']
        max_items = int(state.scratch.get('max_items') or 50)
        if state.scratch.get('sources') is None:
            state.scratch['sources'] = store.list_undelivered(
                job.user_id, max_items=max_items,
            )
        return state

    def route_after_fetch(state: GraphState) -> str:
        sources = state.scratch.get('sources') or []
        if not sources:
            state.scratch['empty'] = True
            return END
        return 'dedupe'

    def dedupe(state: GraphState) -> GraphState:
        state.scratch['sources'] = dedupe_sources(state.scratch.get('sources') or [])
        return state

    def prepare_summarize(state: GraphState) -> GraphState:
        job: Job = state.scratch['job']
        sources = state.scratch.get('sources') or []
        state.messages = [Message(role='user', content=sources_to_prompt(job, sources))]
        return state

    def after_summarize(state: GraphState) -> GraphState:
        # Last assistant message is the Markdown report.
        body = ''
        for msg in reversed(state.messages):
            role = msg.role if hasattr(msg, 'role') else msg.get('role')
            content = msg.content if hasattr(msg, 'content') else msg.get('content')
            if role == 'assistant' and content:
                body = content if isinstance(content, str) else str(content)
                break
        state.scratch['report_markdown'] = body
        return state

    def redact(state: GraphState) -> GraphState:
        body = state.scratch.get('report_markdown') or ''
        state.scratch['report_markdown'] = redact_text(body)
        return state

    def deliver(state: GraphState) -> GraphState:
        if state.scratch.get('dry_run'):
            return state
        job: Job = state.scratch['job']
        body = state.scratch.get('report_markdown') or ''
        channel = state.scratch.get('channel') or get_channel(job.channel)
        # deliver is sync FunctionNode — run async send via helper stored earlier
        send_coro = send_with_retry(
            channel,
            target=job.target,
            subject=f'Scheduled report: {job.topic}',
            body_markdown=body,
            body_html=markdown_to_html(body),
        )
        runner = state.scratch.get('async_runner')
        if runner is None:
            import asyncio

            asyncio.get_event_loop().run_until_complete(send_coro)
        else:
            runner(send_coro)
        state.scratch['delivered'] = True
        return state

    graph = (
        StateGraph(max_steps=20)
        .add_node(FunctionNode('fetch', fetch))
        .add_node(FunctionNode('dedupe', dedupe))
        .add_node(FunctionNode('prepare_summarize', prepare_summarize))
        .add_node(AgentNode('summarize', agent))
        .add_node(FunctionNode('after_summarize', after_summarize))
        .add_node(FunctionNode('redact', redact))
        .add_node(FunctionNode('deliver', deliver))
        .set_entry('fetch')
        .add_conditional_edges('fetch', route_after_fetch)
        .add_edge('dedupe', 'prepare_summarize')
        .add_edge('prepare_summarize', 'summarize')
        .add_edge('summarize', 'after_summarize')
        .add_edge('after_summarize', 'redact')
        .add_edge('redact', 'deliver')
        .add_edge('deliver', END)
    )
    return graph.compile(name='ScheduledReportGraph')


async def run_collector(job: Job, store: JobStore, *, llm=None, handlers=None) -> int:
    """Enable network tools and run the collector Assistant."""
    enable_optional_tools('web_search', 'web_extractor')
    collector = build_collector(llm=llm, handlers=handlers)
    prompt = (
        f'Collect recent sources about: {job.topic}\n'
        f'user_id={job.user_id}\n'
        'Call save_source once per source.'
    )
    before = len(store.list_undelivered(job.user_id, max_items=10_000))
    with scheduling_context(store, job_id=job.id):
        await collector.arun_nonstream([{'role': 'user', 'content': prompt}])
    after = len(store.list_undelivered(job.user_id, max_items=10_000))
    return max(0, after - before)


async def build_report_markdown(
    job: Job,
    sources: Sequence[Source],
    *,
    llm=None,
    rate_limiter: Optional[RateLimiter] = None,
) -> str:
    """Summarize + redact without delivering (used by the shared runner)."""
    reporter = build_reporter(llm=llm, rate_limiter=rate_limiter)
    messages = [{'role': 'user', 'content': sources_to_prompt(job, dedupe_sources(sources))}]
    responses = await reporter.arun_nonstream(messages)
    body = ''
    for msg in reversed(responses or []):
        role = msg.role if hasattr(msg, 'role') else msg.get('role')
        content = msg.content if hasattr(msg, 'content') else msg.get('content')
        if role == 'assistant' and content:
            body = content if isinstance(content, str) else str(content)
            break
    return redact_text(body)


async def deliver_report(
    job: Job,
    body_markdown: str,
    sources: Sequence[Source],
    store: JobStore,
    *,
    channel=None,
) -> None:
    """Send via the job's channel. Caller marks ``delivered_at`` only on success."""
    del sources, store  # watermark is owned by runner
    ch = channel or get_channel(job.channel)
    result = await send_with_retry(
        ch,
        target=job.target,
        subject=f'Scheduled report: {job.topic}',
        body_markdown=body_markdown,
        body_html=markdown_to_html(body_markdown),
    )
    if not result.ok:
        raise RuntimeError(result.error or 'delivery failed')
