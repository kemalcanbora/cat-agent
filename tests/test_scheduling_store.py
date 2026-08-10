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

"""Tests for scheduling JobStore / SourceStore (schema, URL dedupe, watermark)."""

from __future__ import annotations

import time

import pytest

from cat_agent.scheduling.models import Job
from cat_agent.scheduling.store import (
    JobStore,
    content_hash_for,
    make_job_id,
    normalize_url,
    source_id_for,
)


@pytest.fixture()
def store(tmp_path):
    dsn = f'sqlite:///{tmp_path / "scheduling.sqlite"}'
    return JobStore(dsn=dsn)


def _job(**overrides) -> Job:
    now = time.time()
    base = dict(
        id='report:alice:ai-news',
        user_id='alice',
        kind='collect_and_report',
        topic='AI news',
        interval_seconds=5 * 3600,
        channel='smtp',
        target='alice@example.com',
        enabled=True,
        next_run_at=now,
        created_at=now,
        updated_at=now,
    )
    base.update(overrides)
    return Job(**base)


class TestNormalizeUrl:
    def test_strips_tracking_params_and_fragment(self):
        url = (
            'https://Example.COM/path/article/?utm_source=x&utm_medium=y'
            '&fbclid=abc&keep=1#section'
        )
        assert normalize_url(url) == 'https://example.com/path/article?keep=1'

    def test_strips_trailing_slash_and_www(self):
        assert normalize_url('https://www.Example.com/a/b/') == 'https://example.com/a/b'

    def test_preserves_non_tracking_query(self):
        assert normalize_url('https://ex.com/q?page=2&sort=asc') == (
            'https://ex.com/q?page=2&sort=asc'
        )

    def test_empty(self):
        assert normalize_url('') == ''
        assert normalize_url('   ') == ''


class TestSourceId:
    def test_same_normalized_url_same_id(self):
        a = source_id_for('alice', 'https://WWW.News.com/story/?utm_source=feed')
        b = source_id_for('alice', 'https://news.com/story')
        assert a == b
        assert len(a) == 16

    def test_different_users_different_ids(self):
        a = source_id_for('alice', 'https://news.com/story')
        b = source_id_for('bob', 'https://news.com/story')
        assert a != b


class TestSchemaAndCrud:
    def test_creates_tables(self, store):
        assert store.dialect_name == 'sqlite'
        job = _job()
        store.upsert_job(job)
        loaded = store.get_job(job.id)
        assert loaded is not None
        assert loaded.user_id == 'alice'
        assert loaded.topic == 'AI news'
        assert loaded.interval_seconds == 5 * 3600

    def test_list_and_delete(self, store):
        store.upsert_job(_job(id='report:alice:a', topic='a'))
        store.upsert_job(_job(id='report:bob:b', user_id='bob', topic='b'))
        assert len(store.list_jobs(user_id='alice')) == 1
        assert store.count_jobs_for_user('alice') == 1
        assert store.delete_job('report:alice:a') is True
        assert store.get_job('report:alice:a') is None

    def test_make_job_id(self):
        assert make_job_id('alice', 'AI News!') == 'report:alice:ai-news'


class TestDedupe:
    def test_identical_urls_dedupe(self, store):
        s1, created1 = store.save_source(
            user_id='alice',
            url='https://News.com/article/?utm_campaign=x',
            title='One',
            summary='First',
            job_id='report:alice:ai-news',
        )
        s2, created2 = store.save_source(
            user_id='alice',
            url='https://news.com/article',
            title='Two',
            summary='Second',
            job_id='report:alice:ai-news',
        )
        assert created1 is True
        assert created2 is False
        assert s1.id == s2.id
        # Original row retained (first write wins).
        assert s2.title == 'One'

    def test_content_hash_helper(self):
        assert content_hash_for('t', 's') == content_hash_for('t', 's')
        assert content_hash_for('t', 's') != content_hash_for('t', 'other')


class TestWatermark:
    def test_undelivered_only(self, store):
        now = time.time()
        store.save_source(
            user_id='alice',
            url='https://a.example/1',
            title='A',
            summary='a',
            collected_at=now - 100,
        )
        s2, _ = store.save_source(
            user_id='alice',
            url='https://a.example/2',
            title='B',
            summary='b',
            collected_at=now - 50,
        )
        store.save_source(
            user_id='alice',
            url='https://a.example/3',
            title='C',
            summary='c',
            collected_at=now - 10,
        )
        store.mark_delivered([s2.id], delivered_at=now)

        undelivered = store.list_undelivered('alice', max_items=10)
        urls = [s.url for s in undelivered]
        assert urls == ['https://a.example/1', 'https://a.example/3']
        # Watermark is delivered_at IS NULL — not a time window.
        # Even a very old undelivered source remains visible.
        assert all(s.delivered_at is None for s in undelivered)

    def test_time_window_would_be_wrong_but_watermark_keeps_old(self, store):
        old = time.time() - 7 * 24 * 3600
        store.save_source(
            user_id='alice',
            url='https://old.example/kept',
            title='Old',
            summary='missed run',
            collected_at=old,
        )
        undelivered = store.list_undelivered('alice')
        assert len(undelivered) == 1
        assert undelivered[0].url == 'https://old.example/kept'

    def test_mark_delivered_sets_watermark(self, store):
        s, _ = store.save_source(
            user_id='alice',
            url='https://x.example/1',
            title='X',
            summary='x',
        )
        assert s.delivered_at is None
        n = store.mark_delivered([s.id])
        assert n == 1
        loaded = store.get_source(s.id)
        assert loaded is not None
        assert loaded.delivered_at is not None
        assert store.list_undelivered('alice') == []
