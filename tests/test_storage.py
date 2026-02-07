"""Tests for cat_agent.tools.storage."""

import tempfile

import pytest

from cat_agent.tools.storage import KeyNotExistsError, Storage, _norm_key


class TestStorage:

    @pytest.fixture
    def storage_path(self):
        d = tempfile.mkdtemp()
        yield d

    @pytest.fixture
    def storage(self, storage_path):
        return Storage({"storage_root_path": storage_path})

    def test_norm_key(self):
        assert _norm_key("a") == "a"
        assert _norm_key("/a") == "a"
        assert _norm_key("/x/y") == "x/y"

    def test_put_get(self, storage):
        storage.call({"operate": "put", "key": "k1", "value": "v1"})
        out = storage.call({"operate": "get", "key": "k1"})
        assert out == "v1"

    def test_get_missing_raises(self, storage):
        with pytest.raises(KeyNotExistsError, match="does not exist"):
            storage.call({"operate": "get", "key": "nonexistent"})

    def test_delete(self, storage):
        storage.call({"operate": "put", "key": "k2", "value": "v2"})
        out = storage.call({"operate": "delete", "key": "k2"})
        assert "Successfully deleted" in out
        with pytest.raises(KeyNotExistsError):
            storage.call({"operate": "get", "key": "k2"})

    def test_delete_missing_returns_message(self, storage):
        out = storage.call({"operate": "delete", "key": "nonexistent"})
        assert "Delete Failed" in out

    def test_scan_empty_returns_fail_message(self, storage):
        out = storage.call({"operate": "scan", "key": "/"})
        assert "Scan Failed" in out or "does not exist" in out

    def test_scan_returns_keys_under_prefix(self, storage):
        storage.call({"operate": "put", "key": "a/1", "value": "v1"})
        storage.call({"operate": "put", "key": "a/2", "value": "v2"})
        out = storage.call({"operate": "scan", "key": "a"})
        assert "v1" in out and "v2" in out
