"""Tests for cat_agent.utils.parallel_executor."""

from cat_agent.utils.parallel_executor import parallel_exec, serial_exec


class TestParallelExecutor:

    def test_serial_exec(self):
        def add(a, b):
            return a + b

        results = serial_exec(add, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert results == [3, 7]

    def test_parallel_exec(self):
        def double(x):
            return x * 2

        results = parallel_exec(double, [{"x": 1}, {"x": 2}, {"x": 3}])
        assert sorted(results) == [2, 4, 6]
