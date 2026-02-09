import copy
import datetime
import io
import os
import pickle
import traceback
from concurrent.futures import TimeoutError
from contextlib import redirect_stdout
from functools import partial
from typing import Any, Dict, List, Optional, Union

import json5
import regex
from tqdm import tqdm

from cat_agent.log import logger
from cat_agent.tools.base import BaseTool
from cat_agent.utils.utils import extract_code

# Patterns that should never be allowed in executed code.
# This is a best-effort blocklist -- *not* a sandbox.
_DANGEROUS_PATTERNS = [
    r'(?:\s|^|;)input\s*\(',          # interactive input
    r'(?:\s|^|;)os\.system\s*\(',      # shell via os.system
    r'(?:\s|^|;)subprocess\b',         # subprocess module
    r'(?:\s|^|;)shutil\.rmtree\s*\(',  # recursive delete
    r'__import__\s*\(',                # dynamic imports
    r'(?:\s|^|;)exec\s*\(',           # nested exec
    r'(?:\s|^|;)eval\s*\(',           # nested eval
    r'importlib\.import_module\s*\(',  # dynamic imports via importlib
]

_DANGEROUS_RE = regex.compile('|'.join(_DANGEROUS_PATTERNS))


class GenericRuntime:
    GLOBAL_DICT: Dict[str, Any] = {}
    LOCAL_DICT: Optional[Dict[str, Any]] = None
    HEADERS: List[str] = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if _DANGEROUS_RE.search(code_piece):
            raise RuntimeError(
                'Code contains a blocked pattern. '
                'Disallowed constructs: input(), os.system(), subprocess, '
                'shutil.rmtree(), __import__(), exec(), eval(), importlib.import_module()'
            )
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars['answer']


class DateRuntime(GenericRuntime):
    try:
        import dateutil.relativedelta as _dateutil_rel
        GLOBAL_DICT = {
            'datetime': datetime.datetime,
            'timedelta': _dateutil_rel.relativedelta,
            'relativedelta': _dateutil_rel.relativedelta,
        }
    except ImportError:
        GLOBAL_DICT = {
            'datetime': datetime.datetime,
        }


class CustomDict(dict):

    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


def _check_deps_for_python_executor():
    """Verify that optional heavy dependencies are installed."""
    missing = []
    for mod in ('dateutil.relativedelta', 'multiprocess', 'pebble', 'timeout_decorator'):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f'Missing dependencies for Python Executor: {", ".join(missing)}. '
            'Please install them by running: pip install "qwen-agent[python_executor]"'
        )


# @register_tool('python_executor')  # Do not register this tool by default because it is dangerous.
class PythonExecutor(BaseTool):
    name = 'python_executor'
    description = 'For executing python code. Not sandboxed. Do not use it for production purposes.'
    parameters = {
        'type': 'object',
        'properties': {
            'code': {
                'description': 'The python code.',
                'type': 'string',
            }
        },
        'required': ['code'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        _check_deps_for_python_executor()
        super().__init__(cfg)

        runtime: Optional[Any] = self.cfg.get('runtime', None)
        get_answer_symbol: Optional[str] = self.cfg.get('get_answer_symbol', None)
        get_answer_expr: Optional[str] = self.cfg.get('get_answer_expr', None)
        get_answer_from_stdout: bool = self.cfg.get('get_answer_from_stdout', True)
        timeout_length: int = self.cfg.get('timeout_length', 20)

        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length

    def call(self, params: Union[str, dict], **kwargs) -> list:
        try:
            params = json5.loads(params)
            code = params['code']
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return ['', '']

        predictions = self.apply(code)
        return predictions

    def apply(self, code: str) -> list:
        return self.batch_apply([code])[0]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout=None,
        runtime=None,
        answer_symbol=None,
        answer_expr=None,
        timeout_length=20,
    ):
        from timeout_decorator import timeout as timeout_decorator
        try:
            if get_answer_from_stdout:
                program_io = io.StringIO()
                try:
                    with redirect_stdout(program_io):
                        timeout_decorator(timeout_length)(runtime.exec_code)(code)
                    program_io.seek(0)
                    result = program_io.read()
                finally:
                    program_io.close()
            elif answer_symbol:
                timeout_decorator(timeout_length)(runtime.exec_code)(code)
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout_decorator(timeout_length)(runtime.exec_code)(code)
                result = timeout_decorator(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                # Execute all lines except the last, then evaluate the final line
                lines = code.rsplit('\n', 1)
                if len(lines) == 2 and lines[0].strip():
                    timeout_decorator(timeout_length)(runtime.exec_code)(lines[0])
                result = timeout_decorator(timeout_length)(runtime.eval_code)(lines[-1])
            report = 'Done'
            str(result)
            pickle.dumps(result)  # serialization check
        except Exception:
            result = ''
            # Capture the meaningful part of the traceback
            tb = traceback.format_exc().strip()
            tb_lines = [line for line in tb.split('\n') if line.strip()]
            report = tb_lines[-1] if tb_lines else 'Unknown error'
        return result, report

    @staticmethod
    def truncate(s, max_length=800):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + f'... [{len(s) - max_length} chars truncated] ...' + s[-half:]
        return s

    def batch_apply(self, batch_code: List[str]) -> list:
        from pebble import ProcessPool

        timeout_cnt = 0
        all_exec_results = []
        max_workers = min(len(batch_code), os.cpu_count() or 1)

        with ProcessPool(max_workers=max_workers) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length,
            )
            future = pool.map(executor, batch_code, timeout=self.timeout_length)
            iterator = future.result()

            if len(batch_code) > 100:
                progress_bar = tqdm(total=len(batch_code), desc='Execute')
            else:
                progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    logger.warning('PythonExecutor: execution timed out: {}', error)
                    all_exec_results.append(('', 'Timeout Error'))
                    timeout_cnt += 1
                except Exception as error:
                    logger.opt(exception=True).error('PythonExecutor: unexpected error during execution: {}', error)
                    all_exec_results.append(('', f'Execution Error: {error}'))
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

        if timeout_cnt:
            logger.info('PythonExecutor: {}/{} executions timed out', timeout_cnt, len(batch_code))

        batch_results = []
        for code, (res, report) in zip(batch_code, all_exec_results):
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results


def _test():
    batch_code = ["""
        print("Hello world!")
        """]

    executor = PythonExecutor(cfg={'get_answer_from_stdout': True})
    predictions = executor.apply(batch_code[0])
    print(predictions)


if __name__ == '__main__':
    _test()
