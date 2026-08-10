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

import ast
import os
from typing import List, Literal

from cat_agent.env import load_env_file

load_env_file()

# Settings for LLMs
DEFAULT_MAX_INPUT_TOKENS: int = int(os.getenv(
    'CAT_AGENT_DEFAULT_MAX_INPUT_TOKENS', 58000))  # The LLM will truncate the input messages if they exceed this limit
# Fraction of local prompt-token estimate below which we warn about silent server-side truncation
# (e.g. Ollama num_ctx). 0.95 = 5% tokenizer tolerance.
PROMPT_TRUNCATION_TOLERANCE: float = float(os.getenv(
    'CAT_AGENT_PROMPT_TRUNCATION_TOLERANCE', '0.95'))

# Settings for agents
MAX_LLM_CALL_PER_RUN: int = int(os.getenv('CAT_AGENT_MAX_LLM_CALL_PER_RUN', 20))

# Settings for tools
DEFAULT_WORKSPACE: str = os.getenv('CAT_AGENT_DEFAULT_WORKSPACE', 'workspace')

# Settings for RAG
DEFAULT_MAX_REF_TOKEN: int = int(os.getenv('CAT_AGENT_DEFAULT_MAX_REF_TOKEN',
                                           20000))  # The window size reserved for RAG materials
DEFAULT_PARSER_PAGE_SIZE: int = int(os.getenv('CAT_AGENT_DEFAULT_PARSER_PAGE_SIZE',
                                              500))  # Max tokens per chunk when doing RAG
DEFAULT_RAG_KEYGEN_STRATEGY: Literal['None', 'GenKeyword', 'SplitQueryThenGenKeyword', 'GenKeywordWithKnowledge',
                                     'SplitQueryThenGenKeywordWithKnowledge'] = os.getenv(
                                         'CAT_AGENT_DEFAULT_RAG_KEYGEN_STRATEGY', 'GenKeyword')
DEFAULT_RAG_SEARCHERS: List[str] = ast.literal_eval(
    os.getenv('CAT_AGENT_DEFAULT_RAG_SEARCHERS',
              "['keyword_search', 'front_page_search']"))  # Sub-searchers for hybrid retrieval


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


# Tool synthesis mutation gate (example-set quality check after holdout)
MUTATION_ENABLED: bool = _env_bool('CAT_AGENT_MUTATION_ENABLED', True)
MUTATION_LIMIT: int = int(os.getenv('CAT_AGENT_MUTATION_LIMIT', '12'))
MUTATION_THRESHOLD: float = float(os.getenv('CAT_AGENT_MUTATION_THRESHOLD', '0.8'))

# Scheduled source collection / report delivery
# Empty → JobStore defaults to sqlite:///workspace/scheduling/scheduling.sqlite
SCHEDULER_DSN: str = os.getenv('CAT_AGENT_SCHEDULER_DSN', '')
SCHEDULER_MAX_JOBS_PER_USER: int = int(os.getenv('CAT_AGENT_SCHEDULER_MAX_JOBS_PER_USER', '20'))
SCHEDULER_LEASE_SECONDS: int = int(os.getenv('CAT_AGENT_LEASE_SECONDS', '900'))
SCHEDULER_JOB_LIMIT: int = int(os.getenv('CAT_AGENT_JOB_LIMIT', '50'))
SCHEDULER_MAX_REPORT_ITEMS: int = int(os.getenv('CAT_AGENT_SCHEDULER_MAX_REPORT_ITEMS', '50'))
SCHEDULER_BACKOFF_CAP_MULTIPLIER: int = int(
    os.getenv('CAT_AGENT_SCHEDULER_BACKOFF_CAP_MULTIPLIER', '8'))

# On-demand FastAPI agent serve (optional [serve] extra)
SERVE_HOST: str = os.getenv('CAT_AGENT_SERVE_HOST', '127.0.0.1')
SERVE_PORT: int = int(os.getenv('CAT_AGENT_SERVE_PORT', '8080'))
SERVE_TOKEN: str = os.getenv('CAT_AGENT_SERVE_TOKEN', '')
SERVE_MAX_CONCURRENCY: int = int(os.getenv('CAT_AGENT_SERVE_MAX_CONCURRENCY', '1'))
# Max waiters blocked on the per-agent semaphore (0 = reject immediately when busy)
SERVE_MAX_QUEUE: int = int(os.getenv('CAT_AGENT_SERVE_MAX_QUEUE', '8'))
SERVE_RETRY_AFTER_SECONDS: int = int(os.getenv('CAT_AGENT_SERVE_RETRY_AFTER_SECONDS', '1'))
# uvicorn timeout_graceful_shutdown (seconds). MUST be strictly less than the
# orchestrator kill timeout (Nomad kill_timeout / Docker stop -t / k8s
# terminationGracePeriodSeconds). If SIGKILL wins the race, lifespan teardown
# never finishes and agent.aclose() does not run — LLM clients leak connections.
SERVE_SHUTDOWN_TIMEOUT: int = int(os.getenv('CAT_AGENT_SERVE_SHUTDOWN_TIMEOUT', '45'))
# When true, /run error bodies include redacted str(exc). Server logs always
# keep full detail regardless of this flag.
SERVE_VERBOSE_ERRORS: bool = _env_bool('CAT_AGENT_SERVE_VERBOSE_ERRORS', False)
# Inline async jobs (POST /agents/{name}/jobs)
SERVE_JOB_MAX: int = int(os.getenv('CAT_AGENT_SERVE_JOB_MAX', '256'))
SERVE_JOB_TTL_SECONDS: float = float(os.getenv('CAT_AGENT_SERVE_JOB_TTL_SECONDS', '600'))
# Nomad dispatch payload hard cap (Nomad itself rejects above 16KiB)
SERVE_DISPATCH_MAX_PAYLOAD_BYTES: int = int(
    os.getenv('CAT_AGENT_SERVE_DISPATCH_MAX_PAYLOAD_BYTES', str(16 * 1024))
)
