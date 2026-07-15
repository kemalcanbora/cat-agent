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

"""Web search for on-prem deployments (SearxNG by default; Serper opt-in legacy)."""

import os
from typing import Any, List, Union

import requests

from cat_agent.security.offline import guard_outbound_request
from cat_agent.tools.base import BaseTool, register_tool

DEFAULT_SEARXNG_PATH = '/search'


@register_tool(
    'web_search',
    allow_overwrite=True,
    requires_network=True,
    cloud_service=False,
    register_by_default=False,
)
class WebSearch(BaseTool):
    name = 'web_search'
    description = (
        'Search for information using a self-hosted SearxNG instance '
        '(set CAT_AGENT_SEARXNG_URL). Not enabled by default; call '
        'cat_agent.tools.enable_optional_tools("web_search") first.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
            }
        },
        'required': ['query'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params['query']
        search_results = self.search(query)
        return self._format_results(search_results)

    @staticmethod
    def _backend() -> str:
        return os.getenv('CAT_AGENT_WEB_SEARCH_BACKEND', os.getenv('WEB_SEARCH_BACKEND', 'searxng')).lower()

    @classmethod
    def search(cls, query: str) -> List[Any]:
        backend = cls._backend()
        if backend == 'serper':
            return cls._search_serper(query)
        if backend == 'searxng':
            return cls._search_searxng(query)
        raise ValueError(
            f'Unsupported web search backend: {backend}. '
            'Use "searxng" (default, on-prem) or "serper" (legacy cloud).'
        )

    @staticmethod
    def _search_searxng(query: str) -> List[Any]:
        base_url = os.getenv('CAT_AGENT_SEARXNG_URL', os.getenv('SEARXNG_URL', '')).rstrip('/')
        if not base_url:
            raise ValueError(
                'CAT_AGENT_SEARXNG_URL is not set. Point it at your self-hosted SearxNG instance, '
                'e.g. export CAT_AGENT_SEARXNG_URL=http://searxng.internal:8080'
            )
        guard_outbound_request(purpose=f'SearxNG search at {base_url}')
        response = requests.get(
            f'{base_url}{DEFAULT_SEARXNG_PATH}',
            params={'q': query, 'format': 'json'},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get('results', []):
            results.append({
                'title': item.get('title', ''),
                'snippet': item.get('content', '') or item.get('snippet', ''),
                'date': item.get('publishedDate', ''),
            })
        return results

    @staticmethod
    def _search_serper(query: str) -> List[Any]:
        api_key = os.getenv('SERPER_API_KEY', '')
        serper_url = os.getenv('SERPER_URL', 'https://google.serper.dev/search')
        if not api_key:
            raise ValueError(
                'SERPER_API_KEY is not set. Serper is a cloud service and is not recommended for '
                'air-gapped deployments. Use SearxNG (default) or another on-prem index.'
            )
        guard_outbound_request(purpose=f'Serper cloud search at {serper_url}')
        headers = {'Content-Type': 'application/json', 'X-API-KEY': api_key}
        response = requests.post(serper_url, json={'q': query}, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get('organic', [])

    @staticmethod
    def _format_results(search_results: List[Any]) -> str:
        content = '```\n{}\n```'.format('\n\n'.join([
            f"[{i}]\"{doc['title']}\n{doc.get('snippet', '')}\"{doc.get('date', '')}"
            for i, doc in enumerate(search_results, 1)
        ]))
        return content
