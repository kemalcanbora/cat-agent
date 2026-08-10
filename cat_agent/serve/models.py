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

"""Pydantic request/response models for the serve API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageIn(BaseModel):
    """One chat message accepted by ``POST /agents/{name}/run``."""

    model_config = ConfigDict(extra='allow')

    role: str
    content: Any = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class RunRequest(BaseModel):
    messages: List[MessageIn] = Field(..., min_length=1)
    stream: bool = False
    run_timeout: Optional[float] = Field(
        default=None,
        description='Optional wall-clock deadline in seconds (async path only).',
        gt=0,
    )


class AgentInfoOut(BaseModel):
    name: str
    description: str
    agent_class: str
    max_concurrency: int


class RunResponse(BaseModel):
    agent: str
    messages: List[Dict[str, Any]]
    content: Optional[str] = None


class JobCreateRequest(BaseModel):
    messages: List[MessageIn] = Field(..., min_length=1)
    run_timeout: Optional[float] = Field(
        default=None,
        description='Optional wall-clock deadline in seconds.',
        gt=0,
    )


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    agent: str
    state: str
    created_at: float
    updated_at: float
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
