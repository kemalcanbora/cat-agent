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

"""Pluggable conversation-context management (not RAG).

``cat_agent.memory.Memory`` retrieves documents. This package decides what
stays in the model's context window during a long agent run.

References: Lindenbauer et al. arXiv:2508.21433; Sun et al. arXiv:2510.11967;
Mei et al. arXiv:2507.13334; Hu et al. arXiv:2603.07670.
"""

from cat_agent.context.budget import ContextBudget, ContextOverflowError, ContextResult
from cat_agent.context.manager import ContextManager, default_context_manager, get_default_context_manager
from cat_agent.context.residue import ResidueRegistry, generic_residue_extractor
from cat_agent.context.strategies.compaction import SummaryCompactionStrategy
from cat_agent.context.strategies.folding import ContextFoldingStrategy
from cat_agent.context.strategies.masking import ObservationMaskingStrategy

__all__ = [
    'ContextBudget',
    'ContextFoldingStrategy',
    'ContextManager',
    'ContextOverflowError',
    'ContextResult',
    'ObservationMaskingStrategy',
    'ResidueRegistry',
    'SummaryCompactionStrategy',
    'default_context_manager',
    'generic_residue_extractor',
    'get_default_context_manager',
]
