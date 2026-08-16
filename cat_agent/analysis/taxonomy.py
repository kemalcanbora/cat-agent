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

"""MAST failure-mode taxonomy (Cemri et al., NeurIPS 2025).

Source: Cemri, M. et al. (2025). *Why Do Multi-Agent LLM Systems Fail?*
arXiv:2503.13657. Definitions below follow Appendix A of the paper verbatim
in structure and wording. Tier labels (deterministic vs judge) are our
operationalisation for cat-agent traces, not the paper's method.

Categories (paper):
  1. System Design Issues
  2. Inter-Agent Misalignment
  3. Task Verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

Tier = Literal['deterministic', 'judge']

CategoryId = Literal['1', '2', '3']

CATEGORIES: Dict[CategoryId, str] = {
    '1': 'System Design Issues',
    '2': 'Inter-Agent Misalignment',
    '3': 'Task Verification',
}

PAPER_CITATION = (
    'Cemri, M. et al. (2025). Why Do Multi-Agent LLM Systems Fail? '
    'arXiv:2503.13657, NeurIPS 2025.'
)


@dataclass(frozen=True)
class FailureMode:
    id: str
    category: CategoryId
    name: str
    definition: str
    tier: Tier
    tier_rationale: str
    signals: Tuple[str, ...] = field(default_factory=tuple)


# Definitions adapted from Appendix A of arXiv:2503.13657 (NeurIPS 2025 PDF).
MAST_MODES: Tuple[FailureMode, ...] = (
    FailureMode(
        id='1.1',
        category='1',
        name='Disobey Task Specification',
        definition=(
            'Failure to adhere to the specified constraints or requirements of a '
            'given task, leading to suboptimal or incorrect outcomes.'
        ),
        tier='judge',
        tier_rationale='Requires semantic comparison of final output to the task brief.',
        signals=('final_output contradicts initial user task', 'ignored explicit constraints'),
    ),
    FailureMode(
        id='1.2',
        category='1',
        name='Disobey Role Specification',
        definition=(
            'Failure to adhere to the defined responsibilities and constraints of '
            'an assigned role, potentially leading to an agent behaving like another.'
        ),
        tier='judge',
        tier_rationale='Role boundaries are prompt-semantic; needs judge over agent names/roles.',
        signals=('agent performs another agent\'s duties', 'violates declared role'),
    ),
    FailureMode(
        id='1.3',
        category='1',
        name='Step Repetition',
        definition=(
            'Unnecessary reiteration of previously completed steps in a process, '
            'potentially causing delays or errors in task completion.'
        ),
        tier='deterministic',
        tier_rationale='Exact/near-duplicate tool calls and LLM inputs are hashable in traces.',
        signals=('repeated tool_name+arguments', 'near-identical llm_call message hashes'),
    ),
    FailureMode(
        id='1.4',
        category='1',
        name='Loss of Conversation History',
        definition=(
            'Unexpected context truncation, disregarding recent interaction history '
            'and reverting to an antecedent conversational state.'
        ),
        tier='deterministic',
        tier_rationale=(
            'We detect the checkable operationalisation: after a context_op eviction, '
            'a later step re-requests information present in an evicted message.'
        ),
        signals=('context_op then re-request of evicted content',),
    ),
    FailureMode(
        id='1.5',
        category='1',
        name='Unaware of Termination Conditions',
        definition=(
            'Lack of recognition or understanding of the criteria that should trigger '
            'the termination of the agents\' interaction, potentially leading to '
            'unnecessary continuation.'
        ),
        tier='deterministic',
        tier_rationale='Trace status/termination_reason and missing final_output are explicit.',
        signals=('status=terminated with max_steps|max_tokens|wall_clock and no final answer',),
    ),
    FailureMode(
        id='2.1',
        category='2',
        name='Conversation Reset',
        definition=(
            'Unexpected or unwarranted restarting of a dialogue, potentially losing '
            'context and progress made in the interaction.'
        ),
        tier='judge',
        tier_rationale='Hard to distinguish intentional restarts from failures without semantics.',
        signals=('dialogue restarts mid-run', 'agents re-introduce themselves'),
    ),
    FailureMode(
        id='2.2',
        category='2',
        name='Fail to Ask for Clarification',
        definition=(
            'Inability to request additional information when faced with unclear or '
            'incomplete data, potentially resulting in incorrect actions.'
        ),
        tier='judge',
        tier_rationale='Requires judging ambiguity vs assumption-taking.',
        signals=('proceeds despite ambiguous requirements',),
    ),
    FailureMode(
        id='2.3',
        category='2',
        name='Task Derailment',
        definition=(
            'Deviation from the intended objective or focus of a given task, '
            'potentially resulting in irrelevant or unproductive actions.'
        ),
        tier='judge',
        tier_rationale='Semantic drift detection.',
        signals=('off-topic tool use', 'answers a different question'),
    ),
    FailureMode(
        id='2.4',
        category='2',
        name='Information Withholding',
        definition=(
            'Failure to share or communicate important data or insights that an agent '
            'possess and could impact decision-making of other agents if shared.'
        ),
        tier='judge',
        tier_rationale='Requires multi-agent message comparison.',
        signals=('agent knows fact never shared',),
    ),
    FailureMode(
        id='2.5',
        category='2',
        name="Ignored Other Agent's Input",
        definition=(
            'Disregarding or failing to adequately consider input or recommendations '
            'provided by other agents in the system, potentially leading to suboptimal '
            'decisions or missed opportunities for collaboration.'
        ),
        tier='judge',
        tier_rationale='Needs cross-agent utterance alignment.',
        signals=('advice ignored in subsequent actions',),
    ),
    FailureMode(
        id='2.6',
        category='2',
        name='Reasoning-Action Mismatch',
        definition=(
            'Discrepancy between the logical reasoning process and the actual actions '
            'taken by the agent, potentially resulting in unexpected or undesired behaviors.'
        ),
        tier='judge',
        tier_rationale='Compares stated plan to tool_call arguments.',
        signals=('says X then calls tool for Y',),
    ),
    FailureMode(
        id='3.1',
        category='3',
        name='Premature Termination',
        definition=(
            'Ending a dialogue, interaction or task before all necessary information '
            'has been exchanged or objectives have been met, potentially resulting in '
            'incomplete or incorrect outcomes.'
        ),
        tier='judge',
        tier_rationale='Completeness of outcome is semantic.',
        signals=('status=completed but task incomplete',),
    ),
    FailureMode(
        id='3.2',
        category='3',
        name='No or Incomplete Verification',
        definition=(
            '(Partial) omission of proper checking or confirmation of task outcomes or '
            'system outputs, potentially allowing errors or inconsistencies to propagate '
            'undetected.'
        ),
        tier='judge',
        tier_rationale='Verification steps are domain-specific.',
        signals=('no review/test step before finish',),
    ),
    FailureMode(
        id='3.3',
        category='3',
        name='Incorrect Verification',
        definition=(
            'Failure to adequately validate or cross-check crucial information or '
            'decisions during the iterations, potentially leading to errors or '
            'vulnerabilities in the system.'
        ),
        tier='judge',
        tier_rationale='False-positive verification is semantic.',
        signals=('claims verified but evidence contradicts',),
    ),
)


MODES_BY_ID: Dict[str, FailureMode] = {m.id: m for m in MAST_MODES}


def modes_for_tier(tier: Tier) -> List[FailureMode]:
    return [m for m in MAST_MODES if m.tier == tier]
