from cat_agent.agent import Agent, BasicAgent
from cat_agent.multi_agent_hub import MultiAgentHub

from .assistant import Assistant
# DocQAAgent is the default solution for long document question answering.
# The actual implementation of DocQAAgent may change with every release.
from .doc_qa import BasicDocQA as DocQAAgent
from .doc_qa import ParallelDocQA
from .fncall_agent import FnCallAgent
from .group_chat import GroupChat
from .group_chat_auto_router import GroupChatAutoRouter
from .react_chat import ReActChat
from .router import Router
from .user_agent import UserAgent
from .virtual_memory_agent import VirtualMemoryAgent

__all__ = [
    'Agent',
    'BasicAgent',
    'MultiAgentHub',
    'DocQAAgent',
    'ParallelDocQA',
    'Assistant',
    'ReActChat',
    'Router',
    'UserAgent',
    'GroupChat',
    'GroupChatAutoRouter',
    'FnCallAgent',
    'VirtualMemoryAgent',
]
