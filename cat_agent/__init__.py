__version__ = "0.11.5"

from cat_agent.env import load_env_file

load_env_file()

from cat_agent.security.offline import install_offline_guards

install_offline_guards()

from .agent import Agent
from .multi_agent_hub import MultiAgentHub

__all__ = [
    'Agent',
    'MultiAgentHub',
]
