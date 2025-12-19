from abc import ABC, abstractmethod
from typing import Any, Dict
from ..env.appworld_env import AppWorldEnv

class BaseAgent(ABC):
    """
    Abstract base class for all agents (ReAct, Reflexion, ACE).
    """
    
    def __init__(self, llm_client: Any):
        self.llm = llm_client

    @abstractmethod
    def run(
        self, 
        env: AppWorldEnv,
        max_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Run the agent on a given environment.
        
        Args:
            env: The environment to run the agent on.
            max_steps: Maximum number of steps to take.
            
        Returns:
            A dictionary containing the results (success, trajectory)
        """
        pass
