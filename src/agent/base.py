from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from ..env.appworld_env import AppWorldEnv
from ..llm.openai_client import OpenAIClient

class BaseAgent(ABC):
    """
    Abstract base class for all agents (ReAct, Reflexion, ACE, ReflACE).
    """
    
    def __init__(
        self, 
        actor_client: Union[OpenAIClient, Any]
    ):
        self.actor_client = actor_client

    @abstractmethod
    def run(
        self,
        env: AppWorldEnv,
        max_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Run the agent on a given Appworld environment.
        
        Args:
            env: The Appworld environment to run the agent on.
            max_steps: Maximum number of steps to take.
            
        Returns:
            A dictionary containing the results (finished, trajectory)
            - finished : Boolean indicating if the agent has finished the task in given steps.
            - trajectory : The trajectory of the agent.
        """
        pass
