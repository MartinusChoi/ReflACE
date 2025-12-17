from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class BaseAgent(ABC):
    """
    Abstract base class for all agents (ReAct, Reflexion, ACE).
    """
    
    def __init__(self, llm_client: Any, env: Any):
        self.llm = llm_client
        self.env = env
        self.history: List[Dict[str, str]] = []

    @abstractmethod
    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Run the agent on a given task.
        
        Args:
            task: The task description string.
            max_steps: Maximum number of steps to take.
            
        Returns:
            A dictionary containing the results (success, log, etc.)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent state."""
        self.history = []
