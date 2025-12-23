from typing import Dict, Any, List, Literal

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv
from ..core.trajectory import Trajectory
from ..core.messages import (
    UserMessage,
    AIMessage,
)
from ..prompt.reflexion.input_prompt import reflexion_reflector
from ..core.reflection import ReflectionHistory





# -------------------------------------------------------------------------------------
# Reflector Agent Class
# -------------------------------------------------------------------------------------
class ReflectorAgent(BaseAgent):
    """
    Reflector Agent that use ReActAgent for Actor Module with a Reflection loop
    """

    def __init__(
        self, 
        actor_client: OpenAIClient,
    ):
        super().__init__(actor_client=actor_client)
    
    def _build_prompt(
        self, 
        env_wrapper:AppWorldEnv,
        trajectory:Trajectory,
        reflection_history: ReflectionHistory,
    ) -> str:
        return reflexion_reflector.template.format(
            instruction=env_wrapper.get_instruction(),
            trajectory=trajectory.to_str(),
            reflection_history=reflection_history.get_history(),
            success="Success" if env_wrapper.evaluate_env().success else "Failed",
        )

    def run(
        self,
        env_wrapper:AppWorldEnv,
        trajectory:Trajectory,
        reflection_history: ReflectionHistory,
    ) -> ReflectionHistory:

        # create initial reflection module input
        reflection_request = Trajectory([
            UserMessage(content=self._build_prompt(env_wrapper, trajectory, reflection_history))
        ])

        # get response from reflection module llm core with current trajectory
        response = self.actor_client.get_response(reflection_request.to_chat_prompt())

        for message in response:
            if isinstance(message, AIMessage):
                reflection_history.add_reflection(messages=[message])
            else:
                raise ValueError(f"Unexpected message type: {type(message)}")
        
        return reflection_history








# -------------------------------------------------------------------------------------
# Reflexion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent that use ReActAgent for Actor Module with a Reflection loop
    """

    def __init__(
        self, 
        actor_client: OpenAIClient,
        reflector_client: OpenAIClient,
    ):
        super().__init__(
            actor_client=actor_client
        )
        self._actor = ReActAgent(self.actor_client)
        self._reflector = ReflectorAgent(reflector_client)
        self.reflection_history = ReflectionHistory(max_size=None)
    
    def _build_prompt(self) -> None:
        pass
    
    def _evaluator(
        self,
        env_wrapper: AppWorldEnv
    ) -> bool:
        # evaluate agent task results
        evaluation = env_wrapper.evaluate_env()
        # return True if task is success, False otherwise
        return evaluation.success
    
    def run(
        self,
        env_wrapper: AppWorldEnv,
        max_steps: int = 3
    ) -> Dict[str, Any]:

        for _ in range(max_steps):
            # actor action, get trajectory of actor action
            action = self._actor.run(
                env_wrapper=env_wrapper,
                max_steps=15,
                reflection_history=self.reflection_history.get_history(),
            )

            # evaluate action of actor module
            # if task success, done reflexion loop
            if self._evaluator(env_wrapper): break 

            # reflect on actor action
            self.reflection_history = self._reflector.run(
                env_wrapper=env_wrapper,
                trajectory=action['trajectory'],
                reflection_history=self.reflection_history,
            )
        
        return {
            'trajectory' : action['trajectory'],
            'reflection_history' : self.reflection_history.get_history(),
        }