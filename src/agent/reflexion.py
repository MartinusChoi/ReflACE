from typing import Dict, Any, List, Literal

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv
from ..core.trajectory import Trajectory
from ..core.messages import UserMessage, ChatMessageList
from ..prompt.reflexion.input_prompt import reflexion_reflector
from ..core.reflection import ReflectionHistory


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
        self.reflector_client = reflector_client
        self.reflection_history = ReflectionHistory(max_size=3)
    
    def _build_reflect_prompt(
        self, 
        env_wrapper:AppWorldEnv,
        trajectory:Trajectory,
    ) -> str:
        return reflexion_reflector.template.format(
            instruction=env_wrapper.get_instruction(),
            trajectory=trajectory.to_str(),
            reflection_history=self.reflection_history.get_history()
        )

    def _reflector(
        self,
        env_wrapper:AppWorldEnv,
        trajectory:Trajectory,
    ) -> None:

        # create initial reflection module input
        reflection_request = Trajectory([
            UserMessage(content=self._build_reflect_prompt(env_wrapper, trajectory))
        ])

        # get response from reflection module llm core with current trajectory
        response = self.reflector_client.get_response(reflection_request.to_chat_prompt())

        if isinstance(response, ChatMessageList):
            # add current reflection to reflection history
            # ReflectionHistory maintain max size of history automatically
            self.reflection_history.add_reflection(messages=response.messages)

        else:
            # raise error if response is not list of AIMessages
            # in Reflection Module, we expect only AIMessage not ToolCallMessage
            raise ValueError(f"Unknown response type: {type(response)}")
        
    
    def _evaluator(
        self,
        env_wrapper: AppWorldEnv
    ) -> bool:
        # evaluate agent task results
        evaluation = env_wrapper.env.evaluate()
        # return True if task is success, False otherwise
        return evaluation.success
    
    def run(
        self,
        env_wrapper: AppWorldEnv,
        max_steps: int = 5
    ) -> Dict[str, Any]:

        for _ in range(max_steps):
            # actor action, get trajectory of actor action
            action = self._actor.run(
                env_wrapper=env_wrapper,
                max_steps=15,
                reflection_history=self.reflection_history.get_history(),
            )

            # evaluate action of actor module
            is_success = self._evaluator(env_wrapper)
            
            if is_success: break # if task success, done reflexion loop

            # reflect on actor action
            self._reflector(
                env_wrapper=env_wrapper,
                trajectory=action['trajectory']
            )
        
        return {
            'trajectory' : action['trajectory'],
            'reflection_history' : self.reflection_history.get_history(),
        }