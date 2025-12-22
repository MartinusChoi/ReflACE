from typing import Dict, Any, List, Literal
import operator

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv
from ..utils.conditions import is_agent_finished
from ..core.trajectory import Trajectory
from ..core.messages import UserMessage, ChatMessageList

# -------------------------------------------------------------------------------------
# Main Reflexion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent that orchestrates a ReActAgent with a Reflection loop
    """

    def __init__(
        self, 
        action_module_client: OpenAIClient,
        reflection_module_client: OpenAIClient
    ):
        self.action_module = ReActAgent(action_module_client)
        self.reflection_module_client = reflection_module_client
    
    def _build_reflection_prompt(
        self, 
        env:AppWorldEnv,
        trajectory:Trajectory,
        reflection_history:List[str]
    ) -> str:
        # make reflection history into a string(book)
        reflection_book = "\n\n".join(reflection_history)

        return f"""
Task Status: Task Failed. 
Objective: Generate a reflection on the Actor Agent's Action History by analyzing the Python code and execution logs.

Requirements:
1. Pinpoint the exact failure points and logic errors in the code history.
2. Specify clear, actionable improvements for the next attempt.
3. Constraint: Do not use pronouns (e.g., 'it', 'this', 'that'). Refer to variables, functions, and logic by their specific names.
4. Constraint: Use brief, direct, and non-abstract sentences.

Task: {env.get_instruction()}
Action History: {trajectory.to_context()}
Reflection History: {reflection_book}
"""

    def _reflection_module(
        self,
        env:AppWorldEnv,
        trajectory:Trajectory,
        reflection_history:List[str] = [],
    ) -> List[str]:

        # create initial reflection module input
        reflection_request = Trajectory([
            UserMessage(content=self._build_reflection_prompt(env, trajectory, reflection_history))
        ])

        # get response from reflection module llm core with current trajectory
        response = self.reflection_module_client.get_response(reflection_request.to_context())

        if isinstance(response, ChatMessageList):
            # concatenate all current reflection contents
            reflection = "\n\n".join([msg.content for msg in response.messages])
            reflection_history.append(reflection)
            return reflection_history
        else:
            # raise error if response is not list of AIMessages
            # in Reflection Module, we expect only AIMessage not ToolCallMessage
            raise ValueError(f"Unknown response type: {type(response)}")
        
    
    def _evaluation_module(self, env: AppWorldEnv) -> bool:
        # evaluate agent task results
        evaluation = env.env.evaluate()
        # return True if task is success, False otherwise
        return evaluation.success
    
    def run(
        self,
        env: AppWorldEnv,
        max_retries: int = 3
    ) -> Dict[str, Any]:

        reflection_history = []

        for _ in range(max_retries):
            # actor action, get trajectory of actor action
            action_result = self.action_module.run(
                env=env,
                max_steps=30,
                reflection_history=reflection_history
            )

            action_trajectory = "\n".join([str(ctx) for ctx in action_result['trajectory'].to_context()])
            reflection_history.append(action_trajectory)

            # evaluate action of actor module
            evaluation = self._evaluation_module(env)

            # if task success, done reflexion loop
            if evaluation: break

            # reflect on actor action
            reflection_history = self._reflection_module(
                env=env,
                trajectory=action_result['trajectory'],
                reflection_history=reflection_history
            )
        
        return {
            'finished' : is_agent_finished(action_result['trajectory']),
            'trajectory' : action_result['trajectory'],
            'reflection_history' : reflection_history
        }