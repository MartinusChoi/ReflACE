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
# Reflector Prompt
# -------------------------------------------------------------------------------------
PROMPT = """
Task Status: Task Failed. 
Objective: Generate a reflection on the Actor Agent's Action and Reflector's Reflection History by analyzing the Python code, execution logs, and reflection of those actions.

Requirements:
1. Pinpoint the exact failure points and logic errors in the code history.
2. Specify clear, actionable improvements for the next attempt.
3. Constraint: Do not use pronouns (e.g., 'it', 'this', 'that'). Refer to variables, functions, and logic by their specific names.
4. Constraint: Use brief, direct, and non-abstract sentences.

**Task**: 
{instruction}

**trajectory**: 
{trajectory}

**reflection history**: 
{reflection_history}
"""




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
        max_action_history:int = 5,
        max_reflection_history:int = 5
    ):
        super().__init__(
            actor_client=actor_client
        )
        self.actor = ReActAgent(self.actor_client)
        self.reflector_client = reflector_client
        self.reflection_history = []
        self.action_history = []
        self.max_action_history = max_action_history
        self.max_reflection_history = max_reflection_history
        self.reflection_cnt = 1
        self.action_cnt = 1
    
    def reset(
        self, 
        reset_reflection_history:bool = False,
        reset_action_history:bool = True
    ):
        if reset_reflection_history:
            self.reflection_history = []
        if reset_action_history:
            self.action_history = []
    
    def _build_reflect_prompt(
        self, 
        env:AppWorldEnv,
        trajectory:Trajectory,
    ) -> str:

        return PROMPT.format(
            instruction=env.get_instruction(),
            trajectory="".join(self.action_history),
            reflection_history="".join(self.reflection_history)
        )

    def _reflector(
        self,
        env:AppWorldEnv,
        trajectory:Trajectory,
    ) -> None:

        # create initial reflection module input
        reflection_request = Trajectory([
            UserMessage(content=self._build_reflect_prompt(env, trajectory))
        ])

        # get response from reflection module llm core with current trajectory
        response = self.reflector_client.get_response(reflection_request.to_chat_prompt())

        if isinstance(response, ChatMessageList):

            # concatenate all action history to reflection history
            if len(self.action_history) > self.max_action_history: self.action_history.pop(0)
            self.action_history.append(f"<action history {self.action_cnt}>\n")
            for chat_prompt in trajectory.to_chat_prompt():
                self.action_history[-1] += f"{str(chat_prompt)}\n"
            self.action_history[-1] += f"</action history {self.action_cnt}>\n\n"
            self.action_cnt += 1

            # concatenate all current reflection contents
            if len(self.reflection_history) > self.max_reflection_history: self.reflection_history.pop(0)
            self.reflection_history.append(f"<reflection history {self.reflection_cnt}>\n")
            for msg in response.messages:
                self.reflection_history[-1] += f"{msg.content}\n"
            self.reflection_history[-1] += f"</reflection history {self.reflection_cnt}>\n\n"
            self.reflection_cnt += 1

        else:
            # raise error if response is not list of AIMessages
            # in Reflection Module, we expect only AIMessage not ToolCallMessage
            raise ValueError(f"Unknown response type: {type(response)}")
        
    
    def _evaluator(
        self,
        env: AppWorldEnv
    ) -> bool:
        # evaluate agent task results
        evaluation = env.env.evaluate()
        # return True if task is success, False otherwise
        return evaluation.success
    
    def run(
        self,
        env: AppWorldEnv,
        max_steps: int = 3
    ) -> Dict[str, Any]:

        for _ in range(max_steps):
            # actor action, get trajectory of actor action
            action = self.actor.run(
                env=env,
                max_steps=30,
                reflection_history="".join(self.reflection_history),
                action_history="".join(self.action_history)
            )

            # evaluate action of actor module
            is_success = self._evaluator(env)
            
            if is_success: break # if task success, done reflexion loop

            # reflect on actor action
            self._reflector(
                env=env,
                trajectory=action['trajectory']
            )
        
        return {
            'finished' : is_agent_finished(action['trajectory']),
            'trajectory' : action['trajectory'],
            'reflection_history' : "".join(self.reflection_history),
            'action_history' : "".join(self.action_history)
        }