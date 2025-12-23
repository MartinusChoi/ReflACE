from typing import Dict, Any, List, Union
import json

from .base import BaseAgent
from ..env.appworld_env import AppWorldEnv
from ..core.trajectory import Trajectory
from ..core.messages import (
    UserMessage,
    AIMessage,
    ToolCallOutputMessage,
    ToolMessageList,
    ChatMessageList
)
from ..llm.openai_client import OpenAIClient
from ..utils.conditions import is_agent_finished

# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates
# -------------------------------------------------------------------------------------
PROMPT = {
    'react_only' : """
Using these APIs, now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

Task: {instruction}
""",
    'with_reflection' : """
Using these 'APIs' and 'reflection history' of your previous actions, now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

Task: {instruction}

Reflection History:
{reflection_history}
""",
    'with_playbook' : """
""",
    'with_reflection_and_playbook' : """
"""
}


# -------------------------------------------------------------------------------------
# ReAct Agent Class
# -------------------------------------------------------------------------------------
class ReActAgent(BaseAgent):
    """
    ReAct Agent Class
    """
    def __init__(
        self, 
        actor_client : Union[OpenAIClient, Any]
    ):
        super().__init__(
            actor_client=actor_client
        )
    
    def _build_prompt(
        self,
        env: AppWorldEnv,
        reflection_history: str = None
    )-> str:
        """
        build input prompt for ReAct Agent.

        Args:
            env (AppWorldEnv): 
                Appworld environment object.
            reflection_history (str, optional): 
                Reflection history of previous actions. Defaults to None.
                Use this when you use ReAct Agent for Actor Module in Reflexion Agent.

        Returns:
            str: 
                Input prompt for ReAct Agent. 
                If reflection_history is not None, the prompt will include the reflection history.
        """
        # get task instruction and supervisor information
        instruction = env.get_instruction()           # instruction
        supervisor_info = env.get_supervisor_info()   # supervisor information

        if reflection_history is None:
            # build prompt for ReAct Agent without reflection history
            return PROMPT['react_only'].format(**{
                'first_name' : supervisor_info['first_name'],
                'last_name' : supervisor_info['last_name'],
                'email' : supervisor_info['email'],
                'phone_number' : supervisor_info['phone_number'],
                'instruction' : instruction
            })
        
        else:
            # build prompt for ReAct Agent with reflection history : In Reflexion Agent setting
            return PROMPT['with_reflection'].format(**{
                'first_name' : supervisor_info['first_name'],
                'last_name' : supervisor_info['last_name'],
                'email' : supervisor_info['email'],
                'phone_number' : supervisor_info['phone_number'],
                'instruction' : instruction,
                'reflection_history' : reflection_history
            })

    def run(
        self,
        env: AppWorldEnv,
        max_steps: int = 30,
        reflection_history: str = None,
    ) -> Dict[str, Any]:
        """
        Run ReAct Agent for max_steps.

        Args:
            env (AppWorldEnv): 
                Appworld environment object.
            max_steps (int, optional): 
                Maximum number of steps to run ReAct Agent. Defaults to 30.
            reflection_history (str, optional): 
                Reflection history of previous actions. Defaults to None.
                Use this when you use ReAct Agent for Actor Module in Reflexion Agent.

        Returns:
            Dict[str, Any]: 
                Dictionary containing the following keys:
                - 'finished': bool
                - 'trajectory': Trajectory
        """

        if reflection_history is None:
            trajectory = Trajectory([
                UserMessage(content=self._build_prompt(env=env))
            ])
        else:
            trajectory = Trajectory([
                UserMessage(content=self._build_prompt(
                    env=env,
                    reflection_history=reflection_history
                ))
            ])

        # Run ReAct Agent for max_steps
        for step in range(max_steps):

            # get response from llm core with current trajectory
            # trajectory : List[UserMessage, AIMessage, ToolCallMessage,ToolCallOutputMessage]
            response = self.actor_client.get_response(trajectory.to_chat_prompt())

            # if response is list of AIMessages, stop agent workloop
            if isinstance(response, ChatMessageList):
                for message in response.messages:
                    # add each message to trajectory
                    trajectory.append(message)
                break
            
            # if response is list of ToolCallMessage, act on environment and add Observation in trajectory
            elif isinstance(response, ToolMessageList):
                for message in response.messages:
                    # add each message to trajectory
                    trajectory.append(message)

                    # extract code from message
                    # this code act on environment
                    code = json.loads(message.arguments)['code']

                    # Act to environment and get Observation
                    obs = env.action(code)

                    # Add Observation to trajectory
                    trajectory.append(
                        ToolCallOutputMessage(
                            msg_type='function_call_output',
                            call_id=message.call_id,
                            output=obs
                        )
                    )
            else:
                # raise error if response is not list of AIMessages or ToolCallMessage
                raise ValueError(f"Unknown response type: {type(response)}")

        return {
            'finished' : is_agent_finished(trajectory),
            'trajectory' : trajectory,
        }