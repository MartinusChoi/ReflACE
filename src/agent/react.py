from typing import Dict, Any, List
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

class ReActAgent(BaseAgent):
    """
    Standard ReAct Agent.
    """
    def __init__(
        self, 
        llm_client : OpenAIClient
    ):
        super().__init__(llm_client)
        
        # ACE playbook to be injected if needed
        self.playbook = "" 
    
    def _build_prompt(
        self,
        env: AppWorldEnv,
        reflection_history: List[str] = None
    )-> str:
        instruction = env.get_instruction()
        supervisor_info = env.get_supervisor_info()

        if reflection_history is None:

            return f"""
Using these APIs, now generate code to solve the actual task:

My name is {supervisor_info['first_name']} {supervisor_info['last_name']}. 
My personal email is { supervisor_info['email'] } and phone number is { supervisor_info['phone_number'] }.

Task: {instruction}
"""
        
        else:
            reflection_book = "\n".join(reflection_history)
            return f"""
Using these 'APIs' and 'reflection history' of your previous actions, now generate code to solve the actual task:

My name is {supervisor_info['first_name']} {supervisor_info['last_name']}. 
My personal email is { supervisor_info['email'] } and phone number is { supervisor_info['phone_number'] }.

Task: {instruction}

Reflection History:
{reflection_book}
"""

    def run(
        self,
        env: AppWorldEnv,
        max_steps: int = 30,
        reflection_history: List[str] = None,
    ) -> Dict[str, Any]:

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

        # Run for max_steps
        for step in range(max_steps):

            # get response from llm core with current trajectory
            # trajectory : List[UserMessage, AIMessage, ToolCallMessage,ToolCallOutputMessage]
            response = self.llm.get_response(trajectory.to_context())

            # if response is list of AIMessages, stop agent workloop
            if isinstance(response, ChatMessageList):
                for message in response.messages:
                    # add each message to trajectory
                    trajectory.append(message)
                
                return {
                    'finished' : is_agent_finished(trajectory),
                    'trajectory' : trajectory,
                    'step' : step+1
                }
            
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
            'step' : max_steps
        }