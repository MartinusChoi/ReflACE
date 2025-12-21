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
    )-> str:
        instruction = env.get_instruction()
        supervisor_info = env.get_supervisor_info()

        return f"""Using these APIs, now generate code to solve the actual task:

My name is: {supervisor_info['first_name']} {supervisor_info['last_name']}. My personal email is { supervisor_info['email'] } and phone number is { supervisor_info['phone_number'] }.
Task: { instruction }
"""

    def run(
        self,
        trajectory: Trajectory,
        env: AppWorldEnv,
        max_steps: int = 30
    ) -> Dict[str, Any]:

        response = self.llm.get_response(trajectory.to_context())

        if isinstance(response, ChatMessageList):
            for message in response.messages:
                trajectory.append(message)
        elif isinstance(response, ToolMessageList):
            for message in response.messages:
                trajectory.append(message)

                code = json.loads(message.arguments)['code']

                obs = env.action(code)

                trajectory.append(
                    ToolCallOutputMessage(
                        msg_type='function_call_output',
                        call_id=message.call_id,
                        output=obs
                    )
                )
        else:
            raise ValueError(f"Unknown response type: {type(response)}")

        return trajectory