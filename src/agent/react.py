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
from ..prompt.react.input_prompt import (
    react_agent,
    reflexion_actor
)


# -------------------------------------------------------------------------------------
# ReAct Agent Class
# -------------------------------------------------------------------------------------
class ReActAgent(BaseAgent):
    """
    ReAct Agent Class
    """
    def __init__(
        self, 
        actor_client : Union[OpenAIClient, Any],
    ):
        super().__init__(
            actor_client=actor_client
        )
    
    def _build_prompt(
        self,
        env_wrapper: AppWorldEnv,
        reflection_history: str = None,
        action_history: str = None
    )-> str:
        """
        build input prompt for ReAct Agent.

        Args:
            env_wrapper (AppWorldEnv): 
                Appworld environment wrapper object.
            reflection_history (str, optional): 
                Reflection history of previous actions. Defaults to None.
                Use this when you use ReAct Agent for Actor Module in Reflexion Agent.

        Returns:
            str: 
                Input prompt for ReAct Agent. 
                If reflection_history is not None, the prompt will include the reflection history.
        """
        # get task instruction and supervisor information
        instruction = env_wrapper.get_instruction()           # instruction
        supervisor_info = env_wrapper.get_supervisor_info()   # supervisor information

        # build prompt for ReAct Agent
        if reflection_history is None and action_history is None:
            # build prompt for ReAct Agent without reflection history
            return react_agent.template.format(
                first_name = supervisor_info['first_name'],         # supervisor information
                last_name = supervisor_info['last_name'],           # supervisor information
                email = supervisor_info['email'],                   # supervisor information
                phone_number = supervisor_info['phone_number'],     # supervisor information
                instruction = instruction                           # task instruction
            )
        
        # build prompt for ReAct Core of Reflexion's Actor Module
        else:
            # build prompt for ReAct Agent with reflection history : In Reflexion Agent setting
            return reflexion_actor.template.format(
                first_name = supervisor_info['first_name'],         # supervisor information
                last_name = supervisor_info['last_name'],           # supervisor information
                email = supervisor_info['email'],                   # supervisor information
                phone_number = supervisor_info['phone_number'],     # supervisor information
                instruction = instruction,                          # task instruction
                reflection_history = reflection_history             # added in Reflexion Agent setting
            )

    def run(
        self,
        env_wrapper: AppWorldEnv,
        max_steps: int = 15,
        reflection_history: str = None,
    ) -> Dict[str, Any]:
        """
        Run ReAct Agent for max_steps.

        Args:
            env_wrapper (AppWorldEnv): 
                Appworld environment wrapper object.
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
        # react only prompt
        if reflection_history is None:
            prompt = self._build_prompt(env_wrapper=env_wrapper)
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])

        # reflexion actor prompt
        else:
            prompt = self._build_prompt(env_wrapper=env_wrapper, reflection_history=reflection_history)
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])

        # Run ReAct Agent for max_steps
        for step in range(max_steps):
            # get response from llm core with current trajectory
            # trajectory : List[UserMessage, AIMessage, ToolCallMessage,ToolCallOutputMessage]
            response = self.actor_client.get_response(cur_trajectory.to_chat_prompt())

            # if response is list of AIMessages, stop agent workloop
            if isinstance(response, ChatMessageList):
                for message in response.messages:
                    # add each message to trajectory
                    cur_trajectory.append(message)
                break
            
            # if response is list of ToolCallMessage, act on environment and add Observation in trajectory
            elif isinstance(response, ToolMessageList):
                for message in response.messages:
                    # add each message to trajectory
                    cur_trajectory.append(message)

                    # extract code from message
                    # this code act on environment
                    code = json.loads(message.arguments)['code']

                    # Act to environment and get Observation
                    obs = env_wrapper.action(code)

                    # Add Observation to trajectory
                    cur_trajectory.append(
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
            'trajectory' : cur_trajectory,
        }