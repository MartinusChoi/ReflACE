from typing import Dict, Any, List, Union
import json

from .base import BaseAgent
from ..env.appworld_env import AppWorldEnv
from ..core.trajectory import Trajectory
from ..core.messages import (
    UserMessage,
    AIMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
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
        playbook: str = None,
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
        if reflection_history is None and playbook is None:
            # build prompt for ReAct Agent without reflection history
            return react_agent.template.format(
                first_name = supervisor_info['first_name'],         # supervisor information
                last_name = supervisor_info['last_name'],           # supervisor information
                email = supervisor_info['email'],                   # supervisor information
                phone_number = supervisor_info['phone_number'],     # supervisor information
                instruction = instruction                           # task instruction
            )
        
        # build prompt for ReAct Core of Reflexion's Actor Module
        elif reflection_history is not None and playbook is None:
            # build prompt for ReAct Agent with reflection history : In Reflexion Agent setting
            return reflexion_actor.template.format(
                first_name = supervisor_info['first_name'],         # supervisor information
                last_name = supervisor_info['last_name'],           # supervisor information
                email = supervisor_info['email'],                   # supervisor information
                phone_number = supervisor_info['phone_number'],     # supervisor information
                instruction = instruction,                          # task instruction
                reflection_history = reflection_history             # added in Reflexion Agent setting
            )
        
        # build prompt for ReAct Core of ACE's Generator Module
        elif reflection_history is None and playbook is not None:
            # build prompt for ReAct Agent with playbook : In ACE's Generator Module setting
            return react_agent.template.format(
                first_name = supervisor_info['first_name'],         # supervisor information
                last_name = supervisor_info['last_name'],           # supervisor information
                email = supervisor_info['email'],                   # supervisor information
                phone_number = supervisor_info['phone_number'],     # supervisor information
                instruction = instruction,                          # task instruction
                playbook = playbook                                 # added in ACE's Generator Module setting
            )
        
        else:
            raise NotImplementedError("ReflACE's Actor Module is not implemented yet")

    def run(
        self,
        env_wrapper: AppWorldEnv,
        max_steps: int = 15,
        reflection_history: str = None,
        playbook: str = None,
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
            playbook (str, optional): 
                Playbook of previous actions. Defaults to None.
                Use this when you use ReAct Agent for Generator Module in ACE Agent.

        Returns:
            Dict[str, Any]: 
                Dictionary containing the following keys:
                - 'finished': bool
                - 'trajectory': Trajectory
        """
        # react only prompt
        if reflection_history is None and playbook is None:
            prompt = self._build_prompt(env_wrapper=env_wrapper)
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])

        # reflexion actor prompt
        elif reflection_history is not None and playbook is None:
            prompt = self._build_prompt(env_wrapper=env_wrapper, reflection_history=reflection_history)
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])

        # ace generator prompt
        elif reflection_history is None and playbook is not None:
            prompt = self._build_prompt(env_wrapper=env_wrapper, playbook=playbook)
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])
        
        else:
            raise NotImplementedError("ReflACE's Actor Module is not implemented yet")

        # Run ReAct Agent for max_steps
        is_agent_finished = False
        for step in range(max_steps):
            # get response from llm core with current trajectory
            # trajectory : List[UserMessage, AIMessage, ToolCallMessage,ToolCallOutputMessage]
            response = self.actor_client.get_response(cur_trajectory.to_chat_prompt())

            for message in response:
                if isinstance(message, ToolCallMessage):
                    cur_trajectory.append(message)

                    code = json.loads(message.arguments)['code']

                    obs = env_wrapper.action(code)

                    cur_trajectory.append(
                        ToolCallOutputMessage(
                            msg_type='function_call_output',
                            call_id=message.call_id,
                            output=obs
                        )
                    )

                    if "complete_task" in code:
                        is_agent_finished = True
                        break

                elif isinstance(message, AIMessage):
                    cur_trajectory.append(message)
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
            
            if is_agent_finished: break

        return {
            'trajectory' : cur_trajectory,
        }