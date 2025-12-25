from typing import Dict, Any, List, Union, Optional, Literal
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
from ..prompt.react.input_prompt import react_input_prompt
from ..prompt.reflexion.input_prompt import reflexion_actor_input_prompt


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
        agent_type: Literal[
            'react', 
            'reflexion', 
            'ace', 
            'reflace',
        ],
        env_wrapper: AppWorldEnv,
        **kwargs
    )-> str:
        """
        build input prompt for ReAct Agent.

        Args:
            agent_type (Literal): 
                Type of ReAct Agent.
                Use this to determine which prompt to use.
            env_wrapper (AppWorldEnv): 
                Appworld environment wrapper object.
            **kwargs: 
                Additional keyword arguments.
                - trajectory: Optional[Trajectory] = None
                - reflection_history: Optional[str] = None
                - playbook: Optional[str] = None

        Returns:
            str: 
                Input prompt for certain ReAct Agent type.
        """
        # get task instruction and supervisor information
        instruction = env_wrapper.get_instruction()           # instruction
        supervisor_info = env_wrapper.get_supervisor_info()   # supervisor information

        # build prompt for ReAct Agent
        if agent_type == 'react':
            # build prompt for ReAct Agent without reflection history
            return react_input_prompt.template.format(
                first_name = supervisor_info['first_name'],         # supervisor information
                last_name = supervisor_info['last_name'],           # supervisor information
                email = supervisor_info['email'],                   # supervisor information
                phone_number = supervisor_info['phone_number'],     # supervisor information
                instruction = instruction                           # task instruction
            )
        
        # build prompt for ReAct Core of Reflexion's Actor Module
        elif agent_type == 'reflexion':
            # build prompt for ReAct Agent with reflection history : In Reflexion Agent setting
            return reflexion_actor_input_prompt.template.format(
                first_name = supervisor_info['first_name'],                       # supervisor information
                last_name = supervisor_info['last_name'],                         # supervisor information
                email = supervisor_info['email'],                                 # supervisor information
                phone_number = supervisor_info['phone_number'],                   # supervisor information
                instruction = instruction,                                        # task instruction
                reflection_history = kwargs.get('reflection_history')             # added in Reflexion Actor Module setting
            )
        
        # build prompt for ReAct Core of ACE's Generator Module
        elif agent_type == 'ace':
            # build prompt for ReAct Agent with playbook : In ACE's Generator Module setting
            raise NotImplementedError("ReflACE's Generator Module is not implemented yet")
        
        # build prompt for ReAct Core of ACE's Reflector Module
        elif agent_type == 'reflace':
            # build prompt for ReAct Agent with playbook : In ACE's Reflector Module setting
            raise NotImplementedError("ReflACE's Reflector Module is not implemented yet")
        
        else:
            raise NotImplementedError("ReflACE's Actor Module is not implemented yet")

    def run(
        self,
        agent_type: Literal[
            'react', 
            'reflexion', 
            'ace', 
            'reflace',
        ],
        env_wrapper: AppWorldEnv,
        max_steps: int = 25,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run ReAct Agent for max_steps.

        Args:
            agent_type (Literal): 
                Type of ReAct Agent.
                Use this to determine which prompt to use.
            env_wrapper (AppWorldEnv): 
                Appworld environment wrapper object.
            max_steps (int, optional): 
                Maximum number of steps to run ReAct Agent. Defaults to 30.
            **kwargs (optional): 
                Keyword arguments.
                - reflection_history (Optional[str], optional): 
                    Reflection history of previous actions. Defaults to None.
                    Use this when you use ReAct Agent for Actor Module in Reflexion Agent.
                - playbook (Optional[str], optional): 
                    Playbook of previous actions. Defaults to None.
                    Use this when you use ReAct Agent for Generator Module in ACE Agent.

        Returns:
            Dict[str, Any]: 
                Dictionary containing the following keys:
                - 'trajectory': Trajectory
        """

        # react only prompt
        if agent_type == 'react':
            prompt = self._build_prompt(
                agent_type=agent_type,
                env_wrapper=env_wrapper
            )
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])

        # reflexion actor prompt
        elif agent_type == 'reflexion':
            prompt = self._build_prompt(
                agent_type=agent_type,
                env_wrapper=env_wrapper,
                reflection_history=kwargs['reflection_history']
            )
            cur_trajectory = Trajectory(messages=[UserMessage(content=prompt)])

        # ace generator prompt
        elif agent_type == 'ace':
            raise NotImplementedError("Generator in ACE is not implemented yet")

        # ace reflector prompt
        elif agent_type == 'reflace':
            raise NotImplementedError("Actor in ReflACE is not implemented yet")

        else:
            raise ValueError("Invalid agent type")

        # Run ReAct Agent for max_steps
        for step in range(max_steps):
            # get response from llm core with current trajectory
            # trajectory : List[UserMessage, AIMessage, ToolCallMessage,ToolCallOutputMessage]
            response_messages = self.actor_client.get_response(cur_trajectory.to_chat_prompt())

            for message in response_messages:
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
                        print(f"    📍 Actor Agent Done on step {step}")
                        return {'trajectory' : cur_trajectory}

                elif isinstance(message, AIMessage):
                    cur_trajectory.append(message)
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
        
        print(f"    📍 Actor Done on step {max_steps}")
        return {'trajectory' : cur_trajectory}