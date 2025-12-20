from typing import Dict, Any, List, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv


class ReflectorAgent(BaseAgent):
    def __init__(self, llm_client: OpenAIClient):
        super().__init__(llm_client)
        self.llm = llm_client
    
    def _build_prompt(
        self,
        task:str,
        action:str,
        observation:str,
        reflection_history,
    )-> str:

        return f"""주어진 정보에 따라, Actor Agent가 작성한 Python Code에 대해 Reflection을 수행하시오:

Task: {task}
Actoin(Python Code): {action}
Observation: {observation}
Reflection History: {reflection_history}
"""

    
    def run(
        self,
        task:str,
        action:str,
        observation:str,
        reflection_history,
        max_steps: int =20
    ):
        prompt = self._build_prompt(task, action, observation, reflection_history)
        trajectory = Trajectory([UserMessage(content=prompt)])

        for _ in range(max_steps):
            response = self.llm.get_response(trajectory.to_context())

            if isinstance(response, ChatMessageList):
                for message in response.messages:
                    trajectory.append(message)
                break
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
        
        if isinstance(trajectory.messages[-1], AIMessage):
            return {
                'success': True,
                'reflection' : trajectory.messages[-1].content
            }
        else:
            return {
                'success' : False,
                'reflection': "reflection failed. you should analyze the reason of failure and try again."
            }
        
        

# -------------------------------------------------------------------------------------
# Main Reflexion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent that orchestrates a ReActAgent with a Reflection loop using LangGraph.
    """

    def __init__(self, llm_client: OpenAIClient):
        super().__init__(llm_client)
        # We instantiate ReActAgent on demand or keep a reference if lightweight.
        # Since ReActAgent holds no persistent state between runs (it initializes fresh trajectory),
        # we can instantiate it once.
        self.actor_agent = ReActAgent(llm_client)
        self.reflector_agent = ReflectorAgent(llm_client)
    
    def reset_agent(self):
        self.trajectory = [] # Short memory list of 'Action' and 'Observation'
        self.reflection_history = [] # Long memory list of 'Reflection'
        self.trial_num = 0
    
    def run(
        self,
        env: AppWorldEnv,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        
        pass