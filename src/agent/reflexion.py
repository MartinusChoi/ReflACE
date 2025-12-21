from typing import Dict, Any, List, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv


# -------------------------------------------------------------------------------------
# Reflection Module
# -------------------------------------------------------------------------------------
class ReflectionModule(BaseAgent):
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

        return f"""With given information, Conduct a detailed reflection on the Actor Agent's Python code. Evaluate its correctness and suggest specific improvements for the next iteration.:

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
        reflection_history:List[str],
    ) -> Dict[str, Any]:

        prompt = self._build_prompt(task, action, observation, reflection_history)
        trajectory = Trajectory([UserMessage(content=prompt)])

        response = self.llm.get_response(trajectory.to_context())

        if isinstance(response, ChatMessageList):
            reflection = "\n\n".join([msg.content for msg in response.messages])
            return {
                'reflection': reflection,
                'trajectory': trajectory
            }
        else:
            raise ValueError(f"Unknown response type: {type(response)}")


class EvaluatorModule:
    """
    EvaluatorModule은 
    """
    def __init__(
        self,
        llm_client: OpenAIClient,
    ):
        self.llm = llm_client

    def run(
        self,
        trajectory: Trajectory,
        is_actor_finished: bool,
    ) -> Dict[str, Any]:

        if is_actor_finished:
            return """
            현재 Actor Agent가 주어진 Task를 주어진 시도 횟수 내에 완료하지 못했습니다.
            Actor Agent가 현재 Task를 완료하기 위해 생성한 Python Code와 그 결과에 대한 리스트를 기반으로 다음 시도에서 Actor가 Python Code를 작성할 때 참고할 수 있는 성찰/비평문을 작성하세요.
            """



        
        

# -------------------------------------------------------------------------------------
# Main Reflexion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent that orchestrates a ReActAgent with a Reflection loop
    """

    def __init__(self, llm_client: OpenAIClient):
        super().__init__(llm_client)

        self.actor_agent = ReActAgent(llm_client)
        self.reflector_agent = ReflectorModule(llm_client)
    
    def run(
        self,
        env: AppWorldEnv,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        
        # actor action
        actor_action = self.actor_agent.run(env, max_retries=30)
