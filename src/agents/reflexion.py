from .base import BaseAgent
from .react import ReActAgent
from ..prompt.reflexion.system_prompt import (
    ACTOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT
)
from ..state import State

from appworld import AppWorld
from typing import Any

from langchain_openai import ChatOpenAI

from langgraph.graph.state import CompiledStateGraph





class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent class
    """
    def __init__(
        self,
        env: AppWorld,
        actor_system_prompt: str = ACTOR_SYSTEM_PROMPT,
        reflector_system_prompt: str = REFLECTOR_SYSTEM_PROMPT,
        model_config: dict[str, Any] = {
            'model' : 'gpt-4o',
            'temperature' : 0.0,
            'stream_usage' : True
        }
    ):
        self.env = env
        self.actor_system_prompt = actor_system_prompt
        self.reflector_system_prompt: str = reflector_system_prompt

        self.openai_client = ChatOpenAI(**model_config)
        self.openai_client_with_tools = self.openai_client.bind_tools(self._get_tools())

    def _get_actor_node(self):
        actor = ReActAgent(
            env=self.env,
            system_prompt=ACTOR_SYSTEM_PROMPT,
            model_config=self.model_config
        )

        def _actor(state: State):
            return actor.invoke(state)
        
        return _actor
    
    def _get_evaluator_node(self):

        def _evaluator(state: State):
            # get task evaluation result
            eval_result = self.env.evaluate()

            evaluation_report = ""

            # get task requirement information
            # total requirement count
            total_requirement_count = eval_result.total_count
            passed_requirement_count = eval_result.pass_count
            failed_requirement_count = eval_result.fail_count

            evaluation_report += f"Task Status : {"Succeed" if self.env.task_completed else "Failed"}\n\n"

            evaluation_report += f"Task Requirement Count:\n- total requirements : {total_requirement_count}\n- passed requirements : {passed_requirement_count}\n- failed requirements : {failed_requirement_count}\n\n"

            if passed_requirement_count > 0:
                evaluation_report += "Detail of passed requirments: \n"
                for passed_requirement in eval_result.passes:
                    evaluation_report += str({
                        'requirement' : passed_requirement['requirement'],
                        'label' : passed_requirement['label']
                    })
                    evaluation_report += "\n"
                evaluation_report += "\n"


            if failed_requirement_count > 0:
                evaluation_report += "Detail of failed requirments: \n"
                for failed_requirement in eval_result.failures:
                    evaluation_report += str({
                        'requirement' : failed_requirement['requirement'],
                        'failed_reason' : failed_requirement['trace']
                    })
                    evaluation_report += "\n"
                evaluation_report += "\n"
            
            print(f"[Evaluator] 📊 Evaluation Report: \n{evaluation_report}")
            
            return {
                'evaluation' : evaluation_report
            }
        

        return _evaluator
            




    def _get_reflector_node(self):
        pass

    def _build_agent(self):
        # --------------------------------------------
        # Get Nodes
        # --------------------------------------------
        _actor = self._get_actor_node()
        _evaluator = self._get_evaluator_node()
        

    
