from .base import BaseAgent
from .react import ReActAgent
from ..prompt.reflexion import (
    ACTOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    ACTOR_INPUT_PROMPT,
    REFLECTOR_INPUT_PROMPT
)
from ..state import ReActState, ReflexionState
from ..utils.llm import get_response_with_retry
from ..utils.token_usage import get_token_usage_from_message

from appworld import AppWorld
from typing import Any, Callable, Sequence

from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END


# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reflctor Module in Reflexion Agent
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
class ReflectorModule(BaseAgent):
    """
    Reflector Module in Reflexion Agent.
    """

    # -----------------------------------------------------------------------------------------------
    # Define Actor Node
    # -----------------------------------------------------------------------------------------------
    def _get_actor_node(self) -> Callable:

        # Actor Node
        # ============================================================================================================
        def _actor(state: ReActState):

            # add system message in message history
            messages: Sequence[AnyMessage] = state['messages']
            request_messages: Sequence[AnyMessage] = [SystemMessage(content=self.system_prompt)] + messages

            # get response from llm client
            response: AIMessage = get_response_with_retry(
                model_client=self.openai_client_with_tools,
                messages=request_messages,
                max_retries=3
            )

            # get token usages
            token_usage = get_token_usage_from_message(response)

            # update agent state
            return {
                'messages' : [response],
                'input_tokens' : token_usage['input_tokens'],
                'output_tokens' : token_usage['output_tokens'],
                'total_tokens' : token_usage['total_tokens'],
            }
        # ============================================================================================================
        
        return _actor
    
    # ----------------------------------------------------------------------------
    # Define Tool Node
    # ----------------------------------------------------------------------------
    def _get_tool_node(self):
        
        for _tool in self.tool_list:
            if _tool.name == 'action_tool':
                action_tool = _tool

        # Tool Node
        # =============================================================================
        def _tools(state: ReActState):
            last_msg: AIMessage = state['messages'][-1]
            tool_messages = []

            for tool_call in last_msg.tool_calls:
                if tool_call['name'] == 'action_tool':
                    try:
                        tool_message: ToolMessage = ToolMessage(
                            content=action_tool.invoke(tool_call['args']),
                            tool_call_id=tool_call['id']
                        )
                        tool_messages.append(tool_message)
                    except Exception as error:
                        raise error
                    
            return {'messages' : tool_messages}
        # =============================================================================
        
        return _tools
    
    # ----------------------------------------------------------------------------
    # Define conditional edge function
    # ----------------------------------------------------------------------------
    def _get_should_continue(self) -> Callable:
        
        # should continue
        # =============================================================================
        def _should_continue(state:ReActState):
            last_msg:AIMessage = state['messages'][-1]

            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                return 'tools'
            else:
                return 'end'
        # =============================================================================
        
        return _should_continue
    

    def _build_agent(self) -> CompiledStateGraph:
        # ----------------------------------------------------------------------------
        # Get Nodes
        # ----------------------------------------------------------------------------
        _actor = self._get_actor_node()
        _tools = self._get_tool_node()

        # ----------------------------------------------------------------------------
        # Get Conditional Edges
        # ----------------------------------------------------------------------------
        _should_continue = self._get_should_continue()

        # ----------------------------------------------------------------------------
        # Define ReAct Workflow
        # ----------------------------------------------------------------------------
        # create graph builder
        workflow = StateGraph(state_schema=ReActState)

        # add nodes
        workflow.add_node("actor", _actor)   # actor
        workflow.add_node("tools", _tools)   # tool

        # add edges
        workflow.add_edge(START, "actor")
        workflow.add_conditional_edges(
            'actor',
            _should_continue,
            {
                'end' : END,
                'tools' : 'tools'
            }
        )
        workflow.add_edge("tools", "actor")

        # compile graph and return CompiledStateGraph instance
        return workflow.compile()







# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reflexion Module in Reflexion Agent
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        self.model_config = model_config

        self.tool_list = self._get_tool_list()
        self.openai_client = ChatOpenAI(**model_config)
        self.openai_client_with_tools = self.openai_client.bind_tools(self.tool_list)

        self.agent = self._build_agent()

    # -----------------------------------------------------------------------------------------------
    # Define Actor Node
    # -----------------------------------------------------------------------------------------------
    def _get_actor_node(self) -> Callable:
        actor: CompiledStateGraph = ReActAgent(
            env=self.env,
            system_prompt=ACTOR_SYSTEM_PROMPT,
            model_config=self.model_config
        )

        # Actor node
        # ==========================================================================================
        def _actor(state: ReflexionState):

            reflection_history = ""
            for i, reflection in enumerate(state['reflections']):
                reflection_history += f"{i+1}. {reflection}\n\n"

            result: ReActState = actor.invoke({
                'messages' : [
                    HumanMessage(
                        content=ACTOR_INPUT_PROMPT.format(
                            first_name = self.env.task.supervisor.first_name,
                            last_name = self.env.task.supervisor.last_name,
                            email = self.env.task.supervisor.email,
                            phone_number = self.env.task.supervisor.phone_number,
                            instruction = self.env.task.instruction,
                            reflection_history = reflection_history
                        )
                    )
                ]
            })

            return {
                'trajectory' : result['messages'],
                'input_tokens' : result['input_tokens'],
                'output_tokens' : result['output_tokens'],
                'total_tokens' : result['total_tokens']
            }
        # ==========================================================================================
        
        return _actor
    
    # -----------------------------------------------------------------------------------------------
    # Define Evaluator Node
    # -----------------------------------------------------------------------------------------------
    def _get_evaluator_node(self) -> Callable:

        # Evaluator Node
        # ==========================================================================================
        def _evaluator(state: ReflexionState):
            # get task evaluation result
            eval_result = self.env.evaluate()

            evaluation_report = ""

            # get task requirement information
            # total requirement count
            total_requirement_count = eval_result.total_count
            passed_requirement_count = eval_result.pass_count
            failed_requirement_count = eval_result.fail_count

            evaluation_report += f"Task Status : {'Succeed' if (eval_result.pass_count == eval_result.total_count) else 'Failed'}\n---\n\n"


            evaluation_report += f"Task Requirement Count:\n- total requirements : {total_requirement_count}\n- passed requirements : {passed_requirement_count}\n- failed requirements : {failed_requirement_count}\n---\n\n"

            if passed_requirement_count > 0:
                evaluation_report += "Detail of passed requirments: \n"
                for passed_requirement in eval_result.passes:
                    evaluation_report += str({
                        'requirement' : passed_requirement['requirement'],
                        'label' : passed_requirement['label']
                    })
                    evaluation_report += "\n"
                evaluation_report += "---\n\n"


            if failed_requirement_count > 0:
                evaluation_report += "Detail of failed requirments: \n"
                for failed_requirement in eval_result.failures:
                    evaluation_report += str({
                        'requirement' : failed_requirement['requirement'],
                        'failed_reason' : failed_requirement['trace']
                    })
                    evaluation_report += "\n"
                evaluation_report += "---\n\n"
            
            return {'evaluation' : evaluation_report}
        # ==========================================================================================
        

        return _evaluator
    

    # -----------------------------------------------------------------------------------------------
    # Define Reflector Node
    # -----------------------------------------------------------------------------------------------
    def _get_reflector_node(self) -> Callable:
        
        reflector = ReflectorModule(
            env=self.env,
            system_prompt=self.reflector_system_prompt,
            model_config=self.model_config
        )

        # Refelctor Node
        # ==========================================================================================
        def _reflector(state: ReflexionState):
            result: ReActState = reflector.invoke({
                'messages' : [
                    HumanMessage(
                        content = REFLECTOR_INPUT_PROMPT.format(
                            first_name = self.env.task.supervisor.first_name,
                            last_name = self.env.task.supervisor.last_name,
                            email = self.env.task.supervisor.email,
                            phone_number = self.env.task.supervisor.phone_number,
                            instruction = self.env.task.instruction,
                            evaluation_report = state['evaluation'],
                            reflection_history = state['reflections'],
                            trajectory = state['trajectory']
                        )
                    )
                ]
            })

            return {
                'reflections' : [result['messages'][-1].content],
                'input_tokens' : result['input_tokens'],
                'output_tokens' : result['output_tokens'],
                'total_tokens' : result['total_tokens']
            }
        # ==========================================================================================

        return _reflector
    
    # -----------------------------------------------------------------------------------------------
    # Define conditional edge (should continue reflexion loop)
    # -----------------------------------------------------------------------------------------------
    def _get_should_continue(self) -> Callable:

        # should continue
        # ==========================================================================================
        def _should_continue(state: ReflexionState):
            
            max_retries = 3
            if len(state['reflections']) == max_retries:
                return 'end'
            
            elif "Succeed" in state['evaluation']:
                return 'end'
            
            return 'reflector'
        # ==========================================================================================
        
        return _should_continue
    
    # -----------------------------------------------------------------------------------------------
    # Define Reflexion Workflow
    # -----------------------------------------------------------------------------------------------
    def _build_agent(self) -> CompiledStateGraph:
        # --------------------------------------------
        # Get Nodes
        # --------------------------------------------
        _actor = self._get_actor_node()
        _evaluator = self._get_evaluator_node()
        _reflector = self._get_reflector_node()

        # --------------------------------------------
        # Get conditional edge
        # --------------------------------------------
        _should_continue = self._get_should_continue()

        # --------------------------------------------
        # Define reflexion workflow
        # --------------------------------------------
        workflow = StateGraph(ReflexionState)

        # add node
        workflow.add_node("actor", _actor)
        workflow.add_node("evaluator", _evaluator)
        workflow.add_node("reflector", _reflector)

        # add edge
        workflow.add_edge(START, "actor")
        workflow.add_edge("actor", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            _should_continue,
            {
                "reflector" : "reflector",
                "end" : END
            }
        )
        workflow.add_edge("reflector", "actor")

        # compile graph
        return workflow.compile()

    
