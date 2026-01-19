from .base import BaseAgent
from .react import ReActAgent
from ..state import ReActState, ACEState
from ..utils.llm import get_response_with_retry
from ..utils.token_usage import get_token_usage_from_message
from ..prompt.ace import (
    # generator prompts
    GENERATOR_INPUT_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_RESPONSE_MODULE_INPUT_PROMPT,
    GENERATOR_RESPONSE_MODULE_SYSTEM_PROMPT,
    # reflector prompts
    REFLECTOR_INPUT_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    # curator prompts
    CURATOR_INPUT_PROMPT,
    CURATOR_SYSTEM_PROMPT
)

from typing import Callable, Sequence, Dict, Any

from langchain_openai import ChatOpenAI

from langchain.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END

from appworld import AppWorld



# --------------------------------------------------------------------------------------------------------
# Define Reflector Module in ACE Agent (ReAct Pattern)
# --------------------------------------------------------------------------------------------------------
class ReflectorModule(BaseAgent):
    """
    Reflector Module in ACE (Agentic Context Engineering) framework.
    """
    def __init__(
        self,
        env: AppWorld,
        system_prompt: str = REFLECTOR_SYSTEM_PROMPT,
        model_config: Dict[str, Any] = {
            'model' : 'gpt-4o',
            'temperature' : 0.0,
            'stream_usage' : True
        }
    ) -> None:
        self.env = env
        self.system_prompt = system_prompt
        self.model_config = model_config

        self.tool_list = self._get_tool_list()

        self.openai_client = ChatOpenAI(**model_config)
        self.openai_client_with_tools = self.openai_client.bind_tools(self.tool_list)
        self.openai_client_with_structured_output = self.openai_client.with_structured_output(
            ########## NotImplement ##########
        )

    # --------------------------------------------------------------------------------------------------------
    # Define Actor Node
    # --------------------------------------------------------------------------------------------------------
    def _get_actor_node(self) -> Callable:
        """
        Create actor node in ReAct Pattern.

        Return:
            _actor [Callable[ReActState]]
        """

        # Actor Node
        # ================================================================================================================
        def _actor(state: ReActState) -> ReActState:

            request_messages: Sequence[AnyMessage] = [SystemMessage(content=self.system_prompt)] + state['messages']

            response: AIMessage = get_response_with_retry(
                model_client=self.openai_client_with_tools,
                messages=request_messages,
                max_retries=3
            )

            token_usage = get_token_usage_from_message(response)
            
            return {
                'messages' : [response],
                'input_tokens' : token_usage['input_tokens'],
                'output_tokens' : token_usage['output_tokens'],
                'total_tokens' : token_usage['total_tokens']
            }
        # ================================================================================================================
        
        return _actor
    
    # --------------------------------------------------------------------------------------------------------
    # Define Ressponse Node
    # --------------------------------------------------------------------------------------------------------
    def _get_response_node(self) -> Callable:
        """
        Create response node that convert response into structured output.

        Return:
            _response [Callable[ReActState]]
        """

        # Response Node
        # ================================================================================================================
        def _response(state: ReActState) -> ReActState:

            request_messages: Sequence[AnyMessage] = [SystemMessage(content=GENERATOR_RESPONSE_MODULE_SYSTEM_PROMPT)] + [
                HumanMessage(content=GENERATOR_RESPONSE_MODULE_INPUT_PROMPT.format(
                    ############# NotImplemented #############
                ))
            ]

            response: AIMessage = get_response_with_retry(
                model_client=self.openai_client_with_structured_output,
                messages = request_messages,
                max_retries=3
            )

            token_usage = get_token_usage_from_message(response)
            
            return {
                'messages' : [response],
                'input_tokens' : token_usage['input_tokens'],
                'output_tokens' : token_usage['output_tokens'],
                'total_tokens' : token_usage['total_tokens']
            }
        # ================================================================================================================
        
        return _response

    # --------------------------------------------------------------------------------------------------------
    # Define Tool Node
    # --------------------------------------------------------------------------------------------------------
    def _get_tool_node(self) -> Callable:
        """
        Create tool node in ReAct Pattern.

        Return:
            _tools [Callable[ReActState]]
        """

        for _tool in self.tool_list:
            if _tool.name == 'action_tool':
                action_tool = _tool
        
        # Tool node
        # ================================================================================================================
        def _tools(state: ReActState):

            last_msg: AIMessage = state['messages'][-1]
            
            tool_messages: Sequence[ToolMessage] = []

            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:

                for tool_call in last_msg.tool_calls:
                    if tool_call['name'] == 'action_tool':
                        try:
                            tool_messages.append(
                                ToolMessage(
                                    content=action_tool.invoke(tool_call['args']),
                                    tool_call_id=tool_call['id']
                                )
                            )
                        except Exception as error:
                            raise error
                
            return {'messages' : tool_messages}
        # ================================================================================================================

        return _tools
    

    # --------------------------------------------------------------------------------------------------------
    # Define Conditional Edge Function
    # --------------------------------------------------------------------------------------------------------
    def _get_should_continue(self) -> Callable:
        """
        Create Should Continue function in ReAct Pattern.

        Return:
            _should_continue [Callable[ReActState]]
        """

        # should continue
        # ================================================================================================================
        def _should_continue(state: ReActState) -> str:

            last_msg: AIMessage = state['messages']
            
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                return 'tools'
            else:
                return 'response'
        # ================================================================================================================

        return _should_continue
    

    # --------------------------------------------------------------------------------------------------------
    # Build Graphd
    # --------------------------------------------------------------------------------------------------------
    def _build_agent(self) -> CompiledStateGraph:
        
        # ----------------------------------------------------
        # Get Nodes
        # ----------------------------------------------------
        _actor = self._get_actor_node()
        _tools = self._get_tool_node()
        _response = self._get_response_node()

        # ----------------------------------------------------
        # Get Conditional Edge function
        # ----------------------------------------------------
        _should_continue = self._get_should_continue()

        # ----------------------------------------------------
        # Define ReAct Workflow
        # ----------------------------------------------------
        workflow = StateGraph(ReActState)

        # add nodes
        workflow.add_node('actor', _actor)
        workflow.add_node('tools', _tools)
        workflow.add_node('response', _response)
        
        # add edges
        workflow.add_edge(START, 'actor')
        workflow.add_conditional_edges(
            'actor',
            _should_continue,
            {
                'tools' : 'tools',
                'response' : 'response'
            }
        )
        workflow.add_edge('tools', 'actor')
        workflow.add_edge('response', END)

        # compile workflow
        return workflow.compile()
    

























# --------------------------------------------------------------------------------------------------------
# Define ACEAgent that follows ACE (Agentic Context Engineering) framework.
# --------------------------------------------------------------------------------------------------------
class ACEAgent(BaseAgent):
    """
    Agent Class follows ACE (Agentic Context Engineering) framework.
    """
    def __init__(
        self,
        env: AppWorld,
        generator_system_prompt: str = GENERATOR_SYSTEM_PROMPT,
        reflector_system_prompt: str = REFLECTOR_SYSTEM_PROMPT,
        curator_system_prompt: str = CURATOR_SYSTEM_PROMPT,
        model_config: Dict[str, Any] = {
            'model' : 'gpt-4o',
            'temperature' : 0.0,
            'stream_usage' : True
        }
    ) -> None:
        
        self.env = env
        self.generator_system_prompt: str = generator_system_prompt
        self.reflector_system_prompt: str = reflector_system_prompt
        self.curator_system_prompt: str = curator_system_prompt
        self.model_config = model_config

        self.tool_list: Sequence[tool] = self._get_tool_list()

        self.openai_client = ChatOpenAI(**model_config)
        self.openai_client_with_tools = self.openai_client.bind_tools(self.tool_list)

        self.agent = self._build_agent()

    
    # --------------------------------------------------------------------------------------------------------
    # Define Generator
    # --------------------------------------------------------------------------------------------------------
    def _get_generator_node(self) -> Callable:

        generator = ReActAgent(
            env=self.env,
            system_prompt=self.generator_system_prompt,
            model_config=self.model_config
        )

        # Generator Module
        # ================================================================================================================
        def _generator(state: ACEState) -> ACEState:

            try:
                result_state: ACEState = generator.invoke({
                    'messages' : [HumanMessage(content=GENERATOR_INPUT_PROMPT.format(
                        ############# Not Implement #############
                    ))]
                })
            except Exception as error:
                raise error

            return {
                'trajectory' : result_state['messages'],
                'input_tokens' : result_state['input_tokens'],
                'output_tokens' : result_state['output_tokens'],
                'total_tokens' : result_state['total_tokens']
            }
        # ================================================================================================================
        
        return _generator
    
    # -----------------------------------------------------------------------------------------------
    # Define Evaluator Node
    # -----------------------------------------------------------------------------------------------
    def _get_evaluator_node(self) -> Callable:

        # Evaluator Node
        # ==========================================================================================
        def _evaluator(state: ACEState):
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
            
            print(f"[Evaluator] 📊 Evaluation Report: \n{evaluation_report}")
            
            return {'evaluation' : evaluation_report}
        # ==========================================================================================

        return _evaluator
        
    # --------------------------------------------------------------------------------------------------------
    # Define Reflector
    # --------------------------------------------------------------------------------------------------------
    def _get_reflector_node(self) -> Callable:

        reflector = ReflectorModule(
            env=self.env,
            system_prompt=self.reflector_system_prompt,
            model_config=self.model_config
        )

        # Reflector Module
        # ================================================================================================================
        def _reflector(state: ACEState) -> ACEState:

            try:
                result_state: ReActState = reflector.invoke({
                    'messages' : [HumanMessage(content=REFLECTOR_INPUT_PROMPT.format(
                        ############# Not Implement #############
                    ))]
                })
            except Exception as error:
                raise error
            
            return {
                'reflector_output' : dict(result_state['messages'][-1]),
                'input_tokens' : result_state['input_tokens'],
                'output_tokens' : result_state['output_tokens'],
                'total_tokens' : result_state['total_tokens']
            }
        # ================================================================================================================

        return _reflector
    
    # --------------------------------------------------------------------------------------------------------
    # Define Curator
    # --------------------------------------------------------------------------------------------------------
    def _get_curator_node(self) -> Callable:

        openai_client = ChatOpenAI(**self.model_config)
        curator = openai_client.with_structured_output(
            ############ NotImplement ############ 
        )

        # Curator Module
        # ================================================================================================================
        def _curator(state: ACEState) -> ACEState:

            request_messages: Sequence[AnyMessage] = [SystemMessage(content=self.curator_system_prompt)] + [HumanMessage(content=CURATOR_INPUT_PROMPT.format(
                ############ NotImplement ############ 
            ))]

            response: AIMessage = get_response_with_retry(
                model_client=curator,
                messages=request_messages,
                max_retries=3
            )

            token_usage = get_token_usage_from_message(response)
            
            
            return {
                'curator_output' : dict(response.content),
                'input_tokens' : token_usage['input_tokens'],
                'output_tokens' : token_usage['output_tokens'],
                'total_tokens' : token_usage['total_tokens']
            }
        # ================================================================================================================

        return _curator

    # --------------------------------------------------------------------------------------------------------
    # Define conditional edge function
    # --------------------------------------------------------------------------------------------------------
    def _get_should_continue(self) -> Callable:
        
        # should continue
        # ================================================================================================================
        def _should_continue(state: ACEState) -> str:

            if len(state['reflector_output']) == 3:
                return 'end'
            elif 'Succeed' in state['evaluation']:
                return 'end'
            else:
                'reflector'
        # ================================================================================================================

        return _should_continue
    
    # --------------------------------------------------------------------------------------------------------
    # Build Agent
    # --------------------------------------------------------------------------------------------------------
    def _build_agent(self) -> CompiledStateGraph:

        # ----------------------------------------------------
        # Get Nodes
        # ----------------------------------------------------
        _generator = self._get_generator_node()
        _evaluator = self._get_evaluator_node()
        _reflector = self._get_reflector_node()
        _curator = self._get_curator_node()

        # ----------------------------------------------------
        # Get Conditional Edge function
        # ----------------------------------------------------
        _should_continue = self._get_should_continue()

        # ----------------------------------------------------
        # Define ACE Agent workflow
        # ----------------------------------------------------
        workflow = StateGraph(ACEState)

        # add node
        workflow.add_node('generator', _generator)
        workflow.add_node('evaluator', _evaluator)
        workflow.add_node('reflector', _reflector)
        workflow.add_node('curator', _curator)

        # add edges
        workflow.add_edge(START, 'generator')
        workflow.add_edge('generator', 'evaluator')
        workflow.add_conditional_edges(
            'evaluator',
            _should_continue,
            {
                'end' : END,
                'reflector' : 'reflector'
            }
        )
        workflow.add_edge('reflector', 'curator')
        workflow.add_edge('curator', 'generator')

        # build agent
        return workflow.compile()
