from typing import Sequence, Callable

from langchain.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from ..state import ReActState
from .base import BaseAgent
from ..utils.llm import get_response_with_retry
from ..utils.token_usage import get_token_usage_from_message


# --------------------------------------------------------------------------------------------------------
# ReAct Agent
# --------------------------------------------------------------------------------------------------------
class ReActAgent(BaseAgent):
    """
    ReAct Agent Class.
    """
    
    # ----------------------------------------------------------------------------
    # Define Actor Node
    # ----------------------------------------------------------------------------
    def _get_actor_node(self) -> Callable:
        
        # Actor Node
        # ============================================================================================================
        def _actor(state: ReActState):

            # create request message list (insert system message in current message history)
            messages: Sequence[AnyMessage] = state['messages']
            request_messages: Sequence[AnyMessage] = [SystemMessage(content=self.system_prompt)] + messages

            # get response from llm client with retry logic
            response: AIMessage = get_response_with_retry(
                model_client=self.openai_client_with_tools,
                messages=request_messages,
                max_retries=3
            )
            
            # get token usages.
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
    # Define conditional edge (should continue)
    # ----------------------------------------------------------------------------
    def _get_should_continue(self):
        
        # should_continue
        # =============================================================================
        def _should_continue(state: ReActState):

            for msg in reversed(state['messages']):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg
                    break
            
            if hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
                for tool_call in last_ai_msg.tool_calls:
                    if 'complete_task' in tool_call['args']['code']:
                        return 'end'
                
            return 'actor'
        # =============================================================================

        return _should_continue


    def _build_agent(self) -> CompiledStateGraph:
        # ----------------------------------------------------------------------------
        # Get Nodes
        # ----------------------------------------------------------------------------
        _actor = self._get_actor_node()
        _tools = self._get_tool_node()
        
        # ----------------------------------------------------------------------------
        # Get conditional edge function
        # ----------------------------------------------------------------------------
        _should_continue = self._get_should_continue()

        # ----------------------------------------------------------------------------
        # Define ReAct Workflow
        # ----------------------------------------------------------------------------
        # create graph builder
        workflow = StateGraph(state_schema=ReActState)

        # add nodes
        workflow.add_node("actor", _actor)   # actor node
        workflow.add_node("tools", _tools)   # tool node

        # add edges
        # Use `Command` instance instead state/conditional edge 
        workflow.add_edge(START, "actor")
        workflow.add_edge('actor', 'tools')
        workflow.add_conditional_edges(
            'tools',
            _should_continue,
            {
                'actor' : 'actor',
                'end' : END
            }
        )

        # compile graph and return CompiledStateGraph instance
        return workflow.compile()
    
    
    def invoke(self, state: ReActState) -> ReActState:
        return self.agent.invoke(state)