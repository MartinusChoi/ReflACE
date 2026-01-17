from typing import Sequence, Callable

from langchain.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from ..state import ReActState
from .base import BaseAgent


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
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # get response
                    response: AIMessage = self.openai_client_with_tools.invoke(request_messages)
                    print(f"\n[Actor] ✅ Request succeed on attept {attempt+1}/{max_retries}")
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        print(f"[Actor] 🌏 Actor make tool call. > {len(response.tool_calls)} tool calls")
                        break
                    else:
                        print("[Actor] ⚠️ Actor didn't make tool call. Actor will retry.")
                except Exception as error:
                    print(f"[Actor] ⚠️ Request failed on attept {attempt+1}/{max_retries}")
                    # raise error when attempt hit max retry limit.
                    if attempt + 1 == max_retries:
                        print(f"[Actor] ⛔️ Model Request failed. Please Try Later.")
                        raise error
            
            # get token usages.
            try:
                input_tokens = response.usage_metadata['input_tokens']
                output_tokens = response.usage_metadata['output_tokens']
                total_tokens = response.usage_metadata['total_tokens']
                print(f"[Actor] ✅ Token usage is collected successfully.")
            except Exception as error:
                print(f"[Actor] ⛔️ Response message doesn't contain token usage metadata.")
                raise error
            
            # update agent state
            return {
                'messages' : [response],
                'input_tokens' : input_tokens,
                'output_tokens' : output_tokens,
                'total_tokens' : total_tokens,
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
                        print(f"⭐️ Agent Complete Task! (called `complete_task`)")
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