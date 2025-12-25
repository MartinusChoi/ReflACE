from typing import Dict, Any, List, Literal, Optional

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv
from ..core.trajectory import Trajectory
from ..core.messages import (
    UserMessage,
    AIMessage,
    ToolCallMessage,
    ToolCallOutputMessage,
)
from ..prompt.reflexion.input_prompt import (
    reflexion_reflector_input_prompt,
    reflexion_reflector_with_gt_input_prompt
)
from ..core.reflection import ReflectionHistory





# -------------------------------------------------------------------------------------
# Reflector Agent Class
# -------------------------------------------------------------------------------------
class ReflectorAgent(BaseAgent):
    """
    Reflector Agent that use ReActAgent for Actor Module with a Reflection loop
    """

    def __init__(
        self, 
        actor_client: OpenAIClient,
        use_ground_truth: bool = False,
        max_reflection_step: Optional[int] = 10,
    ):
        super().__init__(actor_client=actor_client)
        self.use_ground_truth = use_ground_truth
        if not self.use_ground_truth:
            self.max_reflection_step = max_reflection_step
    
    def _build_prompt(
        self, 
        env_wrapper:AppWorldEnv,
        trajectory:Trajectory,
        reflection_history: ReflectionHistory,
        failure_report: List[Dict[str, str]]
    ) -> str:
        supervisor_info = env_wrapper.get_supervisor_info()   # supervisor information
        instruction = env_wrapper.get_instruction()           # instruction

        failure_report_str = "<failure report>\n"
        for failure in failure_report:
            failure_report_str += f"{failure['trace']}\n"
        failure_report_str += "</failure report>\n"

        if self.use_ground_truth:
            gt = env_wrapper.get_ground_truth()
            gt_api_calls = gt.api_calls
            gt_code = gt.solution_code
            gt_apis = gt.required_apis
            gt_apps = gt.required_apps
            
            return reflexion_reflector_with_gt_input_prompt.template.format(
                first_name = supervisor_info['first_name'],                       # supervisor information
                last_name = supervisor_info['last_name'],                         # supervisor information
                email = supervisor_info['email'],                                 # supervisor information
                phone_number = supervisor_info['phone_number'],                   # supervisor information
                instruction = instruction,                                        # task instruction
                failure_report=failure_report_str,                                # failure report
                reflection_history = reflection_history.get_history(),            # added in Reflexion Agent setting
                trajectory = trajectory.to_str(),                                 # added in Reflexion Agent setting
                ground_truth_api_calls = gt_api_calls,                            # ground truth api calls
                ground_truth_code = gt_code,                                      # ground truth code
                ground_truth_required_apis = gt_apis,                             # ground truth required apis
                ground_truth_required_apps = gt_apps                              # ground truth required apps
            )

        else:
            return reflexion_reflector_input_prompt.template.format(
                first_name = supervisor_info['first_name'],                       # supervisor information
                last_name = supervisor_info['last_name'],                         # supervisor information
                email = supervisor_info['email'],                                 # supervisor information
                phone_number = supervisor_info['phone_number'],                   # supervisor information
                instruction = instruction,                                        # task instruction
                failure_report=failure_report_str,                                # failure report
                reflection_history = reflection_history.get_history(),            # added in Reflexion Agent setting
                trajectory = trajectory.to_str()                                  # added in Reflexion Agent setting
            )

    def run(
        self,
        env_wrapper:AppWorldEnv,
        trajectory:Trajectory,
        reflection_history: ReflectionHistory,
        failure_report: str
    ) -> ReflectionHistory:

        reflection_trajectory = Trajectory(messages=[
            UserMessage(
                content=self._build_prompt(env_wrapper, trajectory, reflection_history, failure_report=failure_report)
            )
        ])

        if self.use_ground_truth:
            # Get Reflection Response
            response_messages = self.actor_client.get_response(messages=reflection_trajectory.to_chat_prompt())

            for message in response_messages:
                if isinstance(message, AIMessage):
                    reflection_history.add_reflection(messages=[message])

            return reflection_history
        else:
            for step in range(self.max_reflection_step):
                response_messages = self.actor_client.get_response(messages=reflection_trajectory.to_chat_prompt())

                for message in response_messages:
                    if isinstance(message, ToolCallMessage):
                        reflection_trajectory.append(message)

                        code = json.loads(message.arguments)['code']

                        obs = env_wrapper.action(code)

                        reflection_trajectory.append(
                            ToolCallOutputMessage(
                                msg_type='function_call_output',
                                call_id=message.call_id,
                                content=obs
                            )
                        )

                    elif isinstance(message, AIMessage):
                        reflection_history.add_reflection(messages=[message])
                        return reflection_history
                    else:
                        raise ValueError(f"Unexpected message type: {type(message)}")
            
            return reflection_history
                                        









# -------------------------------------------------------------------------------------
# Reflexion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent that use ReActAgent for Actor Module with a Reflection loop
    """

    def __init__(
        self, 
        actor_client: OpenAIClient,
        reflector_client: OpenAIClient,
        use_ground_truth: bool = False,
    ):
        super().__init__(actor_client=actor_client)

        self._actor = ReActAgent(self.actor_client)
        self.reflection_history = ReflectionHistory(max_size=3)

        if use_ground_truth:
            self._reflector = ReflectorAgent(reflector_client, use_ground_truth=True)
        else:
            self._reflector = ReflectorAgent(reflector_client)
    
    def _reset_reflection_history(self) -> None:
        self.reflection_history = ReflectionHistory(max_size=3)
    
    def _build_prompt(self) -> None:
        pass
    
    def _evaluator(
        self,
        env_wrapper: AppWorldEnv
    ) -> bool:
        # evaluate agent task results
        evaluation = env_wrapper.evaluate_env()
        # return True if task is success, False otherwise
        return {
            'is_success' : evaluation.success,
            'failure_report' : evaluation.failures,
        }
    
    def run(
        self,
        env_wrapper: AppWorldEnv,
        max_steps: int = 3
    ) -> Dict[str, Any]:

        for step in range(max_steps):
            # actor action, get trajectory of actor action
            action = self._actor.run(
                agent_type='reflexion',
                env_wrapper=env_wrapper,
                reflection_history=self.reflection_history.get_history(),
            )

            # evaluate action of actor module
            # if task success, done reflexion loop
            evaluation_result = self._evaluator(env_wrapper)
            print("    📍 Evaluator Done!")
            if evaluation_result['is_success']: 
                print(f"    📍 Reflexion Loop Done on step {step+1}")
                break

            if step+1 == max_steps: 
                print(f"    📍 Reflexion Loop Done on step {step+1}")
                break

            # reflect on actor action
            self.reflection_history = self._reflector.run(
                env_wrapper=env_wrapper,
                trajectory=action['trajectory'],
                reflection_history=self.reflection_history,
                failure_report=evaluation_result['failure_report'],
            )
            print("    📍 Reflector Done!")
        
        reflection_history = self.reflection_history.get_history()
        self._reset_reflection_history()
        
        return {
            'trajectory' : action['trajectory'],
            'reflection_history' : reflection_history,
        }