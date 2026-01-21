from ..agents.react import ReActAgent
from ..agents.reflexion import ReflexionAgent
from ..agents.ace import ACEAgent
from ..utils.token_usage import calc_token_price
from ..prompt.react import SYSTEM_PROMPT, INPUT_PROMPT
from ..core.playbook import PlayBook

from langchain.messages import HumanMessage

from typing import Literal, List, Dict

from appworld import AppWorld, load_task_ids

class AppWorldEvalator:
    def __init__(
        self,
        agent_type: Literal['react', 'reflexion', 'ace'],
        dataset_type: Literal['train', 'dev'],
        experiment_name: str,
        first_k_task: int = None,
        model_config: Dict[str, str | float | bool] = {
            'model' : 'gpt-4o',
            'temperature' : 0.0,
            'stream_usage' : True
        }
    ) -> None:
        self.agent_type = agent_type
        self.experiment_name = experiment_name
        self.model_config = model_config

        self.task_ids: List[str] = load_task_ids(dataset_name=dataset_type)
        if first_k_task:
            self.task_ids = self.task_ids[:first_k_task]

        self.result: Dict[str, Dict[str, str | int | float]] = {}

        if self.agent_type == 'ace':
            self.playbook:PlayBook = None       # playbook that retain over task ids in ACEAgent
        elif self.agent_type == 'reflexion':
            self.reflections:List[str] = None     # reflection that retain over task ids in ReflexionAgent
        
    def evaluate(self) -> Dict[str, Dict[str, str | int | float]]:

        for task_id in self.task_ids:
            print(f"⏳ Start task '{task_id}'...")
            # ----------------------------------------------------------------------------------------
            # get AppWorld instance with current 'task_id'
            # ----------------------------------------------------------------------------------------
            env = AppWorld(
                task_id=task_id, 
                ground_truth_mode='full',
                random_seed=42,
                experiment_name=self.experiment_name
            )

            # ----------------------------------------------------------------------------------------
            # initialize agent instance with current task AppWorld instance
            # ----------------------------------------------------------------------------------------
            if self.agent_type == 'react':                                    # ReAct Agent
                agent = ReActAgent(
                    env=env, 
                    system_prompt=SYSTEM_PROMPT,
                    model_config=self.model_config
                )
            elif self.agent_type == 'reflexion':                              # Reflexion Agent
                agent = ReflexionAgent(
                    env=env,
                    model_config=self.model_config
                )
            elif self.agent_type == 'ace':                                    # ACE Agent
                agent = ACEAgent(
                    env=env,
                    model_config=self.model_config
                )
            else:
                raise ValueError("Unknown Agent Type. It must be one of : 'react', 'reflexion', 'ace'")
            

            # ----------------------------------------------------------------------------------------
            # run agent on current task
            # ----------------------------------------------------------------------------------------

            # create input state for agent
            if self.agent_type == 'react':                                         # ReAct Agent input state
                input_state = {
                    'messages' : [
                        HumanMessage(
                            content=INPUT_PROMPT.format(
                                first_name = agent.env.task.supervisor.first_name,
                                last_name = agent.env.task.supervisor.last_name,
                                email = agent.env.task.supervisor.email,
                                phone_number = agent.env.task.supervisor.phone_number,
                                instruction = agent.env.task.instruction
                        ))
                    ],
                }
            elif self.agent_type == 'reflexion':                                   # Reflexion Agent input state
                input_state = {'reflections' : [] if self.reflections == None else self.reflections}
            elif self.agent_type == 'ace':                                         # ACE Agent input state
                input_state = {'playbook' : {} if not self.playbook == None else self.playbook}

            # run agent on task
            result = agent.invoke(input_state)

            # ----------------------------------------------------------------------------------------
            # get metadata of current agent run
            # ----------------------------------------------------------------------------------------

            if self.agent_type == 'reflexion':
                self.reflections = result['reflections']
            elif self.agent_type == 'ace':
                self.playbook = result['playbook']

            # get agent latency
            latency = result['latency']

            # get token usage info
            input_tokens = result['input_tokens']
            output_tokens = result['output_tokens']
            total_tokens = result['total_tokens']

            # calculate price with used tokens
            price = calc_token_price(
                model='gpt-4o',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Task Result Evaluation
            evaluation = agent.env.evaluate()

            # Get evaluation result
            task_status = (evaluation.pass_count == evaluation.total_count)
            pass_requirements = evaluation.pass_count
            fail_requirements = evaluation.fail_count
            total_requirements = evaluation.total_count
            pass_requirement_info = evaluation.passes
            fail_requirement_info = evaluation.failures


            # ----------------------------------------------------------------------------------------
            # add evaluation metadata of current task_id
            # ----------------------------------------------------------------------------------------
            self.result[task_id] = {
                'latency' : latency,
                'input_tokens' : input_tokens,
                'output_tokens' : output_tokens,
                'total_tokens' : total_tokens,
                'price' : price,
                'task_status' : task_status,
                'pass_requirements' : pass_requirements,
                'fail_requirements' : fail_requirements,
                'total_requirements' : total_requirements,
                'pass_requirement_info' : pass_requirement_info,
                'fail_requirement_info' : fail_requirement_info
            }

            print(f"✅ Task '{task_id}' complete.\n")
        

        print(f"✅ All {len(self.task_ids)} tasks are completed!")

        return self.result