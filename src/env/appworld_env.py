from appworld import AppWorld, load_task_ids
from typing import Literal, Union

class AppWorldEnv:
    def __init__(
        self, 
        task_type: Literal["train", "test", "dev"] = "train",
    ):
        self.task_ids = load_task_ids(task_type)
        self.env = None
    
    def set_env(
        self,
        task_id:int=0,
        experiment_name:str = "sample"
    ):
        self.experiment_name = experiment_name
        self.env = AppWorld(
            task_id=self.task_ids[task_id],
            experiment_name=self.experiment_name
        )
        self.cur_task_id = task_id
    
    def reset_env(self):
        if self.env is None:
            raise ValueError("Environment is not initialized. Call set_env() first.")
        
        return self.env.reset()
    
    def action(self, action: str):
        if self.env is None:
            raise ValueError("Environment is not initialized. Call set_env() first.")
    
        return self.env.execute(action)
    
    def step(self):
        if self.env is None:
            raise ValueError("Environment is not initialized. Call set_env() first.")
        
        if (self.cur_task_id+1) >= len(self.task_ids):
            return None
        
        self.set_env(
            task_id=self.cur_task_id+1,
            experiment_name=self.experiment_name
        )
        
    def get_instruction(self):
        if self.env is None:
            raise ValueError("Environment is not initialized. Call set_env() first.")
        
        return self.env.task.instruction
    
    def get_supervisor_info(self):
        return {
            'first_name' : self.env.task.supervisor.first_name,
            'last_name' : self.env.task.supervisor.last_name,
            'email' : self.env.task.supervisor.email,
            'phone_number' : self.env.task.supervisor.phone_number
        }


def evaluate_agent(
    agent,
    env:AppWorldEnv,
    max_task:Union[int, None] = None
):
    success_cnt = 0
    failed_cnt = 0

    max_task = max_task or len(env.task_ids)

    for i in range(max_task):

        result = agent.run(env=env)

        evaluation = env.env.evaluate()

        success_cnt += evaluation.pass_count
        failed_cnt += evaluation.fail_count
        
        env.step()
    
    return {
        'success_cnt' : success_cnt,
        'failed_cnt' : failed_cnt,
        'total_cnt' : len(success_cnt + failed_cnt),
        'success_rate' : success_cnt / (success_cnt + failed_cnt)
    }