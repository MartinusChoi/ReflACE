from typing import Tuple, List, Dict

class MockAlfworldEnv:
    """
    A mock environment for testing agent logic without the full ALFWorld setup.
    """
    def __init__(self):
        self.task = "Simple task"
        self.steps = 0
        self.finished = False

    def reset(self, task_idx: int = 0) -> Tuple[str, Dict]:
        self.steps = 0
        self.finished = False
        return "You are in a room. There is an apple on the table. Your task is to eat the apple.", {"valid_actions": ["eat apple", "look"]}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        self.steps += 1
        observation = ""
        reward = 0.0
        done = False
        info = {}

        if action.strip() == "eat apple":
            observation = "You eat the apple."
            reward = 1.0
            done = True
            self.finished = True
        else:
            observation = "Nothing happens."
            reward = 0.0
            done = False
        
        return observation, reward, done, False, info
