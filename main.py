import os
import argparse
from src.llm.openai_client import OpenAIClient
from src.env.alfworld_env import MockAlfworldEnv
from src.agent.react import ReActAgent
from src.agent.reflexion import ReflexionAgent
from src.agent.ace import ACEAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["react", "reflexion", "ace", "reflace"], required=True)
    args = parser.parse_args()

    # Initialize components
    # NOTE: Set OPENAI_API_KEY env var or mock it for this test if needed.
    # For this verification, we are using a real client structure but maybe a mock if no key.
    # But let's assume valid key or gracefull failure.
    llm_client = OpenAIClient(temperature=0.0)
    env = MockAlfworldEnv()
    
    task_desc, _ = env.reset()
    print(f"Task: {task_desc}")

    agent = None
    if args.mode == "react":
        agent = ReActAgent(llm_client, env)
    elif args.mode == "reflexion":
        agent = ReflexionAgent(llm_client, env)
    elif args.mode == "ace":
        agent = ACEAgent(llm_client, env, use_reflexion=False)
    elif args.mode == "reflace":
        agent = ACEAgent(llm_client, env, use_reflexion=True)
        
    print(f"Running Agent Mode: {args.mode}")
    result = agent.run(task_desc)
    
    print("Result:", result)
    
    if args.mode in ["ace", "reflace"]:
        print("Playbook content:")
        print(agent.playbook.get_content())

if __name__ == "__main__":
    main()
