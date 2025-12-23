import os
import argparse
from src.llm.openai_client import OpenAIClient
from src.agent.react import ReActAgent
from src.agent.reflexion import ReflexionAgent
from src.llm.tools import TOOLS
from src.env.appworld_env import AppWorldEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["react", "reflexion", "ace", "reflace"], required=True)
    parser.add_argument("--model_name", type=str, default='gpt-4o')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--env", type=str, choices=["appworld"], default="appworld")
    parser.add_argument("--task_type", type=str, choices=["train", "test", "dev"], default="train")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="sample")
    args = parser.parse_args()

    # --------------------------------------------------------------------------------------------------------------
    # Setup LLM Client and Agent
    # --------------------------------------------------------------------------------------------------------------
    agent = None
    if args.agent == 'react':
        with open(os.path.join('src', 'prompt', 'react', 'system_prompt_oneshot.txt'), 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        actor_client = OpenAIClient(
            model_name=args.model_name,
            temperature=args.temperature,
            tools=TOOLS,
            system_prompt=system_prompt
        )

        agent = ReActAgent(actor_client, env)

    elif args.agent == 'reflexion':
        with open(os.path.join('src', 'prompt', 'react', 'system_prompt_oneshot.txt'), 'r', encoding='utf-8') as f:
            actor_system_prompt = f.read()
        with open(os.path.join('src', 'prompt', 'reflexion', 'reflector_system_prompt.txt'), 'r', encoding='utf-8') as f:
            reflector_system_prompt = f.read()
        
        actor_client = OpenAIClient(
            model_name=args.model_name,
            temperature=args.temperature,
            tools=TOOLS,
            system_prompt=actor_system_prompt
        )
        reflector_client = OpenAIClient(
            model_name=args.model_name,
            temperature=args.temperature,
            system_prompt=reflector_system_prompt
        )

        agent = ReflexionAgent(
            actor_client=actor_client,
            reflector_client=reflector_client,
            env=env
        )
    elif args.agent == 'ace':
        raise NotImplementedError()
    elif args.agent == 'reflace':
        raise NotImplementedError()
    
    print(f"📌 Running Agent Mode: {args.agent}")
    print(f"    📍 LLM Core: {args.model_name}")
    print(f"    📍 LLM Temperature: {args.temperature}")

    
    # --------------------------------------------------------------------------------------------------------------
    # Setup Environment
    # --------------------------------------------------------------------------------------------------------------
    env = AppWorldEnv(task_type=args.task_type)
    env.set_env(
        task_id=args.task_id,
        experiment_name=args.experiment_name
    )

    print(f"📌 Running Environment: {args.env}")
    print(f"    📍 Task Type: {args.task_type}")
    print(f"    📍 Experiment Name: {args.experiment_name}")
    
    result = agent.run(env)
    
    print("Result:", result)
    
    if args.mode in ["ace", "reflace"]:
        print("Playbook content:")
        print(agent.playbook.get_content())

if __name__ == "__main__":
    main()
