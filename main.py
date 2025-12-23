import os
import argparse
from src.evaluation.setup import setup_pipeline
from src.evaluation.pipeline import run_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["react", "reflexion", "ace", "reflace"], required=True)
    parser.add_argument("--model_name", type=str, default='gpt-4o')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--task_type", type=str, choices=["train", "test", "dev"], default="train")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="sample")
    parser.add_argument("--task_limit", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./evaluation_results")
    args = parser.parse_args()
    
    print(f"📌 Running Agent Mode: {args.agent}")
    print(f"    📍 LLM Core: {args.model_name}")
    print(f"    📍 LLM Temperature: {args.temperature}")
    print(f"    📍 Max Steps: {args.max_steps}")
    print(f"📌 Running Environment: AppWorld")
    print(f"    📍 Task Type: {args.task_type}")
    print(f"    📍 Experiment Name: {args.experiment_name}")
    print(f"    📍 Task Limit: {args.task_limit}")
    print(f"📌 Save Directory: {args.save_dir}\n\n")
    

    agent, env_wrapper = setup_pipeline(
        agent=args.agent,
        model_name=args.model_name,
        temperature=args.temperature,
        task_type=args.task_type,
        task_id=args.task_id,
        experiment_name=args.experiment_name
    )
    
    run_evaluation(
        agent=agent,
        env_wrapper=env_wrapper,
        task_limit=args.task_limit,
        experiment_name=args.experiment_name,
        max_steps=args.max_steps,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
