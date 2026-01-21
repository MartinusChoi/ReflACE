import json
import os
import argparse
from src.tests.evaluate import AppWorldEvalator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, choices=["react", "reflexion", "ace", "reflace"], required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset_type", type=str, choices=["train", "test", "dev"], default="dev")
    parser.add_argument("--experiment_name", type=str, default="sample")
    parser.add_argument("--first_k_task", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./evaluation_results")
    args = parser.parse_args()
    
    print("=="*50)
    print(f"📌 Running Agent Type: {args.agent_type}")
    print(f"    📍 LLM Core Name: {args.model_name}")
    print(f"    📍 LLM Core Temperature: {args.temperature}")
    print(f"📌 Running Environment: AppWorld")
    print(f"    📍 Dataset Type: {args.dataset_type}")
    print(f"    📍 Experiment Name: {args.experiment_name}")
    print(f"    📍 Number of Task: {args.first_k_task if args.first_k_task is not None else 'Full'}")
    print(f"📌 Save Directory: {args.save_dir}")
    print("=="*50 + "\n\n")
    

    evaluator = AppWorldEvalator(
        agent_type=args.agent_type,
        dataset_type=args.dataset_type,
        experiment_name=args.experiment_name,
        first_k_task=args.first_k_task,
        model_config={
            'model' : args.model_name,
            'temperature' : args.temperature,
            'stream_usage' : True
        }
    )
    
    result = evaluator.evaluate()
    with open(os.path.join(f"{args.save_dir}", f"{args.experiment_name}.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
