
import os
import sys
import importlib.util
import argparse

# Check for appworld installation
if importlib.util.find_spec("appworld") is None:
    print("❌ 'appworld' package not found. Please install it first.")
    sys.exit(1)

import appworld
from appworld import AppWorld, load_task_ids

# Ensure we can import from src
# Assuming the script is run from the project root
if os.path.isdir('src'):
    pass
else:
    print("❌ 'src' directory not found. Please run this script from the project root.")
    sys.exit(1)

try:
    from src.env.appworld_env import AppWorldEnv
    from src.agent.react import ReActAgent
    from src.llm.openai_client import OpenAIClient
    from src.llm.tools import TOOLS
except ImportError as e:
    print(f"❌ Failed to import project modules: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)


def setup_pipeline():
    """
    Setup the evaluation pipeline.
    Checks for API keys and environment setup.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY environment variable not set.")
        # Proceeding might fail later if agent needs it
    
    print("✅ Pipeline setup complete.")


def run_evaluation(task_limit: int = None, experiment_name: str = "react_eval_default"):
    """
    Run the evaluation loop.
    """
    print(f"🚀 Starting evaluation experiment: {experiment_name}")
    
    # Load system prompt
    prompt_path = os.path.join('src', 'prompt', 'react', 'system_prompt_oneshot.txt')
    system_prompt = None
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    else:
        print(f"⚠️ Warning: System prompt file not found at {prompt_path}")

    # Initialize OpenAI Client
    # Pass TOOLS and system_prompt to match main.py configuration
    client = OpenAIClient(
        model_name="gpt-4o",
        temperature=0.0,
        tools=TOOLS,
        system_prompt=system_prompt
    ) 
    
    # Initialize Agent
    agent = ReActAgent(actor_client=client)
    
    # Initialize Environment Wrapper
    # task_type "dev" is standard for evaluation during development
    env_wrapper = AppWorldEnv(task_type="dev")
    
    # Load task IDs directly to know how many we have
    task_ids = load_task_ids("dev")
    total_tasks = len(task_ids)
    
    if task_limit:
        print(f"ℹ️ Limiting evaluation to first {task_limit} tasks out of {total_tasks}.")
        total_tasks = min(total_tasks, task_limit)
    
    print(f"📊 Evaluating on {total_tasks} tasks...")
    
    results = {
        "passed": [],
        "failed": [],
        "errors": []
    }
    
    for i in range(total_tasks):
        current_task_id = task_ids[i]
        print(f"\n[Task {i+1}/{total_tasks}] ID: {current_task_id}")
        
        try:
            # Set environment for the specific task
            # AppWorldEnv.set_env takes index, so we pass i (which maps to task_ids[i] internally)
            # wait, env.set_env(task_id=...) implementation:
            # self.env = AppWorld(task_id=self.task_ids[task_id], ...)
            # So if we pass 'i', it uses self.task_ids[i]. Correct.
            
            env_wrapper.set_env(task_id=i, experiment_name=experiment_name)
            
            # Run Agent
            # agent.run(env) returns {'finished': bool, 'trajectory': ...}
            run_output = agent.run(env=env_wrapper)
            
            # Evaluate
            # env.env.evaluate() returns an Evaluation object
            evaluation = env_wrapper.env.evaluate()
            
            # Check success
            # AppWorld evaluation has pass_count and fail_count
            is_success = (evaluation.fail_count == 0)
            
            if is_success:
                print("  ✅ PASSED")
                results["passed"].append(current_task_id)
            else:
                print(f"  ❌ FAILED (Passed: {evaluation.pass_count}, Failed: {evaluation.fail_count})")
                results["failed"].append(current_task_id)
                
            # Close environment resource
            env_wrapper.env.close()
            
        except Exception as e:
            print(f"  ⚠️ ERROR executing task {current_task_id}: {e}")
            results["errors"].append(current_task_id)
            if env_wrapper.env:
                try:
                    env_wrapper.env.close()
                except:
                    pass

    # Final Report
    print("\n" + "="*50)
    print("📝 EVALUATION REPORT")
    print("="*50)
    print(f"Total Tasks: {total_tasks}")
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Errors: {len(results['errors'])}")
    
    success_rate = (len(results['passed']) / total_tasks) * 100 if total_tasks > 0 else 0
    print(f"Success Rate: {success_rate:.2f}%")
    
    if results['failed']:
        print(f"\nFailed Task IDs: {results['failed']}")
    if results['errors']:
        print(f"Errored Task IDs: {results['errors']}")
        
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AppWorld Evaluation Pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to run (for testing)")
    parser.add_argument("--experiment", type=str, default="react_eval_v1", help="Experiment name")
    
    args = parser.parse_args()
    
    setup_pipeline()
    run_evaluation(task_limit=args.limit, experiment_name=args.experiment)
