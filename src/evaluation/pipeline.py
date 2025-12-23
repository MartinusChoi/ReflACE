import os
from typing import Union
import json
from ..env.appworld_env import AppWorldEnv
from ..agent.react import ReActAgent
from ..agent.reflexion import ReflexionAgent

def run_evaluation(
    agent:Union[ReActAgent, ReflexionAgent],
    env_wrapper:AppWorldEnv,
    task_limit: int = None,
    experiment_name: str = "SampleExperiment",
    save_dir: str = None,
):
    """
    Run the evaluation loop.

    Args:
        agent (Union[ReActAgent, ReflexionAgent, ACEAgent, RefLACEAgent]): The agent to evaluate.
        env_wrapper (AppWorldEnv): The environment wrapper to evaluate on.
        task_limit (int, optional): The number of tasks to evaluate. Defaults to None.
        experiment_name (str, optional): The name of the experiment. Defaults to "SampleExperiment".
    """

    print(f"🚀 Starting evaluation experiment: {experiment_name}\n\n")

    # Set Total task limit
    # set for taking partial evaluation for few tasks
    total_tasks = len(env_wrapper.task_ids)
    if task_limit:
        print(f"🗂️ Limiting evaluation to first {task_limit} tasks out of {total_tasks}.")
        total_tasks = min(total_tasks, task_limit)
    print(f"📊 Evaluating on {total_tasks} tasks...")
    
    results = {
        "passed_task_id": [],
        "failed_task_id": [],
        "errors_task_id": []
    }

    trajectory = {}
    
    for i in range(total_tasks):
        current_task_id = env_wrapper.task_ids[i]
        print(f"\n📌 [Task {i+1}/{total_tasks}] ID: {current_task_id}")
        
        try:
            # Set environment for the specific task
            env_wrapper.set_env(task_id=i, experiment_name=experiment_name)
            
            # Run Agent
            run_output = agent.run(env_wrapper=env_wrapper)
            
            if isinstance(agent, ReActAgent):
                trajectory[current_task_id] = {
                    'trajectory': run_output['trajectory'].to_chat_prompt()
                }
            else:
                trajectory[current_task_id] = {
                    'trajectory': run_output['trajectory'].to_chat_prompt(),
                    'reflection_history' : run_output['reflection_history'],
                }
            
            # Evaluate
            evaluation = env_wrapper.env.evaluate()
            
            # Check success
            if evaluation.success:
                print("    ✅ PASSED")
                results["passed_task_id"].append(current_task_id)
            else:
                print(f"    ❌ FAILED (Passed Requirements: {evaluation.pass_count}, Failed Requirements: {evaluation.fail_count})")
                results["failed_task_id"].append(current_task_id)
                
            # Close environment resource
            env_wrapper.env.close()
            
        except Exception as e:
            print(f"    ⚠️ ERROR executing task {current_task_id}: {e}")
            results["errors_task_id"].append(current_task_id)
            if env_wrapper.env:
                try:
                    env_wrapper.env.close()
                except:
                    pass

    # Final Report
    print("\n" + "="*50)
    print("📝 EVALUATION REPORT")
    print("="*50)
    print(f"    🗂️ Total Tasks: {total_tasks}")
    print(f"    ✅ Passed: {len(results['passed_task_id'])}")
    print(f"    ❌ Failed: {len(results['failed_task_id'])}")
    print(f"    ⚠️ Errors: {len(results['errors_task_id'])}")
    
    success_rate = (len(results['passed_task_id']) / total_tasks) * 100 if total_tasks > 0 else 0
    print(f"    📊 Success Rate: {success_rate:.2f}%")
    
    if results['failed_task_id']:
        print(f"\n❌ Failed Task IDs: {results['failed_task_id']}")
    if results['errors_task_id']:
        print(f"\n⚠️ Errored Task IDs: {results['errors_task_id']}")
    print("="*50)

    performance_report = {
        'total_tasks': total_tasks,
        'passed_tasks': len(results['passed_task_id']),
        'failed_tasks': len(results['failed_task_id']),
        'errored_tasks': len(results['errors_task_id']),
        'success_rate': success_rate,
        'passed_task_ids': results['passed_task_id'],
        'failed_task_ids': results['failed_task_id'],
        'errored_task_ids': results['errors_task_id'],
    }
    
    # Save performance report
    with open(f"{save_dir}/performance-report_[{experiment_name}].json", "w") as f:
        json.dump(performance_report, f, indent=4)
    
    # Save trajectories
    with open(f"{save_dir}/trajectories_[{experiment_name}].json", "w") as f:
        json.dump(trajectory, f, indent=4)