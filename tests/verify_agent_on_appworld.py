from typing import Union
import argparse
from src.evaluation.setup import setup_pipeline
from src.env.appworld_env import AppWorldEnv
from src.agent.react import ReActAgent
from src.agent.reflexion import ReflexionAgent

def verify_agent(agent:str) -> None:
    """
    Run a verification loop similar to the main evaluation loop but without saving results to disk.
    
    Returns:
        bool: True if the loop completes without unhandled exceptions, False otherwise.
    """
    print(f"🚀 Starting verification for : {agent} agent\n")

    agent, env_wrapper = setup_pipeline(
        agent=agent,
        model_name="gpt-4o",
        temperature=0.0,
        task_type="dev",
        task_id=0,
        experiment_name=f"verify_{agent}"
    )
    total_tasks = 1
    try:
        current_task_id = env_wrapper.task_ids[0]
        
        # Set environment for the specific task
        env_wrapper.set_env(task_id=0, experiment_name=f"verify_{agent}")
        
        run_output = agent.run(
            env_wrapper=env_wrapper,
            max_steps=5
        )
        
        # Check output structure (basic validation)
        if 'trajectory' not in run_output:
            raise ValueError("⚠️ Agent run output missing 'trajectory' key.")
        
        # Close environment resource
        env_wrapper.env.close()
        
        print("✅ Verification Loop Completed Successfully\n\n")

    except Exception as e:
        print(f"⛔️ Verification Loop Crashed: {e}")