import json

def show_trajectory(experiment_name, task_idx):
    
    trajectories = json.load(open(f"./evaluation_results/trajectories_[{experiment_name}].json"))

    task_ids = list(trajectories.keys())
    
    trajectory = trajectories[task_ids[task_idx]]
    
    result = ""
    for msg in trajectory['trajectory']:
        if 'role' in msg and msg['role'] == 'user':
            result += "========================================== [ 👤 User ] ============================================\n"
            result += f"{msg['content']}\n"
        elif 'role' in msg and msg['role'] == 'assistant':
            result += "======================================== [ 🤖 Assistant ] ==========================================\n"
            result += f"{msg['content']}\n"
        elif 'type' in msg and msg['type'] == 'function_call':
            result += "======================================== [ 🤖 Assistant ] ==========================================\n"
            result += f"{json.loads(msg['arguments'])['code']}\n"
        elif 'type' in msg and msg['type'] == 'function_call_output':
            result += "================================== [ 🌏 Environment Feedback ] ====================================\n"
            result += f"{msg['output']}\n"
        result += "====================================================================================================\n\n\n"
    

    print(result)

    with open(f"./evaluation_results/trajectory_{task_ids[task_idx]}.txt", "w", encoding="utf-8") as f:
        f.write(result)
    
