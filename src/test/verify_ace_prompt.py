from src.agent.ace import PROMPT

def verify_ace_prompt():
    try:
        instruction = "Test Instruction"
        trajectory = "Test Trajectory"
        predicted_answer = "Test Answer"
        ground_truth_answer = "True Answer"
        environment_feedback = "Feedback"
        playbook = "Playbook Content"

        formatted_prompt = PROMPT['only_ace'].format(
            instruction=instruction,
            trajectory=trajectory,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer,
            environment_feedback=environment_feedback,
            playbook=playbook
        )
        print("✅ Prompt formatting successful!")
        print("-" * 20)
        print(formatted_prompt)
        print("-" * 20)
    except Exception as e:
        print(f"❌ Prompt formatting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_ace_prompt()
