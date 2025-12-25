from src.utils.show_result import show_trajectory
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--task_idx", type=int, required=True)
    args = parser.parse_args()

    show_trajectory(args.experiment_name, args.task_idx)