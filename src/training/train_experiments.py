import os
import sys
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Ensure the repository `src` directory is on sys.path so top-level
# imports like `environment.nav_env` work when this file is executed
# directly (e.g. `python src/training/train_experiments.py`).
file_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(file_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from environment.nav_env import CrafterNavigationEnv

RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGORITHMS = ["DQN", "PPO"]
ACTION_LEVELS = ["simple", "moderate", "complex"]
SEEDS = [42, 123, 999]
TOTAL_TIMESTEPS = 20000
EVAL_FREQ = 1000

def run_experiments():
    print(f"Starting $2 \\times 3$ Matrix Execution ({len(ALGORITHMS) * len(ACTION_LEVELS) * len(SEEDS)} Total Runs)...")
    
    for algo in ALGORITHMS:
        for action_level in ACTION_LEVELS:
            for seed in SEEDS:
                run_id = f"{algo.lower()}_{action_level}_seed{seed}"
                print(f"\n==================================================")
                print(f"Executing Run: {run_id}")
                print(f"==================================================")
                
                log_path = os.path.join(RESULTS_DIR, run_id)
                os.makedirs(log_path, exist_ok=True)
                
                # Wrap environment with Monitor for callback logging
                train_env = Monitor(CrafterNavigationEnv(action_level=action_level))
                eval_env = Monitor(CrafterNavigationEnv(action_level=action_level))
                
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=log_path,
                    log_path=log_path,
                    eval_freq=EVAL_FREQ,
                    n_eval_episodes=5,
                    deterministic=True,
                    verbose=0
                )
                
                if algo == "DQN":
                    model = DQN("MlpPolicy", train_env, learning_rate=1e-3, seed=seed, verbose=0)
                elif algo == "PPO":
                    model = PPO("MlpPolicy", train_env, learning_rate=1e-3, seed=seed, verbose=0)
                
                model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
                print(f"Completed: {run_id}")

if __name__ == "__main__":
    run_experiments()