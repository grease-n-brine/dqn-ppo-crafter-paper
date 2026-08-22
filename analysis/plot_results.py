import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "experiments/results"
OUTPUT_DIR = "analysis/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_and_plot():
    data = []
    
    for run_dir in glob.glob(os.path.join(RESULTS_DIR, "*")):
        if not os.path.isdir(run_dir):
            continue
        
        folder_name = os.path.basename(run_dir)
        parts = folder_name.split("_")
        algo = parts[0].upper()
        action_level = parts[1]
        
        eval_file = os.path.join(run_dir, "evaluations.npz")
        if os.path.exists(eval_file):
            eval_data = np.load(eval_file)
            timesteps = eval_data["timesteps"]
            results = eval_data["results"] # Shape: (eval_steps, n_eval_episodes)
            mean_rewards = results.mean(axis=1)
            
            for step, reward in zip(timesteps, mean_rewards):
                data.append({
                    "Algorithm": algo,
                    "Action Complexity": action_level,
                    "Timestep": step,
                    "Mean Reward": reward
                })

    df = pd.DataFrame(data)
    if df.empty:
        print("No evaluation data found yet. Run experiments first!")
        return

    # Plot Learning Curves
    plt.figure(figsize=(10, 6))
    for (algo, action_level), group in df.groupby(["Algorithm", "Action Complexity"]):
        avg_curve = group.groupby("Timestep")["Mean Reward"].mean()
        plt.plot(avg_curve.index, avg_curve.values, label=f"{algo} ({action_level})")
        
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title("RL Learning Curves Across Action Space Complexity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"), dpi=300)
    print(f"Saved plots to {OUTPUT_DIR}/learning_curves.png")

if __name__ == "__main__":
    parse_and_plot()