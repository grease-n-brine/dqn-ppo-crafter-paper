# train_baseline.py
from stable_baselines3 import DQN, PPO
from environment.nav_env import CrafterNavigationEnv

def run_crafter_baseline():
    print("--- Verifying Crafter Integration with SB3 ---")
    
    # 1. Instantiate Crafter Environment
    env = CrafterNavigationEnv(action_level="simple")
    
    # 2. Test DQN
    print("\n[1/2] Training Baseline DQN on Crafter...")
    dqn_model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3, seed=42)
    dqn_model.learn(total_timesteps=5000)
    print("DQN Training Complete!")
    
    # 3. Test PPO
    print("\n[2/2] Training Baseline PPO on Crafter...")
    ppo_model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3, seed=42)
    ppo_model.learn(total_timesteps=5000)
    print("PPO Training Complete!")

if __name__ == "__main__":
    run_crafter_baseline()