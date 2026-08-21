import gymnasium as gym
from gymnasium import spaces
import numpy as np
import crafter

class CrafterNavigationEnv(gym.Env):
    """
    Custom Gym Wrapper around Crafter to study RL performance 
    under varying action space complexities.
    """
    def __init__(self, action_level="simple", max_steps=300):
        super().__init__()
        self.env = crafter.Env()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space configurations using Crafter's discrete actions
        self.action_sets = {
            "simple": [1, 2, 3, 4],                            # Left, Right, Up, Down
            "moderate": [0, 1, 2, 3, 4, 5, 6],                 # + No-op, Sleep, Place Stone
            "complex": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # + Crafting / Mining
        }
        
        self.active_actions = self.action_sets[action_level]
        self.action_space = spaces.Discrete(len(self.active_actions))
        
        # State vector: s_t = [delta_x, delta_y]
        self.observation_space = spaces.Box(
            low=-64.0, high=64.0, shape=(2,), dtype=np.float32
        )
        
        self.goal_pos = None
        self.goal_radius = 1.5

    def _get_agent_pos(self):
        return np.array(self.env._player.pos, dtype=np.float32)

    def _get_obs(self):
        agent_pos = self._get_agent_pos()
        delta = self.goal_pos - agent_pos
        return delta.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
            
        self.current_step = 0
        agent_pos = self._get_agent_pos()
        
        # Deterministic relative goal position (+10, +10 blocks away)
        self.goal_pos = agent_pos + np.array([10.0, 10.0], dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action_idx):
        self.current_step += 1
        
        # Map sub-action index to actual Crafter action enum
        mapped_action = self.active_actions[action_idx]
        
        prev_dist = np.linalg_norm(self._get_obs())
        
        # Execute action in Crafter engine
        _, reward_crafter, done, info = self.env.step(mapped_action)
        
        curr_dist = np.linalg_norm(self._get_obs())
        
        # Distance-reduction reward + sparse completion bonus
        r_t = (prev_dist - curr_dist)
        
        terminated = False
        if curr_dist < self.goal_radius:
            r_t += 50.0  # Success bonus
            terminated = True
            
        truncated = self.current_step >= self.max_steps or done

        return self._get_obs(), float(r_t), terminated, truncated, {"success": terminated}