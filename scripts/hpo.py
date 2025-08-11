import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

import numpy as np
import random
import time
import trackio as trackio

from src.dynaq import DynaQAgent, DynaQBAgent

def train(config=None, seed=42):
    """
    Training function to run the Dyna-Q agent in a Minigrid environment.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # --- Configuration ---
    ENV_NAME = "MiniGrid-Empty-6x6-v0"
    NUM_STEPS = 5_000
    WARMUP_STEPS = 1_500
    RENDER = False
    ENSEMBLE_SIZE=5

    # --- Environment Setup ---
    # Base environment
    env = gym.make(ENV_NAME, render_mode="human" if RENDER else None)
    env = ImgObsWrapper(FullyObsWrapper(env))
    env.action_space.seed(seed)
    env.reset(seed=seed)

    # --- Agent Initialization ---
    agent = DynaQBAgent(
        env,
        learning_rate=config['learning_rate'],
        discount_factor=0.9999,
        epsilon=1.0,
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=0.05,
        planning_steps=config['planning_steps'],
        ensemble_size=ENSEMBLE_SIZE,
        warmup_steps=WARMUP_STEPS,
        kappa=config['kappa']
    )

    print(f"Starting training on {ENV_NAME} with config: {config}")

    # --- Training Loop ---
    steps = 0
    episode = 0
    all_episode_rewards = []
    
    while steps < NUM_STEPS:
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward_episode = 0
        episode_length = 0

        while not terminated and not truncated and steps < NUM_STEPS:
            if RENDER:
                env.render()
                time.sleep(0.01)

            obs, reward, terminated, truncated, info = agent.learn(obs)
            total_reward_episode += reward
            steps += 1
            episode_length += 1

        episode += 1

        if terminated or truncated: # check if complete episode
            all_episode_rewards.append(total_reward_episode)
            print(f"Episode {episode} | Step {steps}/{NUM_STEPS} | Reward: {total_reward_episode:.2f}")

    env.close()
    
    return np.sum(all_episode_rewards)

def main():
    """
    Main function to run Hyperparameter Optimization.
    """
    num_runs = 250
    results = []

    for i in range(num_runs):
        # Randomly sample hyperparameters
        learning_rate = random.uniform(0.1, 0.5)
        kappa = random.choice([0.5, 0.75, 1])

        # Train with the sampled hyperparameters
        hparams = {
            'learning_rate': learning_rate,
            'planning_steps': 10,
            'epsilon_decay': 0.99975,
            'kappa': kappa
        }
        
        avg_reward = train(config=hparams)
        results.append({
            'avg_reward': avg_reward,
            'hyperparameters': hparams
        })

    # Find the best set of hyperparameters based on the highest reward
    best_result = max(results, key=lambda x: x['avg_reward'])

    # Print the best set of hyperparameters and its corresponding average reward
    print("Best Hyperparameters:")
    import json
    print(json.dumps(best_result['hyperparameters'], indent=4))
    print(f"Best Average Reward: {best_result['avg_reward']}")


if __name__ == "__main__":
    main()