import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

import numpy as np
import random
import time
import trackio as trackio

from src.dynaq import DynaQAgent, DynaQBAgent
from src.wrappers import DecreasingStochasticityWrapper

def train(config=None, seed=42, logs=False):
    """
    Training function to run the Dyna-Q agent in a Minigrid environment.
    If logs are activated, the training metrics get logged via trackio.
    """
    # Setup trackio logging project
    if logs:
        run = trackio.init(project="DYNAQ-BETA", config=config, name=f"run_{seed}")
        config = trackio.config
    else:
        config = config

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
    env = gym.make(ENV_NAME, render_mode="human" if RENDER else None)
    env = ImgObsWrapper(FullyObsWrapper(env))
    env = DecreasingStochasticityWrapper(env)
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
        #kappa=0.5,
    )

    print(f"Starting training on {ENV_NAME} with config: {config} and seed: {seed}")

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
            if logs:
            # Log episode metrics
                trackio.log({
                    "episode_reward": total_reward_episode,
                    "episode_length": episode_length,
                    "total_steps": steps,
                    "episode": episode
                })
            print(f"Episode {episode} | Step {steps}/{NUM_STEPS} | Reward: {total_reward_episode:.2f}")

    env.close()

    if logs:
        run.finish()

def main():
    """
    Main function to run Training.
    """

    config = {
        'learning_rate': 0.1,
        'planning_steps': 10,
        'epsilon_decay': 0.99975,
    }

    # Single Run
    train(config=config, seed=2, logs=False)

    # Runs over multiple seeds with logging
    #num_seeds = 20
    #for seed in range(num_seeds):
    #    train(config=config, seed=seed, logs=True)


if __name__ == "__main__":
    main()