import gymnasium as gym
import numpy as np

class DecreasingStochasticityWrapper(gym.Wrapper):
    """
    A wrapper for MiniGrid (or any discrete-action Gym env) that injects action noise
    which decreases over time.

    With probability p_noise, the agent's chosen action is replaced by a random action.
    p_noise decays linearly from p_start to p_end over decay_steps environment steps.
    """

    def __init__(self, env, p_start=0.75, p_end=0.0, decay_steps=5_000, seed=None):
        super().__init__(env)
        self.p_start = p_start
        self.p_end = p_end
        self.decay_steps = decay_steps
        self.total_steps = 0
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        frac = min(self.total_steps / self.decay_steps, 1.0)
        p_noise = self.p_start + frac * (self.p_end - self.p_start)

        if self.rng.random() < p_noise:
            action = self.env.action_space.sample()

        self.total_steps += 1
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)