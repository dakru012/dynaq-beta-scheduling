import numpy as np
import random
from collections import defaultdict

class BaseModel:
    """
    Base class for a world model used by Dyna-Q agents.

    Stores observed transitions and provides a way to sample from them for planning.
    Subclasses must implement `learn` and `predict`.

    Attributes:
        T (defaultdict): Nested dictionary storing transitions:
                         T[state][action] = list of (reward, next_state) tuples.
        observed_states (set): Set of states observed so far.
    """

    def __init__(self):
        self.T = defaultdict(lambda: defaultdict(list))
        self.observed_states = set()

    def learn(self, state, action, reward, next_state):
        """
        Record a real experience tuple (s, a, r, s').

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def sample(self):
        """
        Sample a random previously observed state and an action taken from it.

        Returns:
            tuple: (state, action)
        """
        state = random.choice(list(self.observed_states))
        action = random.choice(list(self.T[state].keys()))
        return state, action

    def predict(self, state, action):
        """
        Predict the next state and reward for a given (state, action) pair.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class TransitionCountsModel(BaseModel):
    """
    Tabular transition model using empirical counts.

    Tracks:
        - counts of next states for each (state, action)
        - rewards observed for (state, action, next_state)

    Can return empirical or Dirichlet-sampled transitions for planning.
    """

    def __init__(self):
        super().__init__()
        self.counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.rewards = defaultdict(list)

    def learn(self, state, action, reward, next_state):
        """
        Record an observed transition into the model.

        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Observed next state.
        """
        self.observed_states.add(state)
        self.counts[state][action][next_state] += 1
        self.rewards[(state, action, next_state)].append(reward)
        self.T[state][action].append((reward, next_state))

    def predict(self, state, action, sample_from_dirichlet=False, alpha=1.0):
        """
        Predict the next state and reward for a given (state, action).

        Args:
            state: Current state.
            action (int): Action taken.
            sample_from_dirichlet (bool): If True, sample from a Dirichlet posterior
                                          over next-state probabilities.
            alpha (float): Dirichlet concentration parameter.

        Returns:
            tuple: (reward, next_state)
        """
        if state not in self.counts or action not in self.counts[state]:
            return 0.0, state  # No data; return zero reward and same state.

        next_states = list(self.counts[state][action].keys())
        counts = np.array([self.counts[state][action][ns] for ns in next_states], dtype=float)

        if sample_from_dirichlet:
            probs = np.random.dirichlet(counts + alpha)
        else:
            probs = counts / counts.sum()

        next_state = random.choices(next_states, weights=probs, k=1)[0]

        rewards_list = self.rewards[(state, action, next_state)]
        reward = np.mean(rewards_list) if rewards_list else 0.0

        return reward, next_state

