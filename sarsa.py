"""Tile coding and true online Sarsa(lambda) for discrete-action environments."""

from __future__ import annotations

import numpy as np


class StateActionFeatureVectorWithTile:
    def __init__(self, state_low, state_high, num_actions, num_tilings, tile_width):
        self.state_low = np.asarray(state_low, dtype=float)
        self.state_high = np.asarray(state_high, dtype=float)
        self.num_actions = int(num_actions)
        self.num_tilings = int(num_tilings)
        self.tile_width = np.asarray(tile_width, dtype=float)
        self.num_tiles = np.ceil((self.state_high - self.state_low) / self.tile_width).astype(int) + 1
        self.tiles_per_tiling = int(np.prod(self.num_tiles))

    def feature_vector_len(self) -> int:
        return self.num_actions * self.num_tilings * self.tiles_per_tiling

    def __call__(self, state, done: bool, action: int) -> np.ndarray:
        features = np.zeros(self.feature_vector_len(), dtype=float)
        if done:
            return features
        state = np.asarray(state, dtype=float)
        offsets = (np.arange(self.num_tilings)[:, None] / self.num_tilings) * self.tile_width
        coordinates = np.floor((state - self.state_low + offsets) / self.tile_width).astype(int)
        coordinates = np.clip(coordinates, 0, self.num_tiles - 1)
        multipliers = np.concatenate(([1], np.cumprod(self.num_tiles[:-1])))
        indices = np.sum(coordinates * multipliers, axis=1)
        action_offset = action * self.num_tilings * self.tiles_per_tiling
        for tiling, index in enumerate(indices):
            features[action_offset + tiling * self.tiles_per_tiling + index] = 1.0
        return features

    def greedy_action(self, state, weights: np.ndarray) -> int:
        values = [weights @ self(state, False, action) for action in range(self.num_actions)]
        return int(np.argmax(values))


def train_true_online_sarsa(env, features, episodes: int, alpha: float, epsilon: float, seed: int, gamma: float = 1.0, lam: float = 0.9):
    """Train a true online Sarsa(lambda) agent and return weights and episode lengths."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.feature_vector_len(), dtype=float)
    episode_lengths = []

    def select_action(state):
        if rng.random() < epsilon:
            return int(rng.integers(features.num_actions))
        return features.greedy_action(state, weights)

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        action = select_action(state)
        traces = np.zeros_like(weights)
        q_old = 0.0
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
            x = features(state, False, action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = 0 if done else select_action(next_state)
            next_x = features(next_state, done, next_action)
            q = weights @ x
            next_q = weights @ next_x
            delta = reward + gamma * next_q - q
            traces = gamma * lam * traces + (1 - alpha * gamma * lam * (traces @ x)) * x
            weights += alpha * (delta + q - q_old) * traces - alpha * (q - q_old) * x
            q_old = next_q
            state, action = next_state, next_action
            steps += 1
        episode_lengths.append(steps)
    return weights, episode_lengths
