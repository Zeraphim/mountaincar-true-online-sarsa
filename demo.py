"""Train true online Sarsa(lambda) on MountainCar and record an evaluation GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image

from sarsa import StateActionFeatureVectorWithTile, train_true_online_sarsa


def record_policy(env: gym.Env, weights, features, output_path: Path, seed: int) -> int:
    state, _ = env.reset(seed=seed)
    frames = []
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        frames.append(Image.fromarray(env.render()))
        action = features.greedy_action(state, weights)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
    frames.append(Image.fromarray(env.render()))
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=30, loop=0)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_env = gym.make("MountainCar-v0")
    features = StateActionFeatureVectorWithTile(train_env.observation_space.low, train_env.observation_space.high, train_env.action_space.n, 8, (train_env.observation_space.high - train_env.observation_space.low) / 8)
    weights, episode_lengths = train_true_online_sarsa(train_env, features, args.episodes, 0.3 / 8, 0.05, args.seed)
    train_env.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(episode_lengths, linewidth=0.8, alpha=0.7, label="episode length")
    plt.plot(range(99, len(episode_lengths)), [sum(episode_lengths[i - 99 : i + 1]) / 100 for i in range(99, len(episode_lengths))], linewidth=2, label="100-episode mean")
    plt.xlabel("Training episode")
    plt.ylabel("Steps to termination")
    plt.title("True Online Sarsa(lambda) learns MountainCar")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "learning_curve.png", dpi=180)

    render_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    rollout_steps = record_policy(render_env, weights, features, args.output_dir / "mountaincar_policy.gif", args.seed)
    render_env.close()
    print(f"Saved artifacts. Evaluation rollout finished in {rollout_steps} steps.")


if __name__ == "__main__":
    main()
