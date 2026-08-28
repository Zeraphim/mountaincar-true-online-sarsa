# MountainCar with True Online Sarsa(lambda)

A from-scratch reinforcement-learning implementation that learns a tile-coded action-value function for Gymnasium's MountainCar environment.

## What it does

The car must build momentum by first driving away from the hill, then reversing uphill to reach the goal. The agent learns from reward feedback using true online Sarsa(lambda), eligibility traces, epsilon-greedy exploration, and tile-coded state-action features.

## Run the demo

```bash
conda create -n mountaincar-rl python=3.12 -y
conda activate mountaincar-rl
pip install -r requirements.txt
python demo.py --episodes 8000
```

The command saves:

- `artifacts/learning_curve.png` — episode length during learning;
- `artifacts/mountaincar_policy.gif` — a newly recorded evaluation rollout.

Set `--seed` for repeatability.

## Current training run

The committed assets were generated with `python demo.py --episodes 8000`. The run includes successful training episodes, but its single deterministic evaluation rollout timed out at MountainCar's 200-step limit. It is included as a reproducible baseline, not evidence of reliable task completion.

![Learning curve for the current 8,000-episode run](assets/learning_curve.png)

![Current deterministic evaluation rollout](assets/mountaincar_policy.gif)

## Implementation

- `sarsa.py` implements tile coding and true online Sarsa(lambda).
- `demo.py` trains the agent, plots episode length, and records its learned policy.

## LinkedIn demo

Use a verified successful rollout beside the learning curve. Open with two seconds of an untrained car failing, then show the learned car building momentum and reaching the flag. State the environment and learning rule on screen.

## Limitations

MountainCar is a small benchmark environment. This project demonstrates the mechanics of online control learning, not a production control system.
