"""Shared helpers for all four trainers (DQN, REINFORCE, PPO, A2C).

It keeps the small network, the greedy test used to score a model, and the
code that saves one experiment, so every trainer saves its results the same way.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import environment  # importing this registers FloodGuardian-v0
from environment import ENV_ID

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = "model.pth"


def qnetwork(obs_size, num_actions, hidden=(128, 128)):
    """A small feed forward network. It reads the 14 observation numbers and gives
    one value for each of the 5 actions. No images here, so no need for a CNN."""
    layers = []
    size = obs_size
    for width in hidden:
        layers.append(nn.Linear(size, width))
        layers.append(nn.ReLU())
        size = width
    layers.append(nn.Linear(size, num_actions))
    return nn.Sequential(*layers)


def evaluate(action_of, episodes=20, seed=1000):
    """Play full seasons with no exploration and report how the model really does.

    action_of is a function that takes one observation and returns an action.
    Returns the average total reward and the win rate (seasons survived).
    """
    env = gym.make(ENV_ID)
    rewards, wins = [], 0
    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(action_of(obs))
            total += reward
            done = terminated or truncated
        rewards.append(total)
        if info.get("status") == "win":
            wins += 1
    env.close()
    return float(np.mean(rewards)), wins / episodes


def greedy_action_of(net):
    """Turn a trained q network into a simple greedy action function for evaluate()."""
    def action_of(obs):
        with torch.no_grad():
            values = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(values.argmax(dim=1).item())
    return action_of


def experiment_dir(algo, exp):
    folder = ROOT / "models" / algo / f"exp{int(exp):02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_plot(history, path, title):
    """Draw the reward per episode plus a moving average, so the run is easy to read."""
    rewards = [row["return"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(rewards, color="#9bbcd8", linewidth=1, label="episode reward")
    if len(rewards) >= 10:
        window = 10
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards)), avg, color="#17457a", linewidth=2, label="moving average")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_experiment(algo, exp, net, history, config, greedy_avg, win_rate):
    """Save everything for one experiment in models/<algo>/exp<nn>/ and log one line."""
    folder = experiment_dir(algo, exp)

    torch.save(net.state_dict(), folder / MODEL_FILE)

    fieldnames = list(history[0].keys()) if history else ["episode", "return", "loss"]
    with open(folder / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    config = dict(config)
    config["episodes"] = len(history)
    config["greedy_avg_reward"] = round(greedy_avg, 2)
    config["win_rate"] = round(win_rate, 3)
    with open(folder / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_plot(history, folder / "training_plot.png", f"{algo.upper()} experiment {int(exp)}")

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    line = f"{algo},exp{int(exp):02d},episodes={len(history)},greedy_avg={greedy_avg:.2f},win_rate={win_rate:.2f}\n"
    with open(logs / "runs.csv", "a") as f:
        f.write(line)

    return folder
