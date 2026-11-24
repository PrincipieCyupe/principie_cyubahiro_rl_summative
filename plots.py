# plots.py, a script to generate the useful plots for the report

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "screenshots"
ALGOS = ["dqn", "reinforce", "ppo", "ac"]
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "ac": "A2C"}
COLORS = {"dqn": "#1f6f3d", "reinforce": "#b5553c", "ppo": "#2e6fa7", "ac": "#c8641e"}


def read_best(algo):
    """Read the best.json that select_best saved for an algorithm, or None if it is missing."""
    path = ROOT / "models" / algo / "best_model" / "best.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_history(algo, experiment):
    """Read the per episode reward from one experiment's history.csv."""
    path = ROOT / "models" / algo / experiment / "history.csv"
    returns = []
    with open(path) as f:
        for row in csv.DictReader(f):
            returns.append(float(row["return"]))
    return returns


def moving_average(values, window=50):
    """Smooth a noisy reward curve so the trend is easy to see."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def draw_win_rate(ax, algos, bests):
    wins = [bests[a]["win_rate"] for a in algos]
    ax.bar([LABELS[a] for a in algos], wins, color=[COLORS[a] for a in algos])
    ax.set_ylim(0, 1)
    ax.set_ylabel("win rate (share of seasons survived)")
    ax.set_title("Best model win rate by algorithm")
    for i, w in enumerate(wins):
        ax.text(i, w + 0.02, f"{w:.2f}", ha="center")


def draw_curves(ax, algos, bests):
    for algo in algos:
        history = read_history(algo, bests[algo]["experiment"])
        smooth = moving_average(history, 50)
        progress = np.linspace(0, 100, len(smooth))  # percent of training done
        ax.plot(progress, smooth, color=COLORS[algo], label=LABELS[algo], linewidth=2)
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel("reward per season (smoothed)")
    ax.set_title("Learning curve of each algorithm's best experiment")
    ax.legend()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bests = {a: read_best(a) for a in ALGOS}
    bests = {a: b for a, b in bests.items() if b is not None}
    algos = [a for a in ALGOS if a in bests]
    if not algos:
        print("No best models found. Run select_best.py for each algorithm first.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_win_rate(ax, algos, bests)
    fig.tight_layout(); fig.savefig(OUT / "win_rate.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    draw_curves(ax, algos, bests)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    draw_win_rate(axes[0], algos, bests)
    draw_curves(axes[1], algos, bests)
    fig.tight_layout(); fig.savefig(OUT / "results.png", dpi=130); plt.close(fig)

    print(f"Saved win_rate.png, learning_curves.png and results.png in {OUT}")


if __name__ == "__main__":
    main()
