"""Pick the best experiment for one algorithm by greedy play.

It loads every saved model for the algorithm, plays full seasons with no
exploration, ranks them by win rate and then average reward, and copies the
winner to models/<algo>/best_model/. That best model is the one the demo plays
and the one we compare across DQN, REINFORCE, PPO and A2C.

Run it, for example:
    uv run python select_best.py --algo dqn
"""

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import torch

import environment  # registers FloodGuardian-v0
from environment import ENV_ID
from training.common import qnetwork, evaluate, greedy_action_of, MODEL_FILE

ROOT = Path(__file__).resolve().parent


def env_sizes():
    env = gym.make(ENV_ID)
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env.close()
    return obs_size, num_actions


def load_net(folder, obs_size, num_actions):
    config = json.loads((folder / "config.json").read_text())
    hidden = config.get("hidden", 128)
    net = qnetwork(obs_size, num_actions, hidden=(hidden, hidden))
    net.load_state_dict(torch.load(folder / MODEL_FILE, map_location="cpu"))
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(description="Pick the best model for one algorithm.")
    parser.add_argument("--algo", default="dqn", help="dqn, reinforce, ppo or ac")
    parser.add_argument("--episodes", type=int, default=30, help="seasons to score each model on")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()

    algo_dir = ROOT / "models" / args.algo
    experiments = sorted(p for p in algo_dir.glob("exp*") if (p / MODEL_FILE).is_file())
    if not experiments:
        print(f"No saved models found in {algo_dir}")
        return

    obs_size, num_actions = env_sizes()
    results = []
    for folder in experiments:
        net = load_net(folder, obs_size, num_actions)
        avg, win = evaluate(greedy_action_of(net), episodes=args.episodes, seed=args.seed)
        results.append((folder.name, avg, win))
        print(f"{folder.name}: win rate {win:.2f}, average reward {avg:.2f}")

    results.sort(key=lambda r: (r[2], r[1]), reverse=True)
    best_name, best_avg, best_win = results[0]

    best_dir = algo_dir / "best_model"
    best_dir.mkdir(exist_ok=True)
    shutil.copy(algo_dir / best_name / MODEL_FILE, best_dir / MODEL_FILE)
    shutil.copy(algo_dir / best_name / "config.json", best_dir / "config.json")
    (best_dir / "best.json").write_text(json.dumps(
        {"algo": args.algo, "experiment": best_name, "win_rate": round(best_win, 3),
         "greedy_avg_reward": round(best_avg, 2), "episodes": args.episodes}, indent=2))

    print(f"\nBest for {args.algo}: {best_name} (win rate {best_win:.2f}, average reward {best_avg:.2f})")
    print(f"Copied to {best_dir}")


if __name__ == "__main__":
    main()
