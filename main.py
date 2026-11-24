"""Flood Guardian demo.

    uv run main.py                              3D view in the browser, random agent
    uv run main.py --mode rule_based            3D view, the hand written guardian that wins
    uv run main.py --mode dqn                   3D view, the best DQN agent plays
    uv run main.py --mode keyboard              3D view, you drive the guardian
    uv run main.py --view 2d --mode dqn         2D pygame window, the best DQN agent plays
    uv run main.py --model dqn/exp05            add one more trained agent to the picker

In the 3D view the dropdown switches between random, rule_based, keyboard and every trained
agent that was loaded. The best model of each algorithm is loaded automatically.
"""

import argparse
import sys
from pathlib import Path

from environment.custom_env import FloodGuardianEnv, ACTION_NAMES
from environment.baseline import rule_based_action

ROOT = Path(__file__).resolve().parent
ALGOS = ["dqn", "reinforce", "ppo", "ac"]


def load_policy(model_path, env):
    """Load a trained network from a saved model.pth so it can pick actions.

    Every algorithm saves the same kind of small network, one that reads the observation and
    gives a value for each action. The hidden size is read from the config.json saved next to
    the model, so the shape always matches.
    """
    import json
    import torch
    from training.common import qnetwork

    folder = Path(model_path).parent
    hidden = 128
    config = folder / "config.json"
    if config.exists():
        hidden = json.loads(config.read_text()).get("hidden", 128)
    net = qnetwork(env.observation_space.shape[0], env.action_space.n, hidden=(hidden, hidden))
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net


def model_action(net, env):
    """Return a function that picks the best action for the current observation, no randomness."""
    import torch

    def pick():
        obs = env.observation()
        with torch.no_grad():
            values = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(values.argmax(dim=1).item())

    return pick


def resolve_model(spec):
    """Work out an agent name and a model.pth path from one --model value.

    Accepts a path to the model file, a path to its folder, a short 'algo/expNN', or
    'name=path' to choose the name yourself.
    """
    name = None
    if "=" in spec:
        name, spec = spec.split("=", 1)
    path = Path(spec)
    if path.is_dir():
        path = path / "model.pth"
    elif path.suffix != ".pth":
        path = ROOT / "models" / spec / "model.pth"
    if name is None:
        name = f"{path.parent.parent.name}_{path.parent.name}"
    return name, path


class DemoSession:
    """Holds the environment and the episode count, and prints every step to the terminal."""

    def __init__(self, mode, episodes, seed, extra_models=()):
        self.env = FloodGuardianEnv()
        self.mode = mode
        self.episodes = episodes
        self.seed = seed
        self.episode = 0
        self.results = []  # (status, total reward) of every finished episode in this run
        if seed is not None:
            self.env.action_space.seed(seed)

        # name -> function that picks the next action
        self.agents = {
            "random": self.random_action,
            "rule_based": lambda: rule_based_action(self.env),
        }
        # the best trained model of each algorithm, if it has been trained
        for algo in ALGOS:
            path = ROOT / "models" / algo / "best_model" / "model.pth"
            if path.exists():
                self.agents[algo] = model_action(load_policy(path, self.env), self.env)
        # any extra models passed with --model on the command line
        for spec in extra_models:
            name, path = resolve_model(spec)
            if path.exists():
                self.agents[name] = model_action(load_policy(path, self.env), self.env)
                print(f"Loaded agent {name} from {path}")
            else:
                print(f"Could not find a model at {path}, skipping {spec}")

        self.agent = mode if mode in self.agents else "random"

    @property
    def done(self):
        return self.env.status != "running"

    def set_agent(self, name):
        if name in self.agents:
            self.agent = name
            print(f"Agent switched to {name}")

    def agent_action(self):
        return self.agents[self.agent]()

    def new_episode(self):
        self.episode += 1
        seed = None if self.seed is None else self.seed + self.episode - 1
        self.env.reset(seed=seed)
        print()
        print(f"===== Episode {self.episode} | agent: {self.agent} | goal: survive {self.env.max_steps} steps "
              f"without losing a house or half the crops =====")

    def start_over(self):
        # a fresh run: episode numbers and results start again from zero
        self.episode = 0
        self.results = []
        self.new_episode()

    def random_action(self):
        return int(self.env.action_space.sample())

    def take_step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        events = ", ".join(e for e in info["events"] if e not in ("move", "scare")) or "-"
        print(f"Episode {self.episode} | step {info['step']:3d} | action {ACTION_NAMES[action]:5s} | "
              f"reward {reward:+6.2f} | total {self.env.total_reward:+7.2f} | "
              f"status {info['status']:7s} | events {events}")
        if terminated or truncated:
            if info["status"] == "win":
                result = "WIN, the village survived the season"
            elif "no_crops" in info["events"]:
                result = "FAIL, half the harvest was eaten"
            else:
                result = "FAIL, an animal reached the village"
            print(f"----- Episode {self.episode} over: {result}. Total reward {self.env.total_reward:+.2f} "
                  f"in {info['step']} steps, crops left {info['crops_left']} -----")
            self.results.append((info["status"], float(self.env.total_reward)))
            if len(self.results) == self.episodes:
                self.print_summary()

    def print_summary(self):
        wins = sum(status == "win" for status, _ in self.results)
        totals = [total for _, total in self.results]
        print()
        print(f"===== Run finished: {len(self.results)} episodes with agent {self.agent} | "
              f"wins {wins} | losses {len(self.results) - wins} | "
              f"mean reward {sum(totals) / len(totals):+.2f} | best {max(totals):+.2f} =====")


def parse_args():
    parser = argparse.ArgumentParser(description="Flood Guardian environment demo")
    parser.add_argument("--view", choices=["2d", "3d"], default="3d", help="which viewer to open")
    parser.add_argument("--mode", default="random",
                        help="who controls the guardian at the start: random, rule_based, keyboard, "
                             "a trained agent (dqn, reinforce, ppo, ac), or a name added with --model")
    parser.add_argument("--model", action="append", default=[], metavar="SPEC",
                        help="add a trained agent to the picker. SPEC is a path to a model.pth, its "
                             "folder, a short algo/expNN like dqn/exp05, or name=path. Repeat to add more")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to play before stopping")
    parser.add_argument("--seed", type=int, default=None, help="seed for a repeatable world")
    parser.add_argument("--speed", type=float, default=2.0, help="steps per second when an agent plays")
    parser.add_argument("--mute", action="store_true", help="turn off sound")
    parser.add_argument("--port", type=int, default=8000, help="local port for the 3D page")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # print each step straight away, even when piped
    args = parse_args()
    session = DemoSession(args.mode, args.episodes, args.seed, args.model)
    session.new_episode()

    # imported here so nothing graphical loads unless a window is actually opened
    from environment.rendering import run_2d, run_3d
    if args.view == "2d":
        run_2d(session, args.mode, args.speed, args.mute)
    else:
        run_3d(session, args.mode, args.speed, args.mute, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
