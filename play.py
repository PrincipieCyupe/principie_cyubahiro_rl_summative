"""Play the game with a trained agent.

    uv run play.py                  the best DQN agent, 3D view in the browser
    uv run play.py --algo ppo       the best model of another algorithm
    uv run play.py --view 2d        the 2D window instead of the browser
    uv run play.py --episodes 5     play more seasons

It loads the best saved model of the chosen algorithm and plays full seasons,
printing every step to the terminal while the GUI shows the same thing.
"""

import argparse
import sys

from main import DemoSession


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained Flood Guardian agent play")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "ac"], default="dqn",
                        help="which algorithm's best model to play (dqn is the strongest)")
    parser.add_argument("--view", choices=["2d", "3d"], default="3d", help="which viewer to open")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to play before stopping")
    parser.add_argument("--seed", type=int, default=None, help="seed for a repeatable world")
    parser.add_argument("--speed", type=float, default=2.0, help="steps per second")
    parser.add_argument("--mute", action="store_true", help="turn off sound")
    parser.add_argument("--port", type=int, default=8000, help="local port for the 3D page")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    # DemoSession loads the best model of each trained algorithm on its own
    session = DemoSession(args.algo, args.episodes, args.seed)
    if args.algo not in session.agents:
        sys.exit(f"No trained {args.algo} model found. Train it and run select_best.py first.")
    session.new_episode()

    from environment.rendering import run_2d, run_3d
    if args.view == "2d":
        run_2d(session, args.algo, args.speed, args.mute)
    else:
        run_3d(session, args.algo, args.speed, args.mute, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
