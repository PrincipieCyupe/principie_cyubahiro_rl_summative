"""Build one comparison table from every experiment of every algorithm.

It reads the config.json that each run saved, collects the hyperparameters and
the greedy scores, writes logs/results_table.csv, and prints a markdown table.
This is the table for the report, made by a script so nothing is typed by hand.

Run it:
    uv run python results_table.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALGOS = ["dqn", "reinforce", "ppo", "ac"]
COLUMNS = ["algo", "exp", "lr", "gamma", "batch_size", "eps_decay", "target_update",
           "episodes", "greedy_avg_reward", "win_rate"]


def collect_rows():
    rows = []
    for algo in ALGOS:
        algo_dir = ROOT / "models" / algo
        if not algo_dir.is_dir():
            continue
        for folder in sorted(algo_dir.glob("exp*")):
            config_file = folder / "config.json"
            if not config_file.is_file():
                continue
            config = json.loads(config_file.read_text())
            rows.append({col: config.get(col, "") for col in COLUMNS})
    return rows


def write_csv(rows):
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / "results_table.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_markdown(rows):
    print("| " + " | ".join(COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[col]) for col in COLUMNS) + " |")


def main():
    rows = collect_rows()
    if not rows:
        print("No experiments found yet. Train some models first.")
        return
    path = write_csv(rows)
    print_markdown(rows)
    print(f"\nSaved table to {path}")


if __name__ == "__main__":
    main()
