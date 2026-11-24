import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import environment  # registers FloodGuardian-v0
from environment import ENV_ID
from torchrl.envs import GymEnv, TransformedEnv, StepCounter, InitTracker
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import Collector
from torchrl.objectives import ReinforceLoss, A2CLoss, ClipPPOLoss, ValueEstimators
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

from training.common import qnetwork, evaluate, greedy_action_of

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "report"
OUT.mkdir(parents=True, exist_ok=True)

# the best experiment of each algorithm (from models/<algo>/best_model/best.json)
BEST = {"dqn": 10, "reinforce": 1, "ppo": 3, "ac": 2}

COLORS = {"dqn": "#1f6f3d", "reinforce": "#b5553c", "ppo": "#2e6fa7", "ac": "#c8641e"}
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "ac": "A2C"}
LN5 = math.log(5)


def read_history(algo, exp):
    rows = list(csv.DictReader(open(ROOT / "models" / algo / f"exp{exp:02d}" / "history.csv")))
    return [float(r["return"]) for r in rows], [float(r["loss"]) for r in rows]


def moving_average(values, window):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run_pg_entropy(method, cfg):
    """Re-run one policy gradient experiment and record the policy entropy per update."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    env = TransformedEnv(GymEnv(ENV_ID, device="cpu"), StepCounter())
    env.append_transform(InitTracker())
    env.set_seed(cfg["seed"])

    obs_size = env.observation_spec["observation"].shape[-1]
    num_actions = env.action_spec.space.n
    hidden = cfg["hidden"]

    policy_net = qnetwork(obs_size, num_actions, hidden=(hidden, hidden))
    policy_module = TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(module=policy_module, in_keys=["logits"], out_keys=["action"],
                               distribution_class=Categorical, return_log_prob=True)
    value_net = qnetwork(obs_size, 1, hidden=(hidden, hidden))
    critic = ValueOperator(value_net, in_keys=["observation"])
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=True)

    if method == "reinforce":
        loss_module = ReinforceLoss(actor, critic_network=critic)
        loss_module.make_value_estimator(ValueEstimators.GAE, gamma=cfg["gamma"], lmbda=cfg["lmbda"])
    elif method == "ac":
        loss_module = A2CLoss(actor, critic, entropy_bonus=cfg["entropy"] > 0, entropy_coeff=cfg["entropy"])
        loss_module.make_value_estimator(ValueEstimators.GAE, gamma=cfg["gamma"], lmbda=cfg["lmbda"])
    else:
        loss_module = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"],
                                  entropy_bonus=cfg["entropy"] > 0, entropy_coeff=cfg["entropy"])

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    collector = Collector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_steps"], device="cpu")

    entropies = []
    for data in collector:
        batch = data.reshape(-1)
        advantage(batch)
        with torch.no_grad():
            entropies.append(float(actor.get_dist(batch).entropy().mean().item()))
        epochs = cfg.get("epochs", 1) if method == "ppo" else 1
        for _ in range(epochs):
            losses = loss_module(batch)
            if method == "reinforce":
                ent = actor.get_dist(batch).entropy().mean()
                loss = losses["loss_actor"] + losses["loss_value"] - cfg["entropy"] * ent
            else:
                loss = losses["loss_objective"] + losses["loss_critic"]
                if "loss_entropy" in losses.keys():
                    loss = loss + losses["loss_entropy"]
            optimizer.zero_grad()
            loss.backward()
            if method == "ppo":
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
            optimizer.step()
        if method in ("ppo", "ac"):
            collector.update_policy_weights_()
    collector.shutdown()
    return entropies


def load_net(algo, exp):
    folder = ROOT / "models" / algo / f"exp{exp:02d}"
    hidden = json.loads((folder / "config.json").read_text()).get("hidden", 128)
    net = qnetwork(14, 5, hidden=(hidden, hidden))
    net.load_state_dict(torch.load(folder / "model.pth", map_location="cpu"))
    net.eval()
    return net


def fig_cumulative_rewards():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, algo in zip(axes.flat, ["dqn", "reinforce", "ppo", "ac"]):
        returns, _ = read_history(algo, BEST[algo])
        ax.plot(returns, color=COLORS[algo], alpha=0.25, linewidth=0.8)
        ma = moving_average(returns, 50)
        ax.plot(range(len(returns) - len(ma), len(returns)), ma, color=COLORS[algo], linewidth=2)
        ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
        ax.set_title(f"{LABELS[algo]} (best model)", fontsize=11)
        ax.set_xlabel("episode")
        ax.set_ylabel("episode return")
        ax.grid(alpha=0.2)
    fig.suptitle("Cumulative reward per episode, best model of each algorithm", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "cumulative_rewards.png", dpi=130)
    plt.close(fig)


def fig_dqn_loss():
    returns, loss = read_history("dqn", BEST["dqn"])
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ma_loss = moving_average(loss, 50)
    ax1.plot(loss, color="#9fb6c9", linewidth=0.7, alpha=0.5, label="TD loss (raw)")
    ax1.plot(range(len(loss) - len(ma_loss), len(loss)), ma_loss, color="#17457a", linewidth=2, label="TD loss (moving avg)")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("temporal difference loss")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ma_ret = moving_average(returns, 50)
    ax2.plot(range(len(returns) - len(ma_ret), len(returns)), ma_ret, color=COLORS["dqn"], linewidth=2, label="return (moving avg)")
    ax2.set_ylabel("episode return", color=COLORS["dqn"])
    ax1.set_title("DQN training stability: loss and return over training")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "dqn_loss.png", dpi=130)
    plt.close(fig)


def fig_pg_entropy(entropy_data):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for method in ["reinforce", "ppo", "ac"]:
        ent = entropy_data[method]
        ax.plot(np.linspace(0, 100, len(ent)), ent, color=COLORS[method], linewidth=2, label=LABELS[method])
    ax.axhline(LN5, color="#888888", linewidth=0.8, linestyle="--")
    ax.text(1, LN5 - 0.05, "max entropy ln(5) = 1.61 (uniform policy)", fontsize=8, color="#555555")
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Policy entropy over training, policy gradient methods")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "pg_entropy.png", dpi=130)
    plt.close(fig)


def fig_convergence():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cross = None
    for algo in ["dqn", "reinforce", "ppo", "ac"]:
        returns, _ = read_history(algo, BEST[algo])
        ma = moving_average(returns, 50)
        xs = np.linspace(0, 100, len(ma))
        ax.plot(xs, ma, color=COLORS[algo], linewidth=2, label=LABELS[algo])
        if algo == "dqn":
            for i, v in enumerate(ma):
                if v > 0:
                    cross = xs[i]
                    break
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    if cross is not None:
        ax.axvline(cross, color=COLORS["dqn"], linewidth=1, linestyle=":")
        ax.text(cross + 1, ax.get_ylim()[0] + 8, f"DQN turns positive\nat about {cross:.0f}% of training",
                fontsize=8, color=COLORS["dqn"])
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel("moving average return (50 episodes)")
    ax.set_title("Convergence: smoothed return across training")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=130)
    plt.close(fig)


def fig_generalization():
    results = {}
    for algo in ["dqn", "reinforce", "ppo", "ac"]:
        avg, win = evaluate(greedy_action_of(load_net(algo, BEST[algo])), episodes=30, seed=5000)
        results[algo] = (avg, win)
        print(f"  generalization {algo}: avg {avg:.2f}, win {win:.2f}")
    algos = ["dqn", "reinforce", "ppo", "ac"]
    names = [LABELS[a] for a in algos]
    cols = [COLORS[a] for a in algos]
    avgs = [results[a][0] for a in algos]
    wins = [results[a][1] * 100 for a in algos]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4))
    axl.bar(names, avgs, color=cols)
    axl.axhline(0, color="#888888", linewidth=0.7)
    axl.set_ylabel("average return (30 unseen seasons)")
    axl.set_title("Generalization: average return on unseen seeds")
    axl.grid(alpha=0.2, axis="y")
    for i, v in enumerate(avgs):
        axl.text(i, v + (2 if v >= 0 else -4), f"{v:.1f}", ha="center", fontsize=9)
    axr.bar(names, wins, color=cols)
    axr.set_ylabel("win rate (%)")
    axr.set_ylim(0, 105)
    axr.set_title("Generalization: win rate on unseen seeds")
    axr.grid(alpha=0.2, axis="y")
    for i, v in enumerate(wins):
        axr.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "generalization.png", dpi=130)
    plt.close(fig)


def main():
    print("cumulative rewards ...")
    fig_cumulative_rewards()
    print("dqn loss ...")
    fig_dqn_loss()
    print("convergence ...")
    fig_convergence()

    print("policy entropy re-runs (a few minutes) ...")
    cfgs = {
        "reinforce": {"lr": 1e-3, "gamma": 0.99, "frames_per_batch": 1000, "lmbda": 1.0,
                      "entropy": 0.01, "total_steps": 60000, "hidden": 128, "seed": 42},
        "ppo": {"lr": 1e-4, "gamma": 0.99, "clip": 0.2, "epochs": 4, "frames_per_batch": 2000,
                "lmbda": 0.95, "entropy": 0.01, "total_steps": 60000, "hidden": 128, "seed": 42},
        "ac": {"lr": 3e-3, "gamma": 0.99, "frames_per_batch": 500, "lmbda": 0.95,
               "entropy": 0.01, "total_steps": 60000, "hidden": 128, "seed": 42},
    }
    entropy_data = {m: run_pg_entropy(m, cfgs[m]) for m in ["reinforce", "ppo", "ac"]}
    fig_pg_entropy(entropy_data)

    print("generalization ...")
    fig_generalization()
    print(f"done, figures in {OUT}")


if __name__ == "__main__":
    main()
