"""Train one REINFORCE experiment on the Flood Guardian environment with TorchRL.

REINFORCE is policy based and Monte Carlo. The network reads the observation and gives a
probability for each action, the agent samples an action from those probabilities, and at
the end of each episode the actions that led to higher reward are made more likely. A small
value network is used as a baseline to make the updates less noisy. TorchRL gives us the
ready parts, and this file just wires them together:

    ProbabilisticActor   the policy network that samples an action (our qnetwork as logits)
    ValueOperator        the baseline value network (our qnetwork with one output)
    GAE                  turns the episode rewards into an advantage for each step
    ReinforceLoss        the policy gradient loss plus the value loss
    Collector            plays the env and collects whole episodes

Run one experiment, for example:
    uv run python training/reinforce_training.py --exp 1 --lr 1e-3 --gamma 0.99
"""

import argparse
import sys
from pathlib import Path

# let "uv run python training/reinforce_training.py" find the environment and training packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.distributions import Categorical

import environment  # registers FloodGuardian-v0
from environment import ENV_ID
from torchrl.envs import GymEnv, TransformedEnv, StepCounter, InitTracker
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import Collector
from torchrl.objectives import ReinforceLoss, ValueEstimators
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

from training.common import qnetwork, evaluate, greedy_action_of, save_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Train one REINFORCE experiment.")
    p.add_argument("--exp", type=int, required=True, help="experiment number, saved in models/reinforce/exp<nn>")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--frames-per-batch", type=int, default=1_000, help="steps collected before each update")
    p.add_argument("--lmbda", type=float, default=1.0, help="1.0 is pure Monte Carlo, lower adds some bootstrap") # since REINFORCE is pure monte carlo algorithm
    p.add_argument("--entropy", type=float, default=0.01, help="entropy bonus to keep the policy exploring")
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # the environment, with a step counter and an init flag
    env = TransformedEnv(GymEnv(ENV_ID, device=device), StepCounter())
    env.append_transform(InitTracker())
    env.set_seed(args.seed)

    obs_size = env.observation_spec["observation"].shape[-1]
    num_actions = env.action_spec.space.n

    # the policy: the network gives logits, the actor samples an action from them
    policy_net = qnetwork(obs_size, num_actions, hidden=(args.hidden, args.hidden)).to(device)
    policy_module = TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(module=policy_module, in_keys=["logits"], out_keys=["action"],
                               distribution_class=Categorical, return_log_prob=True)

    # the baseline: a small value network that scores a state, used to steady the updates
    value_net = qnetwork(obs_size, 1, hidden=(args.hidden, args.hidden)).to(device)
    critic = ValueOperator(value_net, in_keys=["observation"])

    # the advantage and the loss
    advantage = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=critic, average_gae=True)
    loss_module = ReinforceLoss(actor, critic_network=critic)
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    collector = Collector(env, actor, frames_per_batch=args.frames_per_batch,
                          total_frames=args.total_steps, device=device)

    history = []
    episode_return = 0.0
    last_loss = 0.0
    frames = 0
    report_every = 25  # print a progress line every this many finished episodes
    next_report = report_every

    print(f"Training REINFORCE experiment {args.exp} for {args.total_steps} steps on {device}")

    for data in collector:
        batch = data.reshape(-1)
        frames += batch.numel()

        # one Monte Carlo update on the whole fresh batch of episodes
        advantage(batch)
        losses = loss_module(batch)
        # an entropy bonus keeps the policy from collapsing to one action too early
        entropy = actor.get_dist(batch).entropy().mean()
        loss = losses["loss_actor"] + losses["loss_value"] - args.entropy * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        rewards = batch["next", "reward"].squeeze(-1).tolist()
        dones = batch["next", "done"].squeeze(-1).tolist()
        for reward, done in zip(rewards, dones):
            episode_return += reward
            if done:
                history.append({"episode": len(history) + 1, "return": round(episode_return, 2),
                                "loss": round(last_loss, 4)})
                episode_return = 0.0

        if len(history) >= next_report:
            recent = [row["return"] for row in history[-report_every:]]
            avg = sum(recent) / len(recent)
            print(f"  step {frames}/{args.total_steps}  episodes {len(history)}  "
                  f"recent avg reward {avg:.1f}  loss {last_loss:.3f}", flush=True)
            next_report += report_every

    collector.shutdown()

    # score the trained policy by greedy play, taking the most likely action
    greedy_avg, win_rate = evaluate(greedy_action_of(policy_net), episodes=args.eval_episodes, seed=1000)

    config = {"algo": "reinforce", "exp": args.exp, "lr": args.lr, "gamma": args.gamma,
              "frames_per_batch": args.frames_per_batch, "lmbda": args.lmbda,
              "entropy": args.entropy, "total_steps": args.total_steps,
              "hidden": args.hidden, "seed": args.seed}
    folder = save_experiment("reinforce", args.exp, policy_net, history, config, greedy_avg, win_rate)

    print(f"REINFORCE experiment {args.exp} done")
    print(f"  episodes played: {len(history)}")
    print(f"  greedy average reward: {greedy_avg:.2f}")
    print(f"  win rate: {win_rate:.2f}")
    print(f"  saved to: {folder}")


if __name__ == "__main__":
    main()
