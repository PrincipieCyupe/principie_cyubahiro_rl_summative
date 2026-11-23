"""Train one Actor Critic (A2C) experiment on Flood Guardian with TorchRL.

A2C is policy based with a helper. The actor is the policy that picks actions, and the critic
scores how good a state is. After a short rollout the critic says whether each action did
better or worse than expected, and the actor is nudged that way. Because the critic bootstraps,
A2C learns from the per step rewards instead of waiting for a whole episode, so it is steadier
than REINFORCE. TorchRL gives us the ready parts:

    ProbabilisticActor   the policy that samples an action (our qnetwork as logits)
    ValueOperator        the critic that scores a state (our qnetwork with one output)
    GAE                  turns rewards into an advantage for each step
    A2CLoss              the policy loss, the critic loss and an entropy bonus
    Collector            plays the env and collects short rollouts

Run one experiment, for example:
    uv run python training/actor_critic_training.py --exp 1 --lr 7e-4 --gamma 0.99
"""

import argparse
import sys
from pathlib import Path

# let "uv run python training/actor_critic_training.py" find the environment and training packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.distributions import Categorical

import environment  # registers FloodGuardian-v0
from environment import ENV_ID
from torchrl.envs import GymEnv, TransformedEnv, StepCounter, InitTracker
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import Collector
from torchrl.objectives import A2CLoss, ValueEstimators
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

from training.common import qnetwork, evaluate, greedy_action_of, save_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Train one Actor Critic experiment.")
    p.add_argument("--exp", type=int, required=True, help="experiment number, saved in models/ac/exp<nn>")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--frames-per-batch", type=int, default=500, help="steps collected before each update")
    p.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda, lower trusts the critic more")
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

    env = TransformedEnv(GymEnv(ENV_ID, device=device), StepCounter())
    env.append_transform(InitTracker())
    env.set_seed(args.seed)

    obs_size = env.observation_spec["observation"].shape[-1]
    num_actions = env.action_spec.space.n

    # the actor: the network gives logits, the actor samples an action from them
    policy_net = qnetwork(obs_size, num_actions, hidden=(args.hidden, args.hidden)).to(device)
    policy_module = TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(module=policy_module, in_keys=["logits"], out_keys=["action"],
                               distribution_class=Categorical, return_log_prob=True)

    # the critic: a small value network that scores a state
    value_net = qnetwork(obs_size, 1, hidden=(args.hidden, args.hidden)).to(device)
    critic = ValueOperator(value_net, in_keys=["observation"])

    # the advantage and the loss, with an entropy bonus built in
    advantage = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=critic, average_gae=True)
    loss_module = A2CLoss(actor, critic, entropy_bonus=args.entropy > 0, entropy_coeff=args.entropy)
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

    print(f"Training A2C experiment {args.exp} for {args.total_steps} steps on {device}")

    for data in collector:
        batch = data.reshape(-1)
        frames += batch.numel()

        advantage(batch)
        losses = loss_module(batch)
        loss = losses["loss_objective"] + losses["loss_critic"]
        if "loss_entropy" in losses.keys():
            loss = loss + losses["loss_entropy"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        collector.update_policy_weights_()  # let the collector play with the updated policy
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

    greedy_avg, win_rate = evaluate(greedy_action_of(policy_net), episodes=args.eval_episodes, seed=1000)

    config = {"algo": "ac", "exp": args.exp, "lr": args.lr, "gamma": args.gamma,
              "frames_per_batch": args.frames_per_batch, "lmbda": args.lmbda,
              "entropy": args.entropy, "total_steps": args.total_steps,
              "hidden": args.hidden, "seed": args.seed}
    folder = save_experiment("ac", args.exp, policy_net, history, config, greedy_avg, win_rate)

    print(f"A2C experiment {args.exp} done")
    print(f"  episodes played: {len(history)}")
    print(f"  greedy average reward: {greedy_avg:.2f}")
    print(f"  win rate: {win_rate:.2f}")
    print(f"  saved to: {folder}")


if __name__ == "__main__":
    main()
