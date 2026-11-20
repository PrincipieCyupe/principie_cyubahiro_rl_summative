"""A simple hand written guardian. It is not trained, it only follows a rule.

It is useful in two ways: it shows the game can be won, and it gives the trained
agents (DQN, REINFORCE, PPO, A2C) a sensible score to beat in the report.

The rule: find the calm animal that is closest to a house or a crop, walk toward it,
and scare as soon as it is inside the scare radius.
"""

from environment.custom_env import UP, DOWN, LEFT, RIGHT, SCARE, HOUSE, CROP


def rule_based_action(env):
    gx, gy = env.guardian
    calm = [a for a in env.animals if not a["fleeing"]]
    if not calm:
        return SCARE  # nothing to chase, wait where we are

    targets = env.cells_of(HOUSE, CROP)
    animal = min(calm, key=lambda a: env.nearest_distance(a["x"], a["y"], targets))
    dx, dy = animal["x"] - gx, animal["y"] - gy
    if max(abs(dx), abs(dy)) <= env.scare_radius:
        return SCARE
    if abs(dx) >= abs(dy):
        return RIGHT if dx > 0 else LEFT
    return DOWN if dy > 0 else UP
