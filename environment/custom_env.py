"""Flood Guardian environment.

A guardian protects a village in Rwanda's Eastern Province during the flood season.
Three animals (crocodiles and hippos) live in the river. They keep leaving the water and
walking toward the houses and crops, and the guardian has to scare them back. A chased
animal rests in the river for a while and then comes out again. The village wins if it
gets through the whole season without an animal reaching a house or the guardian.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# rewards
REWARD_STEP = -0.1        # small cost every step so the agent does not waste time
REWARD_PUSH = 1.0         # a scare sent an animal running back toward the river
REWARD_DROWN = 5.0        # a scared animal made it back into the water
REWARD_CROP_LOST = -5.0   # an animal ate a crop
REWARD_BREACH = -20.0     # the village is lost: an animal reached a house or the guardian, or the harvest is lost

# game settings
GRID_SIZE = 12
MAX_STEPS = 100           # one flood season
ANIMALS = 3               # the same three animals live in the river all season
MAX_ON_LAND = 3           # at most this many are out of the water at the same time
REST_MIN = 8              # after being chased back, an animal rests in the river
REST_MAX = 20             # for REST_MIN to REST_MAX steps before it comes out again
MIN_CROPS = 5             # the harvest is lost (game over) once half the crops (4 of 8) are eaten
SCARE_RADIUS = 3          # 4 lets a guardian win without moving, 2 is very hard
ANIMAL_MOVE_CHANCE = 0.5  # animals are slow on land and only move half of the time
ANIMAL_WANDER_CHANCE = 0.2
NEAREST_ANIMALS = 3       # how many animals the agent can see

# cell types
LAND, WATER, FLOOD, CROP, HOUSE = 0, 1, 2, 3, 4

# actions
UP, DOWN, LEFT, RIGHT, SCARE = 0, 1, 2, 3, 4
ACTION_NAMES = ["up", "down", "left", "right", "scare"]
MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

WATER_ROWS = 2  # rows 0 and 1 are the river
FLOOD_ROW = 2   # the muddy flood edge just below the river


class FloodGuardianEnv(gym.Env):
    """Grid world where the guardian scares river animals away from the village.

    Actions (Discrete 5):
        0 up, 1 down, 2 left, 3 right - move one cell. Grid edges and animals block the move.
        4 scare - stay in place. Every animal within the scare radius turns and runs straight back
                  to the river, one cell per step. Back in the water it rests, then returns.

    Observation (14 floats):
        0-1    guardian x and y divided by (size - 1), range 0 to 1
        2-10   the animals on land, nearest first, 3 slots. For each one: dx and dy to the guardian
               divided by (size - 1), range -1 to 1, then the distance divided by the grid
               diagonal, range 0 to 1. Empty slots are filled with 0, 0, 1. Animals resting in
               the river are under water and cannot be seen.
        11     distance from the guardian to the nearest house, range 0 to 1
        12     distance from the guardian to the nearest crop, range 0 to 1 (1 when all crops are gone)
        13     season progress, step / max_steps, range 0 to 1
    Every value is read from the game state after each step.

    The episode ends with terminated=True when an animal reaches a house or the guardian, or the
    crops left drop below MIN_CROPS (fail), or when the season reaches max_steps (win). The season end is the
    win condition and the agent can see the season progress, so it is a real ending and
    truncated stays False.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, size=GRID_SIZE, max_steps=MAX_STEPS, scare_radius=SCARE_RADIUS, render_mode=None):
        super().__init__()
        if size < 8:
            raise ValueError("Grid size must be at least 8.")
        self.size = size
        self.max_steps = max_steps
        self.scare_radius = scare_radius
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        low = np.zeros(2 + NEAREST_ANIMALS * 3 + 3, dtype=np.float32)
        high = np.ones_like(low)
        for i in range(NEAREST_ANIMALS):
            low[2 + i * 3] = -1.0  # dx
            low[3 + i * 3] = -1.0  # dy
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_distance = np.sqrt(2) * (size - 1)  # length of the grid diagonal

        self.grid = None
        self.guardian = (0, 0)
        self.animals = []   # animals on land
        self.resting = []   # animals in the river, each with a countdown before it comes back
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.status = "running"
        self.events = []

    def house_cells(self):
        # a 2 by 2 block of houses near the bottom middle
        cx, bottom = self.size // 2, self.size - 2
        return [(cx - 1, bottom - 1), (cx, bottom - 1), (cx - 1, bottom), (cx, bottom)]

    def crop_cells(self):
        # two 2 by 2 crop patches, one on each side of the village
        top = self.size - 5
        return [(x0 + dx, top + dy) for x0 in (1, self.size - 3) for dx in range(2) for dy in range(2)]

    def build_grid(self):
        grid = np.full((self.size, self.size), LAND, dtype=np.int8)  # indexed as grid[y, x]
        grid[:WATER_ROWS, :] = WATER
        grid[FLOOD_ROW, :] = FLOOD
        for x, y in self.crop_cells():
            grid[y, x] = CROP
        for x, y in self.house_cells():
            grid[y, x] = HOUSE
        return grid

    def animals_come_out(self):
        # rested animals climb out onto a free spot of the flood edge, while there is room on land
        for animal in list(self.resting):
            if animal["rest"] > 0 or len(self.animals) >= MAX_ON_LAND:
                continue
            taken = {(a["x"], a["y"]) for a in self.animals}
            free = [x for x in range(self.size) if (x, FLOOD_ROW) not in taken and (x, FLOOD_ROW) != self.guardian]
            if not free:
                return
            animal.update(x=int(self.np_random.choice(free)), y=FLOOD_ROW, fleeing=False)
            self.resting.remove(animal)
            self.animals.append(animal)
            self.events.append("spawn")

    def rest_time(self):
        return int(self.np_random.integers(REST_MIN, REST_MAX + 1))

    def cells_of(self, *cell_types):
        ys, xs = np.where(np.isin(self.grid, cell_types))
        return list(zip(xs.tolist(), ys.tolist()))

    def nearest_distance(self, x, y, cells):
        if not cells:
            return None
        return min(np.hypot(cx - x, cy - y) for cx, cy in cells)

    def inside(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def reset(self, seed=None, options=None):
        """Start a new season and return the first observation and info."""
        super().reset(seed=seed)
        self.grid = self.build_grid()
        self.guardian = (self.size // 2, self.size - 4)
        self.animals = []
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.status = "running"
        self.events = []

        # all animals start in the river: the first MAX_ON_LAND come out at once, any others a bit later
        self.resting = []
        for i in range(ANIMALS):
            kind = "crocodile" if self.np_random.random() < 0.5 else "hippo"
            rest = 0 if i < MAX_ON_LAND else self.rest_time()
            self.resting.append({"id": i, "kind": kind, "rest": rest, "x": 0, "y": 0, "fleeing": False})
        self.animals_come_out()
        self.events = []

        return self.observation(), self.info()

    def step(self, action):
        """Apply one action, move the animals and return obs, reward, terminated, truncated, info."""
        if self.status != "running":
            raise RuntimeError("The episode is over, call reset() first.")
        action = int(action)
        self.events = []
        reward = REWARD_STEP

        if action == SCARE:
            reward += self.scare()
        else:
            self.move_guardian(action)

        reward += self.move_animals()

        for animal in self.resting:
            animal["rest"] = max(0, animal["rest"] - 1)
        if self.status == "running":
            self.animals_come_out()

        self.step_count += 1
        if self.status == "running" and self.step_count >= self.max_steps:
            self.status = "win"
        terminated = self.status != "running"
        truncated = False

        self.last_action = action
        self.last_reward = reward
        self.total_reward += reward
        return self.observation(), reward, terminated, truncated, self.info()

    def move_guardian(self, action):
        dx, dy = MOVES[action]
        x, y = self.guardian[0] + dx, self.guardian[1] + dy
        blocked = any(a["x"] == x and a["y"] == y for a in self.animals)
        if self.inside(x, y) and not blocked:
            self.guardian = (x, y)
            self.events.append("move")
        else:
            self.events.append("bump")

    def scare(self):
        """Send every calm animal in range running back to the river. Returns the reward."""
        self.events.append("scare")
        reward = 0.0
        gx, gy = self.guardian
        for animal in self.animals:
            near = max(abs(animal["x"] - gx), abs(animal["y"] - gy)) <= self.scare_radius
            if near and not animal["fleeing"]:
                animal["fleeing"] = True  # an animal already running away gives no extra reward
                reward += REWARD_PUSH
                self.events.append("pushed")
        return reward

    def animal_step(self, animal):
        # usually walk toward the closest house or crop, sometimes wander
        x, y = animal["x"], animal["y"]
        if self.np_random.random() < ANIMAL_WANDER_CHANCE:
            dx, dy = MOVES[int(self.np_random.integers(0, 4))]
        else:
            tx, ty = min(self.cells_of(HOUSE, CROP), key=lambda c: abs(c[0] - x) + abs(c[1] - y))
            go_x = abs(tx - x) > abs(ty - y) or (abs(tx - x) == abs(ty - y) and self.np_random.random() < 0.5)
            dx, dy = (int(np.sign(tx - x)), 0) if go_x else (0, int(np.sign(ty - y)))

        nx, ny = x + dx, y + dy
        if not self.inside(nx, ny) or ny < WATER_ROWS:
            return x, y  # animals never walk back into the river by themselves
        return nx, ny

    def move_animals(self):
        reward = 0.0
        kept = []
        for animal in self.animals:
            if animal["fleeing"]:
                # runs straight up one cell per step, but never into the guardian
                if (animal["x"], animal["y"] - 1) != self.guardian:
                    animal["y"] -= 1
                if animal["y"] < WATER_ROWS:
                    reward += REWARD_DROWN
                    self.events.append("drowned")
                    animal["rest"] = self.rest_time()  # it will be back later
                    self.resting.append(animal)
                else:
                    kept.append(animal)
                continue

            kept.append(animal)
            if self.status != "running":
                continue  # the village already fell this step, the rest stay where they are
            if self.np_random.random() < ANIMAL_MOVE_CHANCE:
                animal["x"], animal["y"] = self.animal_step(animal)

            cell = self.grid[animal["y"], animal["x"]]
            if cell == HOUSE or (animal["x"], animal["y"]) == self.guardian:
                reward += REWARD_BREACH
                self.status = "fail"
                self.events.append("breach")
            elif cell == CROP:
                reward += REWARD_CROP_LOST
                self.grid[animal["y"], animal["x"]] = LAND  # the crop is gone, so it only costs once
                self.events.append("crop_lost")
                if (self.grid == CROP).sum() < MIN_CROPS:
                    reward += REWARD_BREACH  # not enough food left for the village
                    self.status = "fail"
                    self.events.append("no_crops")
        self.animals = kept
        return reward

    def observation(self):
        gx, gy = self.guardian
        scale = self.size - 1
        obs = [gx / scale, gy / scale]

        nearest = sorted(self.animals, key=lambda a: np.hypot(a["x"] - gx, a["y"] - gy))
        for i in range(NEAREST_ANIMALS):
            if i < len(nearest):
                a = nearest[i]
                dist = np.hypot(a["x"] - gx, a["y"] - gy) / self.max_distance
                obs += [(a["x"] - gx) / scale, (a["y"] - gy) / scale, dist]
            else:
                obs += [0.0, 0.0, 1.0]

        house = self.nearest_distance(gx, gy, self.cells_of(HOUSE))
        crop = self.nearest_distance(gx, gy, self.cells_of(CROP))
        obs.append(1.0 if house is None else house / self.max_distance)
        obs.append(1.0 if crop is None else crop / self.max_distance)
        obs.append(self.step_count / self.max_steps)
        return np.clip(np.array(obs, dtype=np.float32), self.observation_space.low, self.observation_space.high)

    def info(self):
        return {
            "step": self.step_count,
            "status": self.status,
            "animals": len(self.animals),
            "in_river": len(self.resting),
            "crops_left": int(np.sum(self.grid == CROP)),
            "events": list(self.events),
        }

    def render(self):
        """Return the grid as an RGB numpy image. Works without a screen, for recording on Kaggle."""
        if self.render_mode != "rgb_array":
            return None
        cell = 24
        colours = {LAND: (150, 190, 100), WATER: (50, 110, 200), FLOOD: (140, 120, 80),
                   CROP: (60, 140, 40), HOUSE: (200, 90, 60)}
        image = np.zeros((self.size * cell, self.size * cell, 3), dtype=np.uint8)
        for y in range(self.size):
            for x in range(self.size):
                image[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = colours[int(self.grid[y, x])]
        for a in self.animals:
            colour = (30, 70, 30) if a["kind"] == "crocodile" else (110, 100, 120)
            image[a["y"] * cell + 4:(a["y"] + 1) * cell - 4, a["x"] * cell + 4:(a["x"] + 1) * cell - 4] = colour
        gx, gy = self.guardian
        image[gy * cell + 6:(gy + 1) * cell - 6, gx * cell + 6:(gx + 1) * cell - 6] = (250, 220, 40)
        return image
