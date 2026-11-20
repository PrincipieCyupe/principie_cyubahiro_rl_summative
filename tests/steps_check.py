"""Progress checker for the RL summative.

Run it with:  uv run python tests/steps_check.py

It goes through the whole project, section by section, and marks every item
DONE, WAITING or NOT DONE, then prints a short to do list. It only uses the
standard library, so it works even before any package is installed.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the project folder, one level above tests/

EXPERIMENTS_NEEDED = 10           # the assignment asks for at least 10 runs per algorithm
MODEL_TYPES = (".pt", ".pth")     # the usual file types for saved PyTorch / TorchRL models

DONE = "DONE"
WAITING = "WAITING"
NOT_DONE = "NOT DONE"


def check_file(name):
    path = ROOT / name
    if not path.is_file():
        return NOT_DONE, "missing"
    if path.stat().st_size == 0:
        return WAITING, "file is empty"
    return DONE, ""


def check_group(group):
    # group is (folder, list of files that should be inside it)
    folder, names = group
    found = sum((ROOT / folder / name).is_file() for name in names)
    note = f"{found} of {len(names)} files"
    if found == len(names):
        return DONE, note
    if found == 0:
        return NOT_DONE, note
    missing = [name for name in names if not (ROOT / folder / name).is_file()]
    return WAITING, note + ", missing " + ", ".join(missing)


def check_environment(_):
    # really start the game: reset it and play 20 steps
    sys.path.insert(0, str(ROOT))
    try:
        import gymnasium
        import environment  # noqa: F401  (importing it registers FloodGuardian-v0)
    except ImportError as error:
        return WAITING, f"packages not installed yet, run uv sync ({error})"
    env = gymnasium.make("FloodGuardian-v0")
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()  # random play loses quickly, so start a new season like training does
    return DONE, "reset and 200 random steps worked"


def has_model(folder):
    return any(f.is_file() and f.suffix in MODEL_TYPES for f in folder.iterdir())


def check_models(name):
    path = ROOT / name
    if not path.is_dir():
        return NOT_DONE, f"folder missing, 0 of {EXPERIMENTS_NEEDED} experiments"

    finished = 0
    without_model = 0
    for experiment in path.iterdir():
        if not experiment.is_dir():
            continue
        if has_model(experiment):
            finished += 1
        else:
            without_model += 1

    note = f"{finished} of {EXPERIMENTS_NEEDED} experiments"
    if without_model:
        note += f", {without_model} folder(s) with no model file yet"
    if finished >= EXPERIMENTS_NEEDED:
        return DONE, note
    if finished == 0:
        return NOT_DONE, note
    return WAITING, note


def check_logs(name):
    path = ROOT / name
    files = [f for f in path.rglob("*") if f.is_file()] if path.is_dir() else []
    if not files:
        return NOT_DONE, "no training results saved yet"
    return DONE, f"{len(files)} file(s)"


def check_git(_):
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        return WAITING, "could not ask git"
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    if not changed:
        return DONE, "nothing left to commit"
    return WAITING, f"{len(changed)} file(s) changed or not committed yet"


VENDOR_FILES = [
    "three.module.min.js",
    "LICENSE",
    "addons/objects/Sky.js",
    "addons/objects/Water.js",
    "addons/loaders/GLTFLoader.js",
    "addons/controls/OrbitControls.js",
    "addons/utils/BufferGeometryUtils.js",
    "addons/utils/SkeletonUtils.js",
]

# each section: (title, list of (what to show, how to check it, what to check))
SECTIONS = [
    ("Project setup", [
        ("pyproject.toml", check_file, "pyproject.toml"),
        ("uv.lock", check_file, "uv.lock"),
        ("README.md", check_file, "README.md"),
        (".gitignore", check_file, ".gitignore"),
        ("all work committed to git", check_git, None),
    ]),
    ("Environment", [
        ("environment/__init__.py", check_file, "environment/__init__.py"),
        ("environment/custom_env.py", check_file, "environment/custom_env.py"),
        ("environment/rendering.py", check_file, "environment/rendering.py"),
        ("environment/baseline.py", check_file, "environment/baseline.py"),
        ("main.py", check_file, "main.py"),
        ("environment starts and runs", check_environment, None),
    ]),
    ("3D view and assets", [
        ("3D page (assets/web)", check_group, ("assets/web", ["index.html", "style.css", "app.js", "world.js"])),
        ("three.js files (assets/web/vendor)", check_group, ("assets/web/vendor", VENDOR_FILES)),
        ("water texture", check_file, "assets/web/textures/waternormals.jpg"),
        ("sound clips (assets/audio)", check_group,
         ("assets/audio", ["up.wav", "down.wav", "left.wav", "right.wav", "scare.wav"])),
        ("animal models (assets/meshes)", check_group, ("assets/meshes", ["crocodile.glb", "hippo.glb"])),
        ("assets/make_voice.py", check_file, "assets/make_voice.py"),
        ("assets/convert_blend.py", check_file, "assets/convert_blend.py"),
    ]),
    ("Training", [
        ("training/__init__.py", check_file, "training/__init__.py"),
        ("training/dqn_training.py", check_file, "training/dqn_training.py"),
        ("training/reinforce_training.py", check_file, "training/reinforce_training.py"),
        ("training/ppo_training.py", check_file, "training/ppo_training.py"),
        ("training/actor_critic_training.py", check_file, "training/actor_critic_training.py"),
    ]),
    ("Agents and results", [
        ("models/dqn", check_models, "models/dqn"),
        ("models/reinforce", check_models, "models/reinforce"),
        ("models/ppo", check_models, "models/ppo"),
        ("models/ac", check_models, "models/ac"),
        ("logs (training results)", check_logs, "logs"),
        ("play.py", check_file, "play.py"),
    ]),
]


def main():
    print("Summative progress check")

    done, waiting, not_done = [], [], []
    for title, items in SECTIONS:
        print()
        print(title)
        for name, check, target in items:
            try:
                mark, note = check(target)
            except Exception as error:  # never crash, just report what went wrong
                mark, note = NOT_DONE, f"could not check: {error}"

            line = f"  [{mark:<8}] {name}"
            if note:
                line += f"  ({note})"
            print(line)

            if mark == DONE:
                done.append(name)
            elif mark == WAITING:
                waiting.append(f"{name}  ({note})")
            else:
                not_done.append(name)

    total = len(done) + len(waiting) + len(not_done)
    print()
    print(f"{len(done)} of {total} done, {len(waiting)} waiting, {len(not_done)} not done")

    if waiting:
        print()
        print("Still in progress:")
        for item in waiting:
            print(f"  - {item}")
    if not_done:
        print()
        print("To do:")
        for item in not_done:
            print(f"  - {item}")
    if not waiting and not not_done:
        print()
        print("Everything is in place. Well done!")


if __name__ == "__main__":
    main()
