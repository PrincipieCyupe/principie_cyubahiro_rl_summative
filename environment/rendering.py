"""Viewers for Flood Guardian: a 2D pygame window and a 3D three.js page served on localhost.

Only used for playing locally. Training code should never import this file.
"""

import json
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from environment.custom_env import ACTION_NAMES, LAND, WATER, FLOOD, CROP, HOUSE

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"
MESH_DIR = ASSETS_DIR / "meshes"  # optional .glb files: guardian, crocodile, hippo
WEB_DIR = ASSETS_DIR / "web"      # the 3D page: index.html, app.js, world.js, style.css

END_PAUSE = 2.0  # seconds the result stays on screen before the next episode starts


def audio_file(name):
    path = AUDIO_DIR / f"{name}.wav"
    return path if path.exists() else None


def hud_lines(session):
    env = session.env
    action = "-" if env.last_action is None else ACTION_NAMES[env.last_action]
    return [
        f"Episode: {session.episode}",
        f"Step: {env.step_count} / {env.max_steps}  (survive to win)",
        f"Action: {action}",
        f"Last reward: {env.last_reward:+.2f}",
        f"Total reward: {env.total_reward:+.2f}",
        f"Status: {env.status}",
        f"Animals: {len(env.animals)}   Crops left: {int((env.grid == CROP).sum())}",
    ]


def result_text(env):
    if env.status == "win":
        return "SEASON SAVED - the village is safe"
    if env.status == "fail" and "no_crops" in env.events:
        return "FAILED - half the harvest was eaten"
    if env.status == "fail":
        return "BREACH - an animal reached the village"
    return ""


class Driver:
    """Decides when the game steps: on a timer when an agent plays, on key presses in keyboard mode."""

    def __init__(self, session, mode, speed):
        self.session = session
        self.mode = mode
        self.step_time = 1.0 / speed
        self.timer = 0.0
        self.end_timer = 0.0
        self.quit = False

    def tick(self, dt):
        """Called every frame. Returns the action taken, or None."""
        if self.mode == "keyboard":
            return None
        if self.session.done:
            self.end_timer += dt
            if self.end_timer >= END_PAUSE:
                self.end_timer = 0.0
                if self.session.episode < self.session.episodes:
                    self.session.new_episode()
                else:
                    print("Demo finished.")
                    self.quit = True
            return None
        self.timer += dt
        if self.timer < self.step_time:
            return None
        self.timer = 0.0
        action = self.session.agent_action()
        self.session.take_step(action)
        return action

    def key(self, name):
        """name is an action name, 'restart' or 'quit'. Returns the action taken, or None."""
        if name == "quit":
            self.quit = True
        elif name == "restart":
            self.session.new_episode()
        elif self.mode == "keyboard" and name in ACTION_NAMES and not self.session.done:
            action = ACTION_NAMES.index(name)
            self.session.take_step(action)
            return action
        return None


# 2D view

CELL = 48
HUD_HEIGHT = 120
COLOURS_2D = {
    LAND: (150, 190, 100),
    WATER: (50, 110, 200),
    FLOOD: (140, 120, 80),
    CROP: (90, 70, 40),
    HOUSE: (150, 190, 100),
}


class PygameSounds:
    """Plays the action clips with pygame. Missing files or no audio device just means silence."""

    def __init__(self, mute):
        self.sounds = {}
        self.voice = None
        if mute:
            return
        import pygame
        try:
            pygame.mixer.init()
            pygame.mixer.set_reserved(1)  # channel 0 is only for the spoken cues
            self.voice = pygame.mixer.Channel(0)
        except pygame.error:
            return
        for name in ACTION_NAMES:
            path = audio_file(name)
            if path:
                try:
                    self.sounds[name] = pygame.mixer.Sound(str(path))
                except pygame.error:
                    pass

    def play_action(self, action):
        if action is None:
            return
        name = ACTION_NAMES[action]
        sound = self.sounds.get(name)
        if sound is None:
            return
        if name == "scare":
            sound.play()
        else:
            self.voice.play(sound)  # cuts off the previous word


def draw_cell_2d(screen, pygame, cell_type, rect):
    pygame.draw.rect(screen, COLOURS_2D[cell_type], rect)
    if cell_type == WATER:
        for i in range(2):
            y = rect.y + 14 + i * 18
            pygame.draw.line(screen, (120, 170, 235), (rect.x + 8, y), (rect.right - 8, y), 2)
    elif cell_type == CROP:
        for i in range(3):
            x = rect.x + 10 + i * 14
            pygame.draw.line(screen, (60, 160, 50), (x, rect.y + 8), (x, rect.bottom - 8), 5)
    elif cell_type == HOUSE:
        pygame.draw.rect(screen, (230, 210, 170), (rect.x + 9, rect.y + 20, rect.w - 18, rect.h - 24))
        roof = [(rect.x + 5, rect.y + 22), (rect.centerx, rect.y + 5), (rect.right - 5, rect.y + 22)]
        pygame.draw.polygon(screen, (170, 60, 40), roof)
        pygame.draw.rect(screen, (90, 60, 40), (rect.centerx - 5, rect.bottom - 16, 10, 12))
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)


def draw_animal_2d(screen, pygame, animal):
    rect = pygame.Rect(animal["x"] * CELL, animal["y"] * CELL, CELL, CELL)
    if animal["fleeing"]:
        pygame.draw.rect(screen, (255, 230, 80), rect.inflate(-4, -4), 3)
    if animal["kind"] == "crocodile":
        body = pygame.Rect(rect.x + 4, rect.centery - 9, rect.w - 8, 18)
        pygame.draw.ellipse(screen, (40, 90, 40), body)
        pygame.draw.circle(screen, (240, 220, 60), (body.right - 9, body.y + 5), 3)
    else:
        grey = (120, 105, 125)
        pygame.draw.circle(screen, grey, rect.center, 17)
        pygame.draw.circle(screen, grey, (rect.centerx - 10, rect.y + 9), 5)
        pygame.draw.circle(screen, grey, (rect.centerx + 10, rect.y + 9), 5)
        pygame.draw.circle(screen, (20, 20, 20), (rect.centerx - 6, rect.centery - 4), 3)
        pygame.draw.circle(screen, (20, 20, 20), (rect.centerx + 6, rect.centery - 4), 3)


def draw_2d(screen, pygame, session, font, big_font):
    env = session.env
    screen.fill((25, 25, 30))

    for y in range(env.size):
        for x in range(env.size):
            draw_cell_2d(screen, pygame, int(env.grid[y, x]), pygame.Rect(x * CELL, y * CELL, CELL, CELL))

    # light up the scare area for one step after a scare
    gx, gy = env.guardian
    if env.last_action == ACTION_NAMES.index("scare"):
        r = env.scare_radius
        area = pygame.Surface(((2 * r + 1) * CELL, (2 * r + 1) * CELL), pygame.SRCALPHA)
        area.fill((255, 230, 80, 70))
        screen.blit(area, ((gx - r) * CELL, (gy - r) * CELL))

    for animal in env.animals:
        draw_animal_2d(screen, pygame, animal)

    centre = (gx * CELL + CELL // 2, gy * CELL + CELL // 2)
    pygame.draw.circle(screen, (250, 220, 40), centre, 17)
    pygame.draw.circle(screen, (30, 80, 200), centre, 13)

    top = env.size * CELL + 8
    half = env.size * CELL // 2
    for i, line in enumerate(hud_lines(session)):
        x = 12 if i < 4 else 12 + half
        y = top + (i % 4) * 22
        screen.blit(font.render(line, True, (235, 235, 235)), (x, y))
    if session.mode == "keyboard":
        help_text = "Arrows move, Space scares, R restarts, Esc quits"
    else:
        help_text = "R restarts, Esc quits"
    screen.blit(font.render(help_text, True, (160, 160, 160)), (12, top + 4 * 22))

    message = result_text(env)
    if message:
        text = big_font.render(message, True, (255, 255, 255))
        box = text.get_rect(center=(half, half)).inflate(30, 20)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((30, 130, 60, 210) if env.status == "win" else (170, 30, 30, 210))
        screen.blit(panel, box.topleft)
        screen.blit(text, text.get_rect(center=box.center))


def run_2d(session, mode, speed=2.0, mute=False):
    """Run the demo in a pygame window, seen from above."""
    import pygame

    pygame.init()
    size = session.env.size
    screen = pygame.display.set_mode((size * CELL, size * CELL + HUD_HEIGHT))
    pygame.display.set_caption("Flood Guardian - 2D")
    font = pygame.font.SysFont("arial", 16)
    big_font = pygame.font.SysFont("arial", 24, bold=True)
    sounds = PygameSounds(mute)
    driver = Driver(session, mode, speed)
    clock = pygame.time.Clock()
    keys = {
        pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
        pygame.K_SPACE: "scare", pygame.K_r: "restart", pygame.K_ESCAPE: "quit", pygame.K_q: "quit",
    }

    while not driver.quit:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                driver.quit = True
            elif event.type == pygame.KEYDOWN and event.key in keys:
                sounds.play_action(driver.key(keys[event.key]))
        sounds.play_action(driver.tick(dt))
        draw_2d(screen, pygame, session, font, big_font)
        pygame.display.flip()

    pygame.quit()




# 3D view in the browser
#
# Python keeps the environment. The page (assets/web) asks for the state, draws it with
# three.js, and sends actions back:
#   GET  /api/state            current state as JSON
#   POST /api/step   {action}  one step; action null means "let the selected agent choose"
#   POST /api/reset            start a new episode
#   POST /api/agent  {name}    switch the agent that plays
#
# The same server also answers a small inference API, so other apps can use the trained agent:
#   GET  /api/health              is the server up and which model is loaded
#   GET  /api/act                 help: how to call POST /api/act
#   POST /api/act  {observation}  send the 14 observation numbers, get the next action back
#   GET  /docs                    a Swagger page to try POST /api/act in the browser
#   GET  /openapi.json            the OpenAPI description the Swagger page reads

CONTENT_TYPES = {
    ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
    ".glb": "model/gltf-binary", ".wav": "audio/wav", ".jpg": "image/jpeg", ".png": "image/png",
}


def web_state(session):
    env = session.env
    return {
        "episode": session.episode,
        "max_episodes": session.episodes,
        "size": env.size,
        "grid": env.grid.tolist(),
        "guardian": [int(v) for v in env.guardian],
        "animals": [{"id": int(a["id"]), "x": int(a["x"]), "y": int(a["y"]),
                     "kind": a["kind"], "fleeing": bool(a["fleeing"])} for a in env.animals],
        "in_river": len(env.resting),
        "step": env.step_count,
        "max_steps": env.max_steps,
        "status": env.status,
        "result": result_text(env),
        "last_action": None if env.last_action is None else ACTION_NAMES[env.last_action],
        "last_reward": float(env.last_reward),
        "total_reward": float(env.total_reward),
        "events": list(env.events),
        "scare_radius": env.scare_radius,
        "crops_left": int((env.grid == CROP).sum()),
        "agent": session.agent,
        "agents": list(session.agents),
        "results": [{"status": s, "total": t} for s, t in session.results],
        "meshes": sorted(p.stem for p in MESH_DIR.glob("*.glb")),
    }


def load_best_policy(obs_size, num_actions):
    """Load the strongest trained model so other apps can ask it for an action.

    Returns (net, name), or (None, None) if nothing has been trained yet.
    """
    import json
    import torch
    from training.common import qnetwork

    root = Path(__file__).resolve().parent.parent
    for algo in ("dqn", "ppo", "reinforce", "ac"):
        model = root / "models" / algo / "best_model" / "model.pth"
        if not model.exists():
            continue
        hidden = 128
        config = model.parent / "config.json"
        if config.exists():
            hidden = json.loads(config.read_text()).get("hidden", 128)
        net = qnetwork(obs_size, num_actions, hidden=(hidden, hidden))
        net.load_state_dict(torch.load(model, map_location="cpu"))
        net.eval()
        return net, algo
    return None, None


def policy_action(net, observation):
    """Run the network on one observation and return the best action and the raw scores."""
    import torch

    with torch.no_grad():
        values = net(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
    scores = [round(float(v), 4) for v in values.squeeze(0).tolist()]
    action = max(range(len(scores)), key=lambda i: scores[i])
    return action, scores


def openapi_spec(obs_size):
    """A small OpenAPI description of the agent API, used by the Swagger page at /docs."""
    example = {"observation": [0.5, 0.9, 0.1, -0.1, 0.3, 0.2, 0.1, 0.5, 0, 0, 1, 0.4, 0.3, 0.2]}
    return {
        "openapi": "3.0.0",
        "info": {"title": "Flood Guardian agent API", "version": "1.0.0",
                 "description": f"Ask the trained Flood Guardian agent for its next action. "
                                f"Send {obs_size} observation numbers to POST /api/act."},
        "paths": {
            "/api/act": {
                "post": {
                    "summary": "Get the next action for one observation",
                    "requestBody": {"required": True, "content": {"application/json": {
                        "schema": {"type": "object", "required": ["observation"], "properties": {
                            "observation": {"type": "array", "items": {"type": "number"},
                                            "minItems": obs_size, "maxItems": obs_size}}},
                        "example": example}}},
                    "responses": {"200": {"description": "the chosen action", "content": {"application/json": {
                        "example": {"action": 4, "action_name": "scare",
                                    "scores": [-3.88, -2.26, -2.76, 1.55, 4.30]}}}}},
                }
            },
            "/api/health": {"get": {"summary": "Is the server up and which model is loaded",
                                    "responses": {"200": {"description": "ok"}}}},
        },
    }


SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Flood Guardian API</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css">
</head>
<body>
  <div id="swagger"></div>
  <script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger" });
  </script>
</body>
</html>
"""


class WebHandler(BaseHTTPRequestHandler):
    """Serves the page and its files, and answers the /api calls."""

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            with self.server.lock:
                self.send_json(web_state(self.server.session))
        elif path == "/api/health":
            self.send_json({"status": "ok", "model": self.server.policy_algo,
                            "observation_size": self.server.obs_size, "actions": list(ACTION_NAMES)})
        elif path == "/api/act":
            self.send_json({"how_to": f"POST a body {{\"observation\": [...]}} of {self.server.obs_size} "
                                      "numbers to get the next action. Open /docs for the Swagger page",
                            "actions": list(ACTION_NAMES),
                            "example": {"observation": [0.0] * self.server.obs_size}})
        elif path == "/docs":
            self.send_html(SWAGGER_HTML)
        elif path == "/openapi.json":
            self.send_json(openapi_spec(self.server.obs_size))
        elif path.startswith("/audio/"):
            self.send_file(AUDIO_DIR, path[len("/audio/"):])
        elif path.startswith("/meshes/"):
            self.send_file(MESH_DIR, path[len("/meshes/"):])
        else:
            self.send_file(WEB_DIR, path.lstrip("/") or "index.html")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        path = urlparse(self.path).path
        if path == "/api/act":
            self.handle_act(body)
            return
        session = self.server.session
        with self.server.lock:
            if path == "/api/step":
                if not session.done:
                    action = body.get("action")
                    session.take_step(session.agent_action() if action is None else int(action))
            elif path == "/api/reset":
                if body.get("fresh"):
                    session.start_over()
                else:
                    session.new_episode()
            elif path == "/api/agent":
                session.set_agent(body.get("name", ""))
            else:
                self.send_error(404)
                return
            self.send_json(web_state(session))

    def handle_act(self, body):
        net = self.server.policy
        if net is None:
            self.send_json({"error": "no trained model found; train the agents and run select_best.py first"})
            return
        observation = body.get("observation")
        size = self.server.obs_size
        if not isinstance(observation, list) or len(observation) != size:
            self.send_json({"error": f"send 'observation' as a list of {size} numbers"})
            return
        action, scores = policy_action(net, observation)
        self.send_json({"action": action, "action_name": ACTION_NAMES[action], "scores": scores})

    def send_json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, html):
        payload = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, folder, name):
        # resolve the path and refuse anything that tries to leave the folder
        path = (folder / name).resolve()
        if folder.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass  # keep the terminal for the step printout


def run_3d(session, mode, speed=2.0, mute=False, port=8000, open_browser=True):
    """Serve the 3D page on localhost and open it in the browser. Ctrl+C stops it."""
    server = ThreadingHTTPServer(("127.0.0.1", port), WebHandler)
    server.session = session
    server.lock = threading.Lock()  # page requests arrive on different threads
    # load the best trained model once, so other apps can call POST /api/act
    server.obs_size = session.env.observation_space.shape[0]
    server.policy, server.policy_algo = load_best_policy(server.obs_size, session.env.action_space.n)

    query = f"?mode={mode}&speed={speed}&mute={int(mute)}"
    url = f"http://localhost:{port}/{query}"
    print(f"3D view running at {url}")
    print("Press Ctrl+C here to stop.")
    if server.policy is not None:
        print(f"Agent API: try it at http://localhost:{port}/docs  "
              f"(POST /api/act, serving the {server.policy_algo} best model)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()