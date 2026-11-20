// Talks to the Python server, runs the play loop, and keeps the HUD up to date.

import { World } from './world.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

const ICONS = {
  up: 'M12 19V5M5 12l7-7 7 7',
  down: 'M12 5v14M5 12l7 7 7-7',
  left: 'M19 12H5M12 5l-7 7 7 7',
  right: 'M5 12h14M12 5l7 7-7 7',
  scare: 'M3 11v2a1 1 0 0 0 1 1h3l5 4V6L7 10H4a1 1 0 0 0-1 1zM16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12',
  wait: 'M8 6v12M16 6v12',
};
const SOUND_ON = 'M11 5 6 9H2v6h4l5 4V5zM15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14';
const SOUND_OFF = 'M11 5 6 9H2v6h4l5 4V5zM23 9l-6 6M17 9l6 6';

const EVENT_TEXT = {
  pushed: ['Animal scared back', '+1', 'pos'],
  drowned: ['Animal chased into the river', '+5', 'pos'],
  crop_lost: ['A crop was eaten', '-5', 'neg'],
  breach: ['The village was breached', '-20', 'neg'],
  no_crops: ['Half the harvest is lost', '-20', 'neg'],
  spawn: ['An animal came out of the river', '', ''],
};

const ui = {
  running: params.get('mode') !== 'keyboard',
  speed: Number(params.get('speed')) || 4,
  muted: params.get('mute') === '1',
  keyboard: params.get('mode') === 'keyboard',
  busy: false,
  timer: 0,
  endTimer: 0,
  finished: false,
};

let state = null;
const world = new World($('scene'));

// server calls

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  return response.json();
}

async function send(path, body) {
  if (ui.busy) return;
  ui.busy = true;
  try {
    apply(await api(path, body));
  } catch (error) {
    showBanner('info', 'Lost connection', 'Is "uv run main.py" still running?');
    ui.running = false;
  } finally {
    ui.busy = false;
  }
}

const step = (action = null) => send('/api/step', { action });
// after a finished run, a reset starts a fresh run from episode 1
const reset = () => {
  const fresh = ui.finished;
  ui.finished = false;
  ui.endTimer = 0;
  return send('/api/reset', { fresh });
};

// sounds

const sounds = {};
for (const name of ['up', 'down', 'left', 'right', 'scare']) {
  const audio = new Audio(`/audio/${name}.wav`);
  audio.preload = 'auto';
  audio.addEventListener('error', () => { delete sounds[name]; }); // missing file: stay quiet
  sounds[name] = audio;
}

function playSound(name) {
  if (ui.muted || !sounds[name]) return;
  if (name !== 'scare') {
    for (const word of ['up', 'down', 'left', 'right']) {
      if (sounds[word]) { sounds[word].pause(); sounds[word].currentTime = 0; }
    }
  }
  sounds[name].currentTime = 0;
  sounds[name].play().catch(() => {}); // the browser blocks sound until the first click
}

// applying a new state

function apply(s) {
  const previous = state;
  state = s;
  world.update(s);

  const stepped = previous && s.step !== previous.step && s.episode === previous.episode;
  if (stepped && s.last_action) playSound(s.last_action);
  if (stepped) s.events.forEach(toast);
  if (stepped && (s.events.includes('breach') || s.events.includes('no_crops'))) flash();

  updateHud();
}

function updateHud() {
  const s = state;
  $('episode').textContent = s.episode;
  $('step').textContent = `${s.step} / ${s.max_steps}`;
  $('location').textContent = `${s.guardian[0]}, ${s.guardian[1]}`;
  $('animals').textContent = `${s.animals.length} of ${s.animals.length + s.in_river}`;
  $('crops').textContent = `${s.crops_left} / ${world.cropCells.length}`;

  const dot = $('status-dot');
  dot.className = 'dot' + (s.status === 'win' ? ' win' : s.status === 'fail' ? ' fail' : ui.running || ui.keyboard ? '' : ' paused');

  const action = s.last_action || 'wait';
  $('action-name').textContent = action === 'wait' ? 'WAITING' : action === 'scare' ? 'SCARE' : `MOVE ${action.toUpperCase()}`;
  $('action-icon').innerHTML = `<path d="${ICONS[action]}"/>`;
  $('action').className = action === 'scare' ? 'scare' : '';

  const reward = $('reward');
  reward.textContent = (s.last_reward >= 0 ? '+' : '') + s.last_reward.toFixed(1);
  reward.className = s.last_reward > 0 ? 'pos' : s.last_reward < 0 ? 'neg' : '';
  $('reward-bar').style.width = `${Math.min(100, Math.abs(s.last_reward) * 5)}%`;
  $('reward-bar').style.background = s.last_reward < 0 ? 'var(--red)' : 'var(--green)';
  $('total').textContent = s.total_reward.toFixed(1);

  const season = Math.round((100 * s.step) / s.max_steps);
  $('season').textContent = `${season}%`;
  $('season-bar').style.width = `${season}%`;
  $('season-bar').style.background = 'var(--gold)';

  const select = $('agent');
  const options = [...s.agents, 'keyboard'];
  if (select.options.length !== options.length) {
    const names = { random: 'Random', rule_based: 'Rule based', keyboard: 'Keyboard (you)' };
    select.innerHTML = options.map((a) => `<option value="${a}">${names[a] || a.toUpperCase()}</option>`).join('');
  }
  select.value = ui.keyboard ? 'keyboard' : s.agent;
  $('agent-badge').textContent = ui.keyboard ? 'Manual' : 'Loaded';
  $('agent-badge').className = 'badge' + (ui.keyboard ? ' manual' : '');

  const pause = $('btn-pause');
  pause.textContent = ui.running ? 'Pause simulation' : 'Resume simulation';
  pause.classList.toggle('go', !ui.running);

  if (ui.finished) showSummary();
  else if (s.status === 'win') showBanner('win', 'SEASON SAVED', `The village is safe. Total reward ${s.total_reward.toFixed(1)}`);
  else if (s.status === 'fail') showBanner('fail', s.events.includes('no_crops') ? 'HARVEST LOST' : 'VILLAGE BREACHED', s.result);
  else if (!ui.finished) hideBanner();

  drawMinimap();
}

function showSummary() {
  const results = state.results;
  const wins = results.filter((r) => r.status === 'win').length;
  const mean = results.reduce((sum, r) => sum + r.total, 0) / Math.max(1, results.length);
  showBanner('info', 'RUN FINISHED',
    `${results.length} episodes: ${wins} won, ${results.length - wins} lost, mean reward ${mean.toFixed(1)}. Press Start to run again.`);
}

function toast(event) {
  const info = EVENT_TEXT[event];
  if (!info) return;
  const item = document.createElement('div');
  item.className = 'toast';
  item.innerHTML = `<span>${info[0]}</span><b class="${info[2]}">${info[1]}</b>`;
  const feed = $('feed');
  feed.prepend(item);
  while (feed.children.length > 6) feed.lastChild.remove();
  setTimeout(() => item.remove(), 5000);
}

function showBanner(kind, title, sub) {
  $('banner').className = `show ${kind}`;
  $('banner-title').textContent = title;
  $('banner-sub').textContent = sub;
}

function hideBanner() {
  $('banner').className = '';
}

function flash() {
  const f = $('flash');
  f.classList.add('on');
  setTimeout(() => f.classList.remove('on'), 80);
}

// minimap

const MAP_COLOURS = ['#7f9a45', '#2563eb', '#6b4f33', '#3f6212', '#b45309'];

function drawMinimap() {
  const canvas = $('minimap');
  const g = canvas.getContext('2d');
  const n = state.size;
  const c = canvas.width / n;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      g.fillStyle = MAP_COLOURS[state.grid[y][x]];
      g.fillRect(x * c, y * c, c, c);
    }
  }
  g.strokeStyle = 'rgba(0, 0, 0, 0.18)';
  for (let i = 0; i <= n; i++) {
    g.beginPath(); g.moveTo(i * c, 0); g.lineTo(i * c, n * c); g.stroke();
    g.beginPath(); g.moveTo(0, i * c); g.lineTo(n * c, i * c); g.stroke();
  }
  // scare range around the guardian
  const [gx, gy] = state.guardian;
  const r = state.scare_radius;
  g.fillStyle = 'rgba(250, 204, 21, 0.18)';
  g.fillRect((gx - r) * c, (gy - r) * c, (2 * r + 1) * c, (2 * r + 1) * c);
  for (const a of state.animals) {
    g.fillStyle = a.fleeing ? '#facc15' : a.kind === 'crocodile' ? '#4d7c0f' : '#a78bfa';
    g.beginPath();
    g.arc((a.x + 0.5) * c, (a.y + 0.5) * c, c * 0.36, 0, Math.PI * 2);
    g.fill();
    g.strokeStyle = '#111';
    g.stroke();
  }
  g.fillStyle = '#3b82f6';
  g.strokeStyle = '#fff';
  g.lineWidth = 2;
  g.beginPath();
  g.arc((gx + 0.5) * c, (gy + 0.5) * c, c * 0.4, 0, Math.PI * 2);
  g.fill();
  g.stroke();
  g.lineWidth = 1;
}

// controls

function setCamera(mode) {
  world.setCamera(mode);
  document.querySelectorAll('.cams button').forEach((b) => b.classList.toggle('active', Number(b.dataset.cam) === mode));
}

function setSound(muted) {
  ui.muted = muted;
  document.getElementById('sound-path').setAttribute('d', muted ? SOUND_OFF : SOUND_ON);
}

$('btn-start').onclick = () => { ui.running = true; if (state.status !== 'running') reset(); updateHud(); };
$('btn-reset').onclick = () => reset();
$('btn-step').onclick = () => { ui.running = false; step(ui.keyboard ? 4 : null); };
$('btn-pause').onclick = () => { ui.running = !ui.running; updateHud(); };
$('sound').onclick = () => setSound(!ui.muted);
$('speed').oninput = (e) => { ui.speed = Number(e.target.value); $('speed-value').textContent = ui.speed; };
$('agent').onchange = async (e) => {
  ui.keyboard = e.target.value === 'keyboard';
  if (ui.keyboard) ui.running = false;
  else apply(await api('/api/agent', { name: e.target.value }));
  updateHud();
  e.target.blur(); // so the arrow keys drive the guardian, not the list
};
document.querySelectorAll('.cams button').forEach((b) => { b.onclick = () => setCamera(Number(b.dataset.cam)); });

const KEY_ACTIONS = {
  ArrowUp: 0, KeyW: 0, ArrowDown: 1, KeyS: 1, ArrowLeft: 2, KeyA: 2, ArrowRight: 3, KeyD: 3, Space: 4,
};

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code in KEY_ACTIONS) {
    e.preventDefault();
    if (state && state.status === 'running') {
      ui.running = false; // pressing a key takes over from the agent
      step(KEY_ACTIONS[e.code]);
    }
  } else if (e.code === 'KeyR') reset();
  else if (e.code === 'KeyP' || e.code === 'Enter') { ui.running = !ui.running; updateHud(); }
  else if (e.code === 'KeyM') setSound(!ui.muted);
  else if (e.code === 'KeyG') world.toggleGrid();
  else if (e.code >= 'Digit1' && e.code <= 'Digit5') setCamera(Number(e.code.slice(5)) - 1);
});

// main loop

let last = performance.now();

function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  if (state && ui.running && !ui.keyboard && !ui.busy) {
    if (state.status === 'running') {
      ui.timer += dt;
      if (ui.timer >= 1 / ui.speed) {
        ui.timer = 0;
        step();
      }
    } else {
      // show the result for a moment, then start the next episode
      ui.endTimer += dt;
      if (ui.endTimer > 2.5) {
        ui.endTimer = 0;
        if (state.episode < state.max_episodes) reset();
        else {
          ui.running = false;
          ui.finished = true;
          updateHud();
        }
      }
    }
  }

  world.frame(dt);
  requestAnimationFrame(loop);
}

async function start() {
  $('speed').value = ui.speed;
  $('speed-value').textContent = ui.speed;
  setSound(ui.muted);
  const first = await api('/api/state');
  await world.init(first);
  setCamera(Number(params.get('cam')) || 0);
  apply(first);
  $('loading').classList.add('done');
  requestAnimationFrame(loop);
}

start();
