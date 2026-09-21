// Client loop: keep a websocket to `main.py serve`, hand each tick to the scene and HUD,
// and relay controls back. Rendering interpolates between the last two 20 Hz ticks so a
// 60 Hz screen sees smooth motion at one tick of latency.
import { Scene } from '/scene.mjs';
import { hud } from '/hud.mjs';

const $ = id => document.getElementById(id);
const PREFS_KEY = 'carnav-viewer-prefs';
const prefs = (() => { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}'); } catch { return {}; } })();
const savePrefs = () => { try { localStorage.setItem(PREFS_KEY, JSON.stringify({ view: $('viewSel').value, overlays: $('overlayChk').checked, auto: $('autoChk').checked })); } catch {} };

const state = { ws: null, world: null, prev: null, curr: null, currAt: 0, tickMs: 50, lastIntent: null, keys: { up: false, down: false, left: false, right: false } };
const scene = new Scene($('scene'));
hud.init();
if (prefs.view) $('viewSel').value = prefs.view;
if (typeof prefs.overlays === 'boolean') $('overlayChk').checked = prefs.overlays;
if (typeof prefs.auto === 'boolean') $('autoChk').checked = prefs.auto;
// URL parameters override saved prefs: ?view=aerial to open in a view, ?debug=1 for scene diagnostics.
const params = new URLSearchParams(location.search);
if (params.get('view')) $('viewSel').value = params.get('view');
scene.view = $('viewSel').value; scene.overlays = $('overlayChk').checked;
if (params.has('debug')) {
  const el = $('debug'); el.hidden = false;
  setInterval(() => { el.textContent = JSON.stringify({ ...scene.diag(), tickMs: Math.round(state.tickMs), step: state.curr && state.curr.step, ego: state.curr && state.curr.ego }, null, 1); }, 400);
}

function send(obj) { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); }

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    hud.conn(true); hud.banner(null); send({ cmd: 'auto', on: $('autoChk').checked });
    // Deep-linkable for screenshotting/sharing a specific setup: ?world=rural&traffic=dense.
    if (params.get('world')) send({ cmd: 'world', world: params.get('world') });
    if (params.get('traffic')) send({ cmd: 'traffic', traffic: params.get('traffic') });
    if (params.has('pedestrians') || params.has('signs')) send({ cmd: 'street', pedestrians: params.get('pedestrians') === '1', signs: params.get('signs') === '1' });
  };
  ws.onclose = () => { hud.conn(false); hud.banner('Connection lost — retrying…', 'warn'); state.prev = state.curr = null; setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === 'reset') onReset(m); else if (m.type === 'tick') onTick(m);
  };
  state.ws = ws;
}

function onReset(m) {
  state.world = m; state.prev = state.curr = null; state.lastIntent = null;
  scene.buildWorld(m);
  hud.reset(m);
  $('seedInput').value = m.seed; $('trafficSel').value = m.traffic; $('worldSel').value = m.world;
  if (m.street) { $('pedChk').checked = m.street.pedestrians; $('signChk').checked = m.street.speed_signs; }
}

function onTick(m) {
  if (!state.world) return;
  const now = performance.now();
  if (state.curr && !state.curr.paused) state.tickMs = 0.8 * state.tickMs + 0.2 * Math.min(500, now - state.currAt);
  state.prev = state.curr; state.curr = m; state.currAt = now;
  hud.tick(m, state.world, state.tickMs);
  if (m.driver && m.driver.intent !== state.lastIntent) { hud.timelinePush(m.time, m.driver.intent); state.lastIntent = m.driver.intent; }
  $('manualChk').checked = m.manual; $('rateSel').value = String(m.rate); $('autoChk').checked = m.auto_restart;
  $('stepBtn').disabled = !m.paused;
}

const lerp = (a, b, t) => a + (b - a) * t;
function lerpAngle(a, b, t) { let d = b - a; d -= Math.round(d / (2 * Math.PI)) * 2 * Math.PI; return a + d * t; }
function viewState(now) {
  const c = state.curr, p = state.prev;
  if (!c) return null;
  if (!p || c.paused || p.step >= c.step || p.vehicles.x.length !== c.vehicles.x.length) return c;
  const t = Math.min(1, (now - state.currAt) / Math.max(16, state.tickMs));
  const ego = { ...c.ego, x: lerp(p.ego.x, c.ego.x, t), y: lerp(p.ego.y, c.ego.y, t), heading: lerpAngle(p.ego.heading, c.ego.heading, t), steer: lerp(p.ego.steer, c.ego.steer, t) };
  const n = c.vehicles.x.length, vx = new Array(n), vy = new Array(n), vh = new Array(n);
  for (let i = 0; i < n; i++) { vx[i] = lerp(p.vehicles.x[i], c.vehicles.x[i], t); vy[i] = lerp(p.vehicles.y[i], c.vehicles.y[i], t); vh[i] = lerpAngle(p.vehicles.heading[i], c.vehicles.heading[i], t); }
  let pedestrians = c.pedestrians;
  if (p.pedestrians && c.pedestrians && p.pedestrians.x.length === c.pedestrians.x.length) {
    const k = c.pedestrians.x.length, px = new Array(k), py = new Array(k);
    for (let i = 0; i < k; i++) { px[i] = lerp(p.pedestrians.x[i], c.pedestrians.x[i], t); py[i] = lerp(p.pedestrians.y[i], c.pedestrians.y[i], t); }
    pedestrians = { ...c.pedestrians, x: px, y: py };
  }
  return { ...c, ego, vehicles: { x: vx, y: vy, heading: vh, speed: c.vehicles.speed }, pedestrians };
}

const feedEls = Object.fromEntries([...document.querySelectorAll('.feed')].map(el => [el.dataset.cam, el.querySelector('.feed-view')]));
function feedRects() { const out = {}; for (const [name, el] of Object.entries(feedEls)) out[name] = el.getBoundingClientRect(); return out; }
function fitLayout() {
  const topbarH = document.querySelector('.topbar').offsetHeight;
  const available = window.innerHeight - topbarH - 40 - 40 - 96;
  const tileW = Math.max(150, Math.min(236, Math.floor((available / 4 - 6) * 16 / 9)));
  document.querySelector('.ui').style.setProperty('--feedsW', `${tileW}px`);
}
fitLayout();

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  const v = viewState(now);
  if (v && state.world) { scene.update(v, state.world, dt); scene.render(feedRects()); }
  requestAnimationFrame(frame);
}

// Recording: captures the whole tab (3D view + HUD) via getDisplayMedia, not
// just the <canvas> -- a plain canvas.captureStream() would drop every panel
// drawn in the DOM/CSS layer (reward, speed, camera feeds, minimap), which is
// most of what makes a "look what it can do" clip worth showing. The tradeoff
// is the one-time browser share-this-tab prompt; Chrome's `preferCurrentTab`
// hint skips straight to that choice instead of making the user hunt for it.
const rec = { recorder: null, stream: null, chunks: [], startedAt: 0, timer: null };

function fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
    hud.banner('This browser cannot record (no tab/screen capture support).', 'warn');
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false, preferCurrentTab: true });
  } catch {
    return;   // user dismissed the share picker -- not an error worth a banner
  }
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    .find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
  if (!mime) {
    hud.banner('This browser cannot record (no supported video codec).', 'warn');
    stream.getTracks().forEach(t => t.stop());
    return;
  }
  rec.chunks = [];
  rec.recorder = new MediaRecorder(stream, { mimeType: mime });
  rec.recorder.ondataavailable = e => { if (e.data.size) rec.chunks.push(e.data); };
  rec.recorder.onstop = saveRecording;
  rec.stream = stream;
  // The browser's own "Stop sharing" control lives outside our UI and ends the
  // track directly -- listen for that too, or the button/timer would go stale.
  stream.getVideoTracks()[0].addEventListener('ended', stopRecording);
  rec.recorder.start();
  rec.startedAt = performance.now();
  $('recordBtn').textContent = 'Stop'; $('recordBtn').classList.add('recording');
  $('recTime').hidden = false;
  rec.timer = setInterval(() => { $('recTime').textContent = fmtElapsed(performance.now() - rec.startedAt); }, 500);
}

function stopRecording() {
  if (rec.recorder && rec.recorder.state !== 'inactive') rec.recorder.stop();
  if (rec.stream) rec.stream.getTracks().forEach(t => t.stop());
  clearInterval(rec.timer); rec.timer = null;
  $('recordBtn').textContent = 'Record'; $('recordBtn').classList.remove('recording');
  $('recTime').hidden = true;
}

function saveRecording() {
  if (!rec.chunks.length) return;
  const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType || 'video/webm' });
  const world = state.world || {};
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `carnav-${world.world || 'city'}-seed${world.seed ?? 0}-${stamp}.webm`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  rec.chunks = [];
}

$('recordBtn').addEventListener('click', () => {
  if (rec.recorder && rec.recorder.state === 'recording') stopRecording(); else startRecording();
});

// Controls.
$('startBtn').addEventListener('click', () => { if (state.curr && state.curr.done) send({ cmd: 'new' }); else send({ cmd: 'toggle' }); });
$('stepBtn').addEventListener('click', () => send({ cmd: 'step' }));
$('restartBtn').addEventListener('click', () => send({ cmd: 'reset', seed: Number($('seedInput').value) || 0 }));
$('newBtn').addEventListener('click', () => send({ cmd: 'new' }));
$('trafficSel').addEventListener('change', e => send({ cmd: 'traffic', traffic: e.target.value }));
$('worldSel').addEventListener('change', e => send({ cmd: 'world', world: e.target.value }));
const sendStreet = () => send({ cmd: 'street', pedestrians: $('pedChk').checked, signs: $('signChk').checked });
$('pedChk').addEventListener('change', sendStreet);
$('signChk').addEventListener('change', sendStreet);
$('seedInput').addEventListener('change', e => send({ cmd: 'reset', seed: Number(e.target.value) || 0 }));
$('rateSel').addEventListener('change', e => send({ cmd: 'rate', rate: Number(e.target.value) }));
$('viewSel').addEventListener('change', e => { scene.view = e.target.value; scene.first = true; savePrefs(); });
$('overlayChk').addEventListener('change', e => { scene.overlays = e.target.checked; savePrefs(); });
$('manualChk').addEventListener('change', e => send({ cmd: 'mode', manual: e.target.checked }));
$('autoChk').addEventListener('change', e => { send({ cmd: 'auto', on: e.target.checked }); savePrefs(); });

const ARROWS = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
function setKey(name, down) { if (state.keys[name] === down) return; state.keys[name] = down; send({ cmd: 'keys', ...state.keys }); }
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (ARROWS[e.code]) { e.preventDefault(); if ($('manualChk').checked) setKey(ARROWS[e.code], true); return; }
  if (e.code === 'Space') { e.preventDefault(); $('startBtn').click(); }
  else if (e.key === '.') send({ cmd: 'step' });
  else if (e.key === 'r') $('restartBtn').click(); else if (e.key === 'n') $('newBtn').click();
  else if (e.key === 'm') { $('manualChk').checked = !$('manualChk').checked; $('manualChk').dispatchEvent(new Event('change')); }
  else if (e.key === 'o') { $('overlayChk').checked = !$('overlayChk').checked; $('overlayChk').dispatchEvent(new Event('change')); }
  else if (e.key === 'v') { const s = $('viewSel'); s.selectedIndex = (s.selectedIndex + 1) % s.options.length; s.dispatchEvent(new Event('change')); }
});
window.addEventListener('keyup', e => { if (ARROWS[e.code]) setKey(ARROWS[e.code], false); });
window.addEventListener('blur', () => { for (const k of Object.keys(state.keys)) setKey(k, false); });
window.addEventListener('resize', () => { scene.resize(); fitLayout(); });

connect();
requestAnimationFrame(frame);
