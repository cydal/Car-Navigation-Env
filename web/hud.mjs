// DOM side of the viewer: speed, driver intent and speed caps, reward and its components,
// sensor summaries, minimap, route progress, decision timeline, banners and the end card.
import { FEEDS } from '/scene.mjs';

const $ = id => document.getElementById(id);
const fmt = (v, d = 0) => (v === null || v === undefined || !Number.isFinite(v)) ? '—' : Number(v).toFixed(d);
const COMPASS = ['front', 'front-right', 'right', 'rear-right', 'rear', 'rear-left', 'left', 'front-left'];
const compass = rad => COMPASS[((Math.round(rad / (Math.PI / 4)) % 8) + 8) % 8];
const INTENT = { cruise: 'Cruise', traffic: 'Follow the traffic', clearance: 'Hold back for clearance', turn: 'Ease off while steering', signal: 'Stop for the signal', manual: 'Manual drive', no_safe_heading: 'No safe heading', starting: 'Starting' };
const CAPS = ['cruise', 'clearance', 'turn', 'signal', 'traffic'];
const COMPS = { progress: 'progress', time: 'time', target_bonus: 'waypoint', red_light: 'red light', crash: 'crash' };
const ENDINGS = { success: ['Route complete', 'good'], crash: ['Crashed', 'bad'], stuck: ['Stuck', 'bad'], timeout: ['Out of time', 'bad'] };

export const hud = {
  init() {
    $('capsList').innerHTML = CAPS.map(c => `<li data-c="${c}"><span>${c}</span><span class="bar"><i></i></span><span class="val">—</span></li>`).join('');
    $('compList').innerHTML = Object.entries(COMPS).map(([k, label]) => `<li data-c="${k}"><span>${label}</span><span class="bar"><i></i></span><span class="val">—</span></li>`).join('');
    this.timeline = []; this.mapBase = null; this.mapExtent = [1, 1];
  },
  conn(on) { $('connDot').className = `dot ${on ? 'live' : 'error'}`; $('connText').textContent = on ? 'live' : 'offline'; },
  banner(text, kind = 'info') { const b = $('banner'); if (!text) { b.hidden = true; return; } b.hidden = false; b.textContent = text; b.className = `banner ${kind}`; },

  reset(world) {
    const { width: W, height: H, grid, tile_size } = world.city;
    const off = document.createElement('canvas'); off.width = W; off.height = H;
    const g = off.getContext('2d'), img = g.createImageData(W, H);
    for (let i = 0; i < W * H; i++) { const b = grid.charCodeAt(i) === 49, o = i * 4; img.data[o] = b ? 76 : 30; img.data[o + 1] = b ? 73 : 34; img.data[o + 2] = b ? 70 : 40; img.data[o + 3] = 255; }
    g.putImageData(img, 0, 0);
    this.mapBase = off; this.mapExtent = [W * tile_size, H * tile_size];
    this.timeline = []; $('timeline').innerHTML = '';
    $('wpDots').innerHTML = Array.from({ length: world.cfg.n_targets }, () => '<i></i>').join('');
    $('runInfo').textContent = `seed ${world.seed} · ${world.traffic} traffic · episode ${world.episode}`;
    $('statEp').textContent = world.episode;
    $('intentBig').textContent = 'starting'; $('intentBig').dataset.m = ''; $('execLine').textContent = '';
    $('endCard').hidden = true;
  },

  tick(m, world, tickMs) {
    const e = m.ego, info = m.info || {}, d = m.driver || {};
    $('driverDot').className = 'dot live';
    $('tickRate').textContent = `${fmt(1000 / Math.max(1, tickMs), 0)} Hz${m.rate !== 1 ? ` · ${m.rate}×` : ''}`;
    const chip = $('modeChip'); chip.textContent = m.manual ? 'manual' : 'scripted'; chip.classList.toggle('manual', m.manual);

    // Speed and the driver's target.
    const kmh = e.speed * 3.6, cruise = (d.cruise ?? world.cfg.cruise_speed) * 3.6;
    $('speedNum').textContent = Math.round(kmh);
    const ring = $('targetRing');
    ring.textContent = d.mode === 'scripted' && d.target_speed !== undefined ? Math.round(d.target_speed * 3.6) : '—';
    ring.classList.toggle('over', kmh > cruise + 2);
    $('postedNote').textContent = d.mode === 'scripted' ? `driver target · cruise ${Math.round(cruise)}` : 'manual · arrows drive';

    // Intent.
    const intent = d.intent || 'starting';
    $('intentBig').textContent = INTENT[intent] || intent; $('intentBig').dataset.m = intent;
    if (d.mode === 'scripted' && typeof d.theta_deg === 'number') {
      const side = d.theta_deg > 2 ? 'right' : d.theta_deg < -2 ? 'left' : 'ahead';
      $('execLine').textContent = `heading ${Math.abs(d.theta_deg).toFixed(0)}° ${side} · target ${Math.round(d.target_speed * 3.6)} km/h · clearance ${fmt(d.chosen_clear, 0)} m (needs ${fmt(d.need, 0)} m)${d.in_corridor ? ' · centring in corridor' : ''}`;
    } else $('execLine').textContent = m.manual ? `throttle ${fmt(m.action[0], 2)} · steer ${fmt(m.action[2], 2)}` : '';
    for (const li of document.querySelectorAll('#capsList li')) {
      const cap = d.caps ? d.caps[li.dataset.c] : null, ok = cap !== null && cap !== undefined;
      li.querySelector('i').style.width = ok ? `${Math.round(Math.min(1, cap / Math.max(0.1, d.cruise || 1)) * 100)}%` : '0%';
      li.querySelector('.val').textContent = ok ? `${Math.round(cap * 3.6)}` : '—';
      li.classList.toggle('chosen', li.dataset.c === intent);
    }

    // Reward.
    $('epReward').textContent = fmt(info.episode_reward, 1);
    const sr = $('stepReward'); sr.textContent = `${m.reward >= 0 ? '+' : ''}${fmt(m.reward, 2)}`; sr.style.color = m.reward < -1 ? 'var(--red)' : m.reward > 1 ? 'var(--green)' : '';
    const comps = info.reward_components || {};
    for (const li of document.querySelectorAll('#compList li')) {
      const v = comps[li.dataset.c] ?? 0, bar = li.querySelector('i');
      bar.style.width = `${Math.min(100, Math.round(Math.abs(v) * 40))}%`; bar.className = v >= 0 ? 'pos' : 'neg';
      li.querySelector('.val').textContent = `${v >= 0 ? '+' : ''}${fmt(v, 2)}`;
      li.classList.toggle('live', Math.abs(v) > 1e-6);
    }
    $('statStep').textContent = m.step; $('statWp').textContent = `${info.targets_reached ?? 0}/${info.n_targets ?? world.cfg.n_targets}`;
    $('statRed').textContent = info.red_light_violations ?? 0;

    // Sensors.
    this.feeds(m);
    const kinds = world.vehicles.kind, rows = [];
    m.radar.dist.forEach((dist, i) => {
      if (dist >= world.cfg.radar_range - 0.05) return;
      const closing = m.radar.closing[i] * 3.6, k = m.radar.idx[i] >= 0 ? kinds[m.radar.idx[i]] : 'car';
      rows.push({ dist, html: `<i class="${dist < 12 ? 'red' : dist < 25 ? 'hot' : ''}">${compass(i * Math.PI / 4)} · ${k} ${dist.toFixed(1)} m ${closing > 0.5 ? '▼' : closing < -0.5 ? '▲' : '·'} ${Math.abs(closing).toFixed(0)} km/h</i>` });
    });
    rows.sort((a, b) => a.dist - b.dist);
    $('radarText').innerHTML = rows.length ? rows.slice(0, 4).map(r => r.html).join('') : '<i>no moving vehicle in range</i>';
    let kmin = 0; for (let k = 1; k < m.lidar.length; k++) if (m.lidar[k] < m.lidar[kmin]) kmin = k;
    const dmin = m.lidar[kmin], ahead = m.lidar[Math.floor(m.lidar.length / 2)];
    $('lidarText').innerHTML = `<i class="${dmin < 5 ? 'red' : dmin < 10 ? 'hot' : ''}">closest ${dmin.toFixed(1)} m ${compass(world.lidar_offsets[kmin])}</i><i>ahead ${ahead >= world.cfg.lidar_range - 0.05 ? '> ' + world.cfg.lidar_range : ahead.toFixed(1)} m</i>`;
    let best = null;
    world.city.signals.forEach(([x, y], i) => { const dd = Math.hypot(x - e.x, y - e.y); if (!best || dd < best.dd) best = { dd, i, x, y }; });
    if (best && m.lights[best.i]) {
      const axis = Math.abs(e.y - best.y) > Math.abs(e.x - best.x) ? 'ns' : 'ew', st = m.lights[best.i][axis];
      $('signalText').innerHTML = `<i class="${st === 'red' ? 'red' : st === 'yellow' ? 'hot' : 'green'}">${st} · ${best.dd.toFixed(0)} m · changes in ${(m.lights[best.i].left * world.cfg.dt).toFixed(1)} s</i>`;
    } else $('signalText').textContent = 'no signals on this map';

    // Route.
    const reached = info.targets_reached ?? 0;
    [...$('wpDots').children].forEach((dot, i) => { dot.className = i < reached ? 'done' : i === reached ? 'cur' : ''; });
    $('stepFill').style.width = `${Math.min(100, m.step / world.cfg.max_steps * 100)}%`;
    $('clock').textContent = `${fmt(m.time, 1)} s`;
    $('distText').textContent = `${fmt(info.dist_to_target, 1)} m to waypoint`;
    $('metrics').innerHTML = [['moving', world.vehicles.n_moving], ['parked', kinds.length - world.vehicles.n_moving], ['min lidar', `${dmin.toFixed(1)} m`], ['steer', `${(e.steer * 180 / Math.PI).toFixed(0)}°`], ['accel', `${fmt(e.accel, 1)} m/s²`]]
      .map(([k, v]) => `<span><b>${v}</b>${k}</span>`).join('');
    this.drawMap(m, world);
    this.timelineRender(m.time);
    this.startButton(m.paused, m.done);
    this.endCard(m, world);
  },

  feeds(m) {
    const e = m.ego, v = m.vehicles;
    for (const [name, f] of Object.entries(FEEDS)) {
      const half = f.hfov / 2 * Math.PI / 180; let n = 0;
      for (let i = 0; i < v.x.length; i++) {
        const dx = v.x[i] - e.x, dy = v.y[i] - e.y; if (Math.hypot(dx, dy) > f.range) continue;
        let b = Math.atan2(dy, dx) - e.heading - f.physYaw; b = Math.atan2(Math.sin(b), Math.cos(b));
        if (Math.abs(b) <= half) n++;
      }
      const el = document.querySelector(`.feed[data-cam="${name}"] em`); el.textContent = n; el.classList.toggle('some', n > 0);
    }
  },

  drawMap(m, world) {
    if (!this.mapBase) return;
    const cv = $('minimap'), ctx = cv.getContext('2d'), S = cv.width, [ex, ey] = this.mapExtent;
    const px = x => x / ex * S, py = y => y / ey * S;
    ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, S, S); ctx.drawImage(this.mapBase, 0, 0, S, S);
    world.city.signals.forEach(([x, y], i) => { const l = m.lights[i]; ctx.fillStyle = !l ? '#888' : l.ns === 'green' ? '#3fd08a' : l.ns === 'yellow' ? '#ffd22a' : '#ff4d3d'; ctx.fillRect(px(x) - 2, py(y) - 2, 4, 4); });
    world.targets.slice(m.target_idx).forEach(([x, y], i) => { ctx.fillStyle = i === 0 ? '#3fd08a' : '#ffb020'; ctx.fillRect(px(x) - 3, py(y) - 3, 6, 6); });
    // Other vehicles are deliberately not drawn here: at this scale their dots and
    // the ego arrow all read as "a dot on a map", and the one that matters gets
    // lost in the crowd. A white halo behind the amber arrow keeps it legible
    // over both the dark road and the lighter building fill.
    const e = m.ego, cx = px(e.x), cy = py(e.y), a = e.heading, s = 6;
    const tip = [cx + Math.cos(a) * s, cy + Math.sin(a) * s];
    const l1 = [cx + Math.cos(a + 2.5) * s * 0.8, cy + Math.sin(a + 2.5) * s * 0.8];
    const l2 = [cx + Math.cos(a - 2.5) * s * 0.8, cy + Math.sin(a - 2.5) * s * 0.8];
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(...tip); ctx.lineTo(...l1); ctx.lineTo(...l2); ctx.closePath(); ctx.stroke();
    ctx.fillStyle = '#ffb020';
    ctx.beginPath(); ctx.moveTo(...tip); ctx.lineTo(...l1); ctx.lineTo(...l2); ctx.closePath(); ctx.fill();
  },

  timelinePush(time, intent) { this.timeline.push({ t: time, m: intent }); if (this.timeline.length > 600) this.timeline.shift(); },
  timelineRender(now) {
    const span = 60, start = Math.max(0, now - span), segs = [];
    for (let i = 0; i < this.timeline.length; i++) {
      const a = this.timeline[i], b = this.timeline[i + 1], t0 = Math.max(a.t, start), t1 = Math.min(b ? b.t : now, now);
      if (t1 <= t0) continue;
      segs.push(`<i class="m-${a.m}" style="left:${(t0 - start) / span * 100}%;width:${(t1 - t0) / span * 100}%" title="${INTENT[a.m] || a.m} at ${a.t.toFixed(1)} s"></i>`);
    }
    $('timeline').innerHTML = segs.join('');
  },

  startButton(paused, done) { const b = $('startBtn'); b.textContent = done ? 'Next map' : paused ? 'Resume' : 'Pause'; b.classList.toggle('primary', paused || Boolean(done)); },
  endCard(m, world) {
    const card = $('endCard');
    if (!m.done) { card.hidden = true; return; }
    card.hidden = false;
    const info = m.info || {}, [title, cls] = ENDINGS[m.done] || [m.done, 'bad'];
    $('endTitle').textContent = m.done === 'crash' && info.crash_with ? `Crashed into a ${info.crash_with}` : title; $('endTitle').className = cls;
    $('endBody').innerHTML = [[fmt(m.time, 1), 's'], [m.step, 'steps'], [`${info.targets_reached ?? 0}/${info.n_targets ?? world.cfg.n_targets}`, 'waypoints'], [fmt(info.episode_reward, 1), 'reward'], [info.red_light_violations ?? 0, 'red lights']]
      .map(([v, k]) => `<span><b>${v}</b>${k}</span>`).join('');
    $('endNote').textContent = m.restart_in === null || m.restart_in === undefined ? 'Auto-restart is off. Press Restart or New map.' : `New map in ${Math.ceil(m.restart_in)} s…`;
  },
};
