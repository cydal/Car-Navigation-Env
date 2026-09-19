// Three.js presentation of the streamed simulation. Nothing here is simulation: every
// object is a function of the last `reset` and `tick` messages from serve/server.py.
//
// Frames. Physics: x right, y "down" the map (south), heading 0 = +x, clockwise positive.
// Three: x right, y up, z toward the viewer. Physics (x, y) maps to three (x, h, y), so a
// heading theta points along (cos t, 0, sin t), which is rotation.y = -theta for a model
// authored nose +x. Right-of-travel in physics is (-dy, dx): facing +x that is +y = three +z.
import * as THREE from 'three';
import { GLTFLoader } from '/vendor/GLTFLoader.js';

const V = (x, y, h = 0) => new THREE.Vector3(x, h, y);
const hash01 = (a, b) => { const h = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const srgb = (r, g, b) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);

const C = {
  kerb: 0xa9adb3, white: 0xf2f2ee, amber: 0xf1d67a,
  ego: 0xffb020, target: 0x3fd08a, next: 0xffb020,
  mast: 0x2a2d33, plate: 0x14161a, pole: 0x6f757d, lampGlow: 0xfff0c8,
  lampOn: [0xff2a1a, 0xffd22a, 0x2aff44], lampOff: [0x3a0d0b, 0x3a300b, 0x0b3a10],
  near: new THREE.Color(0xff4d3d), far: new THREE.Color(0x3fd08a),
};
const HEIGHT = 1.35;   // body height every vehicle mesh is fitted to (matches the physics box)
const SENSOR_COLORS = { front: 0x2ad1c9, left: 0x7fd8ff, right: 0x7fd8ff, rear: 0xb59cff };

// World/terrain presets (server-selected, named in `city.theme`). Each one is just a
// different bundle of appearance choices over the *same* tile grid `buildCity` always
// gets -- picking a theme never changes the road network, only what gets drawn on top
// of the BUILDING tiles (a city tower, a house with a garden, nothing but open field
// with the odd barn, or a warehouse) and the sky/fog mood.
//
// `shape` selects which branch of buildCity's per-tile switch runs:
//   tower      city skyscraper: fills the tile, windowed facade, flat roof
//   house      suburb home: smaller than the tile (garden shows through), pitched roof
//   warehouse  flat-roofed industrial shed, no windows, filling most of the tile
//   farm       rural: most tiles get NOTHING (the ground plane's field colour shows
//              through directly), a sparse subset get a barn or a silo
const THEMES = {
  city: {
    sky: [0x5c8cc7, 0xdfe8f1], fog: 0xd6e0ea, fogRange: [70, 380], ground: 0x7d9963,
    shape: 'tower', heightRange: [6, 24],
    palette: [0xd9d0c1, 0xc7ced6, 0xb8a798, 0xa4b2be, 0xcfc0ad, 0x9aa3ae, 0xd3c6b4, 0xb1bcc4],
    curbsLamps: true,
  },
  suburbs: {
    sky: [0x74a8d8, 0xe8eef5], fog: 0xdfe6ee, fogRange: [60, 320], ground: 0x8bab5e,
    shape: 'house', heightRange: [2.6, 4.4], footprint: [0.55, 0.78],
    palette: [0xe8d9c3, 0xd9c2b0, 0xc9d6df, 0xefe3d0, 0xd8cdb8, 0xc7b8a0],
    roofPalette: [0x8a3b2c, 0x6f4a33, 0x4a5a63, 0x7a4a2f],
    curbsLamps: true,
  },
  rural: {
    sky: [0x8bb6dd, 0xf3ecd2], fog: 0xe9dfc0, fogRange: [90, 420], ground: 0x8a9c4a,
    shape: 'farm', farmDensity: 0.16,
    curbsLamps: false,
  },
  industrial: {
    sky: [0x8b98a3, 0xcfd6da], fog: 0xb9c2c8, fogRange: [55, 300], ground: 0x6e6a63,
    shape: 'warehouse', heightRange: [5, 10],
    palette: [0x8d9096, 0x7a828c, 0x9c8468, 0x6f7680, 0xab7248],
    curbsLamps: true,
  },
};

// Camera mounts in the ego frame (x forward, z right). `yaw` turns a camera's default -z gaze
// onto the wanted direction; `physYaw` is the same direction as a physics bearing for the HUD.
export const FEEDS = {
  front: { pos: [1.8, 1.2, 0], yaw: -Math.PI / 2, hfov: 70, range: 200, physYaw: 0 },
  left: { pos: [0.3, 1.0, -0.9], yaw: 0, hfov: 120, range: 30, physYaw: -Math.PI / 2 },
  right: { pos: [0.3, 1.0, 0.9], yaw: Math.PI, hfov: 120, range: 30, physYaw: Math.PI / 2 },
  rear: { pos: [-2.1, 1.0, 0], yaw: Math.PI / 2, hfov: 100, range: 70, physYaw: Math.PI },
};

function flatGeom(w, l) { const g = new THREE.PlaneGeometry(w, l); g.rotateX(-Math.PI / 2); return g; }

function skyDome(topHex, horizonHex) {
  const R = 650, g = new THREE.SphereGeometry(R, 32, 16), pos = g.attributes.position;
  const col = new Float32Array(pos.count * 3), top = new THREE.Color(topHex), hor = new THREE.Color(horizonHex), c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) { c.copy(hor).lerp(top, clamp((pos.getY(i) / R + 0.05) / 0.5, 0, 1)); col.set([c.r, c.g, c.b], i * 3); }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false, toneMapped: false }));
}

// items: [{ pos, scale?, rotY?, color? }]
function instanced(geom, material, items, cast = false) {
  const m = new THREE.InstancedMesh(geom, material, items.length);
  const mat = new THREE.Matrix4(), q = new THREE.Quaternion(), one = new THREE.Vector3(1, 1, 1), up = new THREE.Vector3(0, 1, 0);
  items.forEach((it, i) => { q.setFromAxisAngle(up, it.rotY || 0); mat.compose(it.pos, q, it.scale || one); m.setMatrixAt(i, mat); if (it.color) m.setColorAt(i, it.color); });
  if (m.instanceColor) m.instanceColor.needsUpdate = true;
  m.castShadow = cast; m.receiveShadow = true; m.frustumCulled = false;
  return m;
}

// Facades get a window grid computed from world position in the shader, so every tile
// box -- whatever its height -- shows 3 m storeys and 2.6 m bays without any texture or
// per-instance UV work. Windows are glass-tinted; a few per cent glow warm.
function facadeMaterial() {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.0 });
  m.onBeforeCompile = shader => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNormal;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vec4 wp4 = vec4( transformed, 1.0 );
        vec3 wn = normal;
        #ifdef USE_INSTANCING
          wp4 = instanceMatrix * wp4;
          wn = mat3( instanceMatrix ) * wn;
        #endif
        vWPos = ( modelMatrix * wp4 ).xyz;
        vWNormal = normalize( mat3( modelMatrix ) * wn );`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vWPos;
        varying vec3 vWNormal;
        float winHash( vec2 p ) { return fract( sin( dot( p, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ); }
        // 0 = wall, 1 = glass, 2 = lit glass
        float windowKind( out vec2 cell ) {
          vec3 n = normalize( vWNormal );
          if ( abs( n.y ) > 0.5 || vWPos.y < 1.2 ) return 0.0;
          float u = abs( n.x ) > 0.5 ? vWPos.z : vWPos.x;
          float v = vWPos.y - 1.2;
          vec2 f = vec2( fract( u / 2.6 ), fract( v / 3.0 ) );
          cell = vec2( floor( u / 2.6 ), floor( v / 3.0 ) ) + ( abs( n.x ) > 0.5 ? vec2( 1000.0 * sign( n.x ), 0.0 ) : vec2( 0.0, 1000.0 * sign( n.z ) ) );
          if ( f.x < 0.24 || f.x > 0.76 || f.y < 0.28 || f.y > 0.78 ) return 0.0;
          return winHash( cell ) > 0.9 ? 2.0 : 1.0;
        }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        vec2 winCell; float winK = windowKind( winCell );
        if ( winK > 0.5 ) diffuseColor.rgb = winK > 1.5 ? vec3( 1.0, 0.86, 0.58 ) : mix( vec3( 0.13, 0.17, 0.23 ), diffuseColor.rgb, 0.25 );`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        if ( winK > 1.5 ) totalEmissiveRadiance += vec3( 0.9, 0.7, 0.4 );`);
  };
  return m;
}

function disposeGroup(g) {
  g.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(mt => mt.dispose()); });
  g.clear();
}

// Fit a loaded glTF car to the physics footprint: nose from glTF -z onto +x, base on the
// ground, each axis scaled to its own physical size so a square-footprint model does not
// turn into a slab (the same per-axis rule the Panda renderer uses).
function fitVehicle(model, length, width) {
  model.updateMatrixWorld(true);
  model.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });
  const box = new THREE.Box3().setFromObject(model), size = box.getSize(new THREE.Vector3());
  model.position.set(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
  const inner = new THREE.Group(); inner.add(model); inner.rotation.y = Math.PI / 2;
  const holder = new THREE.Group(); holder.add(inner);
  holder.scale.set(length / Math.max(size.z, 1e-3), HEIGHT / Math.max(size.y, 1e-3), width / Math.max(size.x, 1e-3));
  return holder;
}
const PEOPLE_COLORS = [0xd9534f, 0x3d7dd9, 0x3fb27f, 0xf0a030, 0x8f5fd1, 0x2b2f36];
const _signTex = new Map();
function signTexture(limit) {
  if (_signTex.has(limit)) return _signTex.get(limit);
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#ffffff'; g.beginPath(); g.arc(128, 128, 124, 0, Math.PI * 2); g.fill();
  g.lineWidth = 26; g.strokeStyle = '#d0342c'; g.beginPath(); g.arc(128, 128, 108, 0, Math.PI * 2); g.stroke();
  g.fillStyle = '#111'; g.font = 'bold 118px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(limit), 128, 136);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; _signTex.set(limit, t); return t;
}
// A walking figure: legs, torso, head. Authored facing local +x like every vehicle mesh.
function makePerson(color) {
  const g = new THREE.Group();
  const legs = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.19, 0.8, 10), new THREE.MeshStandardMaterial({ color: 0x2c3e50 })); legs.position.y = 0.4; g.add(legs);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 0.45, 4, 10), new THREE.MeshStandardMaterial({ color })); torso.position.y = 1.08; torso.castShadow = true; g.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 12, 10), new THREE.MeshStandardMaterial({ color: 0xe0b89a, roughness: 0.8 })); head.position.y = 1.6; g.add(head);
  return g;
}

function fallbackVehicle(length, width, color = 0x8a919b) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.BoxGeometry(length, HEIGHT, width), new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.3 }));
  m.position.y = HEIGHT / 2; m.castShadow = true; g.add(m); return g;
}

export class Scene {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.12;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.autoClear = false;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(THEMES.city.fog, ...THEMES.city.fogRange);
    this.themeName = null;
    this.sky = skyDome(...THEMES.city.sky); this.scene.add(this.sky);
    this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x7a746a, 1.55));
    this.sun = new THREE.DirectionalLight(0xfff0dc, 1.8); this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048); this.sun.shadow.camera.near = 10; this.sun.shadow.camera.far = 260;
    Object.assign(this.sun.shadow.camera, { left: -80, right: 80, top: 80, bottom: -80 });
    this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.05;
    this.scene.add(this.sun, this.sun.target);
    this.camera = new THREE.PerspectiveCamera(58, 1, 0.5, 900);
    this.camPos = new THREE.Vector3(0, 6, 14); this.camTarget = new THREE.Vector3(); this.first = true;
    this.view = 'chase'; this.overlays = true;

    this.worldGroup = new THREE.Group(); this.scene.add(this.worldGroup);
    this.dynGroup = new THREE.Group(); this.scene.add(this.dynGroup);
    this.loader = new GLTFLoader(); this.protos = new Map(); this.worldId = 0;
    this.vehicles = []; this.lamps = []; this.markers = []; this.markerGeos = [];

    this.ego = new THREE.Group(); this.scene.add(this.ego);
    this.egoBuilt = false; this.tail = [];
    this.feedCams = {};
    for (const [name, f] of Object.entries(FEEDS)) {
      const c = new THREE.PerspectiveCamera(60, 16 / 9, 0.3, 500);
      c.position.set(...f.pos); c.rotation.y = f.yaw; c.userData.hfov = f.hfov;
      this.ego.add(c); this.feedCams[name] = c;
    }

    this.overlay = new THREE.Group(); this.scene.add(this.overlay);
    this.lidar = null; this.lidarOffsets = [];
    const line = color => new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
    this.intentLine = line(C.ego); this.goalLine = line(0x2ad1c9);
    this.overlay.add(this.intentLine, this.goalLine);
    this.resize();
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------ world
  buildWorld(m) {
    this.worldId++;
    disposeGroup(this.worldGroup);
    this.dynGroup.clear();                 // vehicle clones share the cached prototypes: never dispose
    for (const mk of this.markers) { this.scene.remove(mk.g); mk.ring.material.dispose(); mk.beam.material.dispose(); }
    for (const g of this.markerGeos) g.dispose();
    this.markers = []; this.markerGeos = []; this.vehicles = []; this.lamps = [];
    this.buildCity(m.city);
    this.buildSignals(m.city);
    this.buildMarkers(m);
    this.buildVehicles(m);
    this.buildStreet(m);
    this.buildEgo(m);
    this.buildLidar(m);
    if (m.cameras) {
      this.setCameraSpecs(m.cameras);
      this.buildFovWedges(m.cameras);   // built once; the rig never changes between resets
    }
    this.buildDetectionOutlines();
    this.first = true;
  }

  // Mount the 4 feed cameras (and, below, the FOV wedges) at the exact specs
  // `env/perception.py` used for its occlusion scan -- sent once in the reset
  // message -- instead of a second hand-copied constant here that could
  // quietly drift from the one driving the actual detections.
  setCameraSpecs(specs) {
    for (const [name, spec] of Object.entries(specs)) {
      const cam = this.feedCams[name];
      if (!cam) continue;
      const height = (FEEDS[name] && FEEDS[name].pos[1]) || 1.1;
      cam.position.set(spec.forward, height, spec.right);
      // The camera's default gaze is local -Z; this is the rotation.y that
      // points it along physics bearing `spec.yaw` relative to the ego's own
      // heading (derived the same way every other mesh's -heading rotation
      // is: solving Ry(theta)*(0,0,-1) = (cos yaw, 0, sin yaw) for theta).
      cam.rotation.y = Math.atan2(-Math.cos(spec.yaw), -Math.sin(spec.yaw));
      cam.userData.hfov = spec.fov;
    }
  }

  buildFovWedges(specs) {
    if (this.fovWedgeGroup) return;                    // same rig every reset; build once
    const g = new THREE.Group();
    for (const [name, spec] of Object.entries(specs)) {
      const half = THREE.MathUtils.degToRad(spec.fov) / 2, r = Math.min(spec.range, 40), N = 24;
      const pos = [];
      for (let i = 0; i < N; i++) {
        const a0 = -half + (2 * half) * i / N, a1 = -half + (2 * half) * (i + 1) / N;
        // Local x = forward, z = right (matches every vehicle mesh's own axes),
        // y = 0 -- already flat on the ground, no extra rotation needed.
        pos.push(0, 0, 0, Math.cos(a0) * r, 0, Math.sin(a0) * r, Math.cos(a1) * r, 0, Math.sin(a1) * r);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: SENSOR_COLORS[name] || 0xffffff, transparent: true, opacity: 0.14,
        side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
      const holder = new THREE.Group(); holder.add(mesh);
      holder.position.set(spec.forward, 0.04, spec.right);
      holder.rotation.y = -spec.yaw;                    // same "-angle" convention as every heading rotation
      g.add(holder);
    }
    this.fovWedgeGroup = g;
    this.ego.add(g);                                    // rides along with the car for free
  }

  buildDetectionOutlines() {
    if (this.outlinePool) return;
    this.outlinePool = [];
    const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < 24; i++) {
      const mesh = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.9 }));
      mesh.visible = false;
      this.overlay.add(mesh);
      this.outlinePool.push(mesh);
    }
  }

  // Outline whichever vehicles `env/perception.py` actually detected this tick,
  // coloured by the (first) camera that sees each one -- a visible answer to
  // "what does the car's camera see", not just a badge count on the feed tile.
  updateDetections(view, world) {
    if (!this.outlinePool) return;
    for (const o of this.outlinePool) o.visible = false;
    if (!this.overlays || !view.perception) return;
    const seen = new Map();
    for (const [name, dets] of Object.entries(view.perception.cameras || {})) {
      for (const d of dets) if (!seen.has(d.id)) seen.set(d.id, SENSOR_COLORS[name] ?? 0xffffff);
    }
    const v = world.vehicles;
    let k = 0;
    for (const [idx, color] of seen) {
      if (k >= this.outlinePool.length || idx >= view.vehicles.x.length) continue;
      const o = this.outlinePool[k++];
      const len = v.length[idx] ?? 4.4, wid = v.width[idx] ?? 1.9;
      o.scale.set(wid + 0.2, HEIGHT + 0.15, len + 0.2);
      o.position.set(view.vehicles.x[idx], (HEIGHT + 0.15) / 2, view.vehicles.y[idx]);
      o.rotation.y = -view.vehicles.heading[idx];   // same "-heading" convention as the vehicle mesh itself
      o.material.color.setHex(color);
      o.visible = true;
    }
  }

  // Swaps the sky dome and fog to match a theme; a no-op past the first call for a given
  // theme, so switching worlds every reset doesn't rebuild the (fairly large) sky sphere
  // when the player is just cycling seeds within the same world.
  applyTheme(themeName) {
    const t = THEMES[themeName] || THEMES.city;
    if (this.themeName !== themeName) {
      this.themeName = themeName;
      this.scene.remove(this.sky); this.sky.geometry.dispose(); this.sky.material.dispose();
      this.sky = skyDome(...t.sky); this.scene.add(this.sky);
    }
    this.scene.fog.color.setHex(t.fog);
    [this.scene.fog.near, this.scene.fog.far] = t.fogRange;
    return t;
  }

  buildCity(city) {
    const { width: W, height: H, tile_size: ts, grid, v_roads, h_roads, road_width: rw } = city;
    const solid = (r, c) => r < 0 || r >= H || c < 0 || c >= W || grid.charCodeAt(r * W + c) === 49;
    const g = this.worldGroup, box = new THREE.BoxGeometry(1, 1, 1);
    const std = extra => new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, ...extra });
    const theme = this.applyTheme(city.theme || 'city');

    const ground = new THREE.Mesh(flatGeom(W * ts * 4, H * ts * 4), std({ color: theme.ground, roughness: 1 }));
    ground.position.set(W * ts / 2, -0.03, H * ts / 2); ground.receiveShadow = true; g.add(ground);

    // `towerBoxes` gets the windowed shader (city skyscrapers only); `plainBoxes` covers
    // every other solid-colour building body (house / warehouse / barn) so those themes
    // never pay for, or accidentally show, the window pattern.
    const towerBoxes = [], plainBoxes = [], flatRoofs = [], pitchRoofs = [];
    const silos = [], siloCaps = [];
    const roads = [], kerbs = [], posts = [];
    const [hMin, hMax] = theme.heightRange || [6, 24];
    const SIDES = [[1, 0, 0, 1], [-1, 0, 0, -1], [0, 1, 1, 0], [0, -1, -1, 0]];   // [dr, dc, nx, ny]
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const cx = (c + 0.5) * ts, cy = (r + 0.5) * ts;
      if (!solid(r, c)) {
        const tone = 0.27 + 0.07 * hash01(r * 7 + 1, c * 13 + 3);
        roads.push({ pos: V(cx, cy, 0), color: srgb(tone, tone * 1.03, tone * 1.08) });
        continue;
      }
      const shade = 0.42 + 0.30 * hash01(c, r);
      const h = hMin + (hMax - hMin) * hash01(r, c);
      if (theme.shape === 'tower') {
        const base = new THREE.Color(theme.palette[Math.floor(hash01(r * 3 + 1, c * 7 + 2) * theme.palette.length)]).multiplyScalar(0.8 + 0.45 * shade);
        towerBoxes.push({ pos: V(cx, cy, h / 2), scale: new THREE.Vector3(ts, h, ts), color: base });
        flatRoofs.push({ pos: V(cx, cy, h + 0.1), scale: new THREE.Vector3(ts - 0.5, 0.2, ts - 0.5), color: base.clone().multiplyScalar(1.12) });
      } else if (theme.shape === 'house') {
        const [fMin, fMax] = theme.footprint;
        const fx = ts * (fMin + (fMax - fMin) * hash01(c + 5, r + 9));   // smaller than the tile -- the gap reads as a garden
        const fz = fx * 0.85;
        const base = new THREE.Color(theme.palette[Math.floor(hash01(r * 3 + 1, c * 7 + 2) * theme.palette.length)]).multiplyScalar(0.85 + 0.3 * shade);
        plainBoxes.push({ pos: V(cx, cy, h / 2), scale: new THREE.Vector3(fx, h, fz), color: base });
        const roofCol = new THREE.Color(theme.roofPalette[Math.floor(hash01(r + 2, c + 3) * theme.roofPalette.length)]);
        pitchRoofs.push({ pos: V(cx, cy, h + fx * 0.32), scale: new THREE.Vector3(fx * 0.98, fx * 0.62, fz * 0.98), rotY: Math.PI / 4, color: roofCol });
      } else if (theme.shape === 'warehouse') {
        const base = new THREE.Color(theme.palette[Math.floor(hash01(r * 3 + 1, c * 7 + 2) * theme.palette.length)]).multiplyScalar(0.85 + 0.3 * shade);
        plainBoxes.push({ pos: V(cx, cy, h / 2), scale: new THREE.Vector3(ts * 0.94, h, ts * 0.94), color: base });
        flatRoofs.push({ pos: V(cx, cy, h + 0.08), scale: new THREE.Vector3(ts * 0.94, 0.16, ts * 0.94), color: base.clone().multiplyScalar(0.82) });
      } else if (theme.shape === 'farm') {
        // Most tiles get nothing at all -- the ground plane's field colour shows straight
        // through, which is the whole point (open countryside, not a walled-off lot).
        if (hash01(r * 5 + 2, c * 5 + 7) < theme.farmDensity) {
          const s = 0.8 + 0.5 * hash01(c, r);
          if (hash01(r + 11, c + 13) < 0.5) {
            plainBoxes.push({ pos: V(cx, cy, 2.6 * s), scale: new THREE.Vector3(4.2 * s, 5.2 * s, 3.4 * s), color: new THREE.Color(0xb43f34) });
            pitchRoofs.push({ pos: V(cx, cy, 5.2 * s + 1.4 * s), scale: new THREE.Vector3(3.7 * s, 2.6 * s, 3.7 * s), rotY: Math.PI / 4, color: new THREE.Color(0x3a3f46) });
          } else {
            silos.push({ pos: V(cx, cy, 3.0 * s), scale: new THREE.Vector3(1.7 * s, 6.0 * s, 1.7 * s), color: new THREE.Color(0xc9ccd0) });
            siloCaps.push({ pos: V(cx, cy, 6.0 * s + 0.9 * s), scale: new THREE.Vector3(1.9 * s, 1.8 * s, 1.9 * s), color: new THREE.Color(0x9aa0a6) });
          }
        }
      }
      if (theme.curbsLamps) for (const [dr, dc, nx, ny] of SIDES) if (!solid(r + dr, c + dc)) {
        const fx = cx + nx * ts / 2, fy = cy + ny * ts / 2;
        kerbs.push({ pos: V(fx + nx * 0.14, fy + ny * 0.14, 0.07), scale: new THREE.Vector3(nx ? 0.28 : ts, 0.14, nx ? ts : 0.28) });
        if ((nx ? r : c) % 6 === 3) posts.push({ x: fx + nx * 0.16, y: fy + ny * 0.16, nx, ny });
      }
    }
    g.add(instanced(box, facadeMaterial(), towerBoxes, true));
    g.add(instanced(box, std({}), plainBoxes, true));
    g.add(instanced(box, std({}), flatRoofs));
    g.add(instanced(new THREE.ConeGeometry(0.5, 1, 4), std({}), pitchRoofs));
    g.add(instanced(new THREE.CylinderGeometry(0.5, 0.5, 1, 16), std({ metalness: 0.3, roughness: 0.5 }), silos, true));
    g.add(instanced(new THREE.ConeGeometry(0.5, 1, 16), std({ metalness: 0.3, roughness: 0.5 }), siloCaps));
    g.add(instanced(flatGeom(ts, ts), std({ roughness: 0.95 }), roads));
    g.add(instanced(box, std({ color: C.kerb }), kerbs));
    g.add(instanced(new THREE.CylinderGeometry(0.05, 0.07, 5.2, 8), std({ color: C.pole, roughness: 0.5, metalness: 0.4 }), posts.map(p => ({ pos: V(p.x, p.y, 2.6) }))));
    g.add(instanced(new THREE.BoxGeometry(0.9, 0.12, 0.22), new THREE.MeshStandardMaterial({ color: 0xe8e2d0, emissive: C.lampGlow, emissiveIntensity: 0.35 }),
      posts.map(p => ({ pos: V(p.x + p.nx * 0.5, p.y + p.ny * 0.5, 5.15), rotY: -Math.atan2(p.ny, p.nx) }))));

    // Markings per corridor: amber centre dashes, white edge lines beside kerbs. Rows/cols
    // inside a crossing corridor are left open, as the physics has no lanes there either.
    const half = Math.floor(rw / 2), inH = new Uint8Array(H), inV = new Uint8Array(W);
    for (const hy of h_roads) for (let r = hy; r < hy + rw && r < H; r++) inH[r] = 1;
    for (const vx of v_roads) for (let c = vx; c < vx + rw && c < W; c++) inV[c] = 1;
    const dashes = [], edges = [];
    for (const vx of v_roads) {
      const x = (vx + rw / 2) * ts;
      for (let r = 0; r < H; r++) {
        if (inH[r] || solid(r, vx + half)) continue;
        const y = (r + 0.5) * ts;
        dashes.push({ pos: V(x, y, 0.012) });
        if (solid(r, vx - 1)) edges.push({ pos: V(vx * ts + 0.45, y, 0.012) });
        if (solid(r, vx + rw)) edges.push({ pos: V((vx + rw) * ts - 0.45, y, 0.012) });
      }
    }
    for (const hy of h_roads) {
      const y = (hy + rw / 2) * ts;
      for (let c = 0; c < W; c++) {
        if (inV[c] || solid(hy + half, c)) continue;
        const x = (c + 0.5) * ts;
        dashes.push({ pos: V(x, y, 0.012), rotY: Math.PI / 2 });
        if (solid(hy - 1, c)) edges.push({ pos: V(x, hy * ts + 0.45, 0.012), rotY: Math.PI / 2 });
        if (solid(hy + rw, c)) edges.push({ pos: V(x, (hy + rw) * ts - 0.45, 0.012), rotY: Math.PI / 2 });
      }
    }
    g.add(instanced(flatGeom(0.14, 2.4), new THREE.MeshBasicMaterial({ color: C.amber }), dashes));
    g.add(instanced(flatGeom(0.12, ts), new THREE.MeshBasicMaterial({ color: C.white }), edges));
  }

  // One overhead head per approach at every signalised crossing, mast on the near-right
  // corner, arm out over the corridor, lenses facing the approaching driver -- the layout
  // the physics assumes (stop line = edge of the crossing box; 'ns' heads govern y-travel).
  buildSignals(city) {
    const { tile_size: ts, road_width: rw, signals } = city;
    const half = rw / 2 * ts, g = this.worldGroup;
    const mastMat = new THREE.MeshStandardMaterial({ color: C.mast, roughness: 0.6, metalness: 0.3 });
    const plateMat = new THREE.MeshStandardMaterial({ color: C.plate, roughness: 0.8 });
    const stopMat = new THREE.MeshBasicMaterial({ color: C.white });
    const mastGeo = new THREE.CylinderGeometry(0.11, 0.13, 5.6, 10), lampGeo = new THREE.SphereGeometry(0.15, 12, 8);
    const DIRS = [[1, 0, 'ew'], [0, 1, 'ns'], [-1, 0, 'ew'], [0, -1, 'ns']];
    this.lamps = signals.map(() => ({ ns: [], ew: [] }));
    signals.forEach(([sx, sy], si) => {
      for (const [dx, dy, axis] of DIRS) {
        const rx = -dy, ry = dx;                                   // right of travel
        const lx = sx - dx * half, ly = sy - dy * half;            // stop line, corridor centre
        const stop = new THREE.Mesh(flatGeom(dx ? 0.4 : half, dx ? half : 0.4), stopMat);
        stop.position.set(lx + rx * half / 2, 0.014, ly + ry * half / 2); g.add(stop);
        const mx = lx + rx * half, my = ly + ry * half;
        const mast = new THREE.Mesh(mastGeo, mastMat); mast.position.set(mx, 2.8, my); mast.castShadow = true; g.add(mast);
        const arm = new THREE.Mesh(new THREE.BoxGeometry(dx ? 0.14 : half, 0.14, dx ? half : 0.14), mastMat);
        arm.position.set((mx + lx) / 2, 5.47, (my + ly) / 2); g.add(arm);
        const plate = new THREE.Mesh(new THREE.BoxGeometry(dx ? 0.08 : 0.5, 1.22, dx ? 0.5 : 0.08), plateMat);
        plate.position.set(lx, 4.54, ly); g.add(plate);
        const mats = [];
        [4.93, 4.54, 4.15].forEach((h, j) => {
          const mat = new THREE.MeshStandardMaterial({ color: C.lampOff[j], roughness: 0.4, toneMapped: false });
          const lamp = new THREE.Mesh(lampGeo, mat); lamp.position.set(lx - dx * 0.12, h, ly - dy * 0.12); g.add(lamp); mats.push(mat);
        });
        this.lamps[si][axis].push(mats);
      }
    });
  }

  buildMarkers(m) {
    const R = m.target_radius || 6;
    const ringGeo = new THREE.RingGeometry(R - 0.25, R + 0.25, 72); ringGeo.rotateX(-Math.PI / 2);
    const beamGeo = new THREE.CylinderGeometry(0.45, 0.45, 14, 16, 1, true);
    this.markerGeos = [ringGeo, beamGeo];
    for (let i = 0; i < 8; i++) {
      const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: C.target, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }));
      ring.position.y = 0.02;
      const beam = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({ color: C.target, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending }));
      beam.position.y = 7;
      const g = new THREE.Group(); g.add(ring, beam); g.visible = false; this.scene.add(g);
      this.markers.push({ g, ring, beam });
    }
  }

  proto(name, length, width) {
    if (!this.protos.has(name)) this.protos.set(name, new Promise(resolve =>
      this.loader.load(`/assets/cars/${name}.glb`, gltf => resolve(fitVehicle(gltf.scene, length, width)), undefined, () => resolve(fallbackVehicle(length, width)))));
    return this.protos.get(name);
  }

  buildVehicles(m) {
    const { kind, length, width } = m.vehicles;
    kind.forEach((name, i) => {
      const g = new THREE.Group(), ph = fallbackVehicle(length[i], width[i]);
      g.add(ph); this.dynGroup.add(g);
      const entry = { g, worldId: this.worldId }; this.vehicles.push(entry);
      this.proto(name, length[i], width[i]).then(p => { if (entry.worldId !== this.worldId) return; g.remove(ph); g.add(p.clone()); });
    });
  }

  // Zebra stripes, numbered sign posts and one walking figure per pedestrian, from
  // the reset message's `street` block. All of it is optional: with both flags off
  // the block is empty and nothing here is built.
  buildStreet(m) {
    if (this.pedGroup) { disposeGroup(this.pedGroup); this.scene.remove(this.pedGroup); }
    this.pedGroup = new THREE.Group(); this.scene.add(this.pedGroup); this.peds = [];
    const st = m.street; if (!st) return;
    const g = this.worldGroup, half = st.half;
    const white = new THREE.MeshBasicMaterial({ color: C.white });
    const stripes = [];
    for (const c of st.crossings) {
      // Bars run along the corridor (3 m) and repeat across it every 1.2 m.
      const rot = -Math.atan2(c.along[1], c.along[0]);
      for (let k = -half + 1.1; k <= half - 1.1; k += 1.2) stripes.push({ pos: V(c.x + c.perp[0] * k, c.y + c.perp[1] * k, 0.013), rotY: rot });
    }
    if (stripes.length) g.add(instanced(flatGeom(3.0, 0.55), white, stripes));
    if (st.signs.length) {
      const poleGeo = new THREE.CylinderGeometry(0.05, 0.05, 2.3, 8), discGeo = new THREE.CircleGeometry(0.42, 32);
      const poleMat = new THREE.MeshStandardMaterial({ color: 0x777c84 }), backMat = new THREE.MeshStandardMaterial({ color: 0x8a8f96 });
      for (const s of st.signs) {
        const pole = new THREE.Mesh(poleGeo, poleMat); pole.position.set(s.x, 1.15, s.y); g.add(pole);
        const disc = new THREE.Mesh(discGeo, new THREE.MeshBasicMaterial({ map: signTexture(s.limit_kmh) }));
        // The face looks back at the traffic it addresses, i.e. along -face.
        disc.position.set(s.x - s.face[0] * 0.04, 2.55, s.y - s.face[1] * 0.04);
        disc.rotation.y = Math.atan2(-s.face[0], -s.face[1]);
        g.add(disc);
        const back = new THREE.Mesh(discGeo, backMat); back.position.set(s.x + s.face[0] * 0.04, 2.55, s.y + s.face[1] * 0.04);
        back.rotation.y = Math.atan2(s.face[0], s.face[1]); g.add(back);
      }
    }
    for (let i = 0; i < st.n_pedestrians; i++) {
      const fig = makePerson(PEOPLE_COLORS[i % PEOPLE_COLORS.length]);
      this.pedGroup.add(fig); this.peds.push(fig);
    }
  }

  updatePedestrians(view) {
    const p = view.pedestrians; if (!p || !this.peds.length) return;
    for (let i = 0; i < this.peds.length && i < p.x.length; i++) {
      const fig = this.peds[i];
      fig.position.set(p.x[i], 0, p.y[i]);
      fig.rotation.y = -p.heading[i];
      const walking = p.state[i] === 1;
      fig.position.y = walking ? Math.abs(Math.sin(performance.now() / 140 + i)) * 0.05 : 0;   // a light step bob
    }
  }

  buildEgo(m) {
    if (this.egoBuilt) return;
    this.egoBuilt = true;
    const { length, width } = m.car;
    const ph = fallbackVehicle(length, width, C.ego); this.ego.add(ph);
    this.proto('race', length, width).then(p => { this.ego.remove(ph); this.ego.add(p.clone()); });
    const lampGeo = new THREE.BoxGeometry(0.08, 0.14, 0.34);
    for (const z of [-0.62, 0.62]) {
      const t = new THREE.Mesh(lampGeo, new THREE.MeshStandardMaterial({ color: 0xff3020, emissive: 0xff2010, emissiveIntensity: 0.8 }));
      t.position.set(-length / 2 - 0.02, 0.62, z); this.ego.add(t); this.tail.push(t);
    }
  }

  buildLidar(m) {
    if (this.lidar) { this.overlay.remove(this.lidar); this.lidar.geometry.dispose(); }
    const n = m.cfg.n_beams, geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
    this.lidar = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 }));
    this.lidar.frustumCulled = false; this.overlay.add(this.lidar);
    this.lidarOffsets = m.lidar_offsets; this.lidarRange = m.cfg.lidar_range;
  }

  // ------------------------------------------------------------------ per frame
  update(view, world, dt) {
    const e = view.ego, fx = Math.cos(e.heading), fy = Math.sin(e.heading);
    this.ego.position.set(e.x, 0, e.y); this.ego.rotation.y = -e.heading;
    const braking = (view.action[1] + 1) / 2 > 0.15 || view.action[0] < -0.05;
    for (const t of this.tail) t.material.emissiveIntensity = braking ? 3.2 : 0.8;

    const v = view.vehicles;
    for (let i = 0; i < this.vehicles.length && i < v.x.length; i++) { const g = this.vehicles[i].g; g.position.set(v.x[i], 0, v.y[i]); g.rotation.y = -v.heading[i]; }
    this.updatePedestrians(view);

    view.lights.forEach((l, i) => {
      const L = this.lamps[i]; if (!L) return;
      for (const axis of ['ns', 'ew']) {
        const on = l[axis] === 'red' ? 0 : l[axis] === 'yellow' ? 1 : 2;
        for (const head of L[axis]) head.forEach((mat, j) => { const lit = j === on; mat.color.setHex(lit ? C.lampOn[j] : C.lampOff[j]); mat.emissive.setHex(lit ? C.lampOn[j] : 0x000000); mat.emissiveIntensity = lit ? 1.8 : 0; });
      }
    });

    const pending = world.targets.slice(view.target_idx);
    this.markers.forEach((mk, i) => {
      const t = pending[i]; if (!t) { mk.g.visible = false; return; }
      const cur = i === 0, col = cur ? C.target : C.next;
      mk.g.visible = true; mk.g.position.set(t[0], 0, t[1]);
      mk.ring.material.color.setHex(col); mk.beam.material.color.setHex(col);
      mk.ring.material.opacity = cur ? 0.9 : 0.4; mk.beam.material.opacity = cur ? 0.25 : 0.08;
    });

    // Overlays: LIDAR fan coloured by range, the driver's chosen heading (amber) and the goal bearing (teal).
    this.overlay.visible = this.overlays;
    if (this.overlays && this.lidar) {
      const pos = this.lidar.geometry.attributes.position, col = this.lidar.geometry.attributes.color, c = new THREE.Color();
      for (let k = 0; k < view.lidar.length; k++) {
        const a = e.heading + this.lidarOffsets[k], d = view.lidar[k];
        c.copy(C.near).lerp(C.far, clamp(d / this.lidarRange, 0, 1));
        pos.setXYZ(2 * k, e.x, 0.5, e.y); pos.setXYZ(2 * k + 1, e.x + Math.cos(a) * d, 0.5, e.y + Math.sin(a) * d);
        col.setXYZ(2 * k, c.r, c.g, c.b); col.setXYZ(2 * k + 1, c.r, c.g, c.b);
      }
      pos.needsUpdate = col.needsUpdate = true;
      const d = view.driver;
      if (d && !view.manual && typeof d.theta_deg === 'number') {
        const th = e.heading + d.theta_deg * Math.PI / 180, len = clamp(d.chosen_clear, 3, 28);
        this.intentLine.geometry.setFromPoints([V(e.x, e.y, 0.3), V(e.x + Math.cos(th) * len, e.y + Math.sin(th) * len, 0.3)]);
        const bh = e.heading + d.bearing_deg * Math.PI / 180;
        this.goalLine.geometry.setFromPoints([V(e.x, e.y, 0.25), V(e.x + Math.cos(bh) * 12, e.y + Math.sin(bh) * 12, 0.25)]);
        this.intentLine.visible = this.goalLine.visible = true;
      } else this.intentLine.visible = this.goalLine.visible = false;
    }
    if (this.fovWedgeGroup) this.fovWedgeGroup.visible = this.overlays;
    this.updateDetections(view, world);

    // Sun follows the car so the shadow frustum stays tight; sky dome follows the camera.
    this.sun.position.set(e.x + 40, 70, e.y + 30); this.sun.target.position.set(e.x + fx * 15, 0, e.y + fy * 15); this.sun.target.updateMatrixWorld();

    let want, look;
    if (this.view === 'aerial') { want = V(e.x - fx * 20, e.y - fy * 20, 46); look = V(e.x + fx * 14, e.y + fy * 14, 0); }
    // Bonnet cam: just past the nose (the race car's body reaches 2.2 m ahead of centre).
    else if (this.view === 'driver') { want = V(e.x + fx * 2.35, e.y + fy * 2.35, 1.15); look = V(e.x + fx * 30, e.y + fy * 30, 0.9); }
    // 6.6 m clears the signal arms (5.6 m) the camera otherwise flies through at crossings.
    else { want = V(e.x - fx * 13, e.y - fy * 13, 6.6); look = V(e.x + fx * 9, e.y + fy * 9, 0.8); }
    const k = (this.first || this.view === 'driver') ? 1 : 1 - Math.exp(-dt * 4); this.first = false;
    this.camPos.lerp(want, k); this.camTarget.lerp(look, k);
    this.camera.position.copy(this.camPos); this.camera.lookAt(this.camTarget);
    this.sky.position.copy(this.camera.position);
  }

  // Numbers a headless DOM dump can show when the picture is wrong (?debug=1).
  diag() {
    const size = o => { const b = new THREE.Box3().setFromObject(o); if (b.isEmpty()) return 'empty'; const s = b.getSize(new THREE.Vector3()); return [s.x, s.y, s.z].map(v => +v.toFixed(2)); };
    const r3 = v => v.toArray().map(x => +x.toFixed(1));
    return {
      view: this.view, cam: r3(this.camera.position), look: r3(this.camTarget), egoPos: r3(this.ego.position),
      egoSize: size(this.ego), egoChildren: this.ego.children.map(c => c.type),
      vehicles: this.vehicles.length, veh0Size: this.vehicles[0] ? size(this.vehicles[0].g) : null, veh0Pos: this.vehicles[0] ? r3(this.vehicles[0].g.position) : null,
      protos: [...this.protos.keys()], drawCalls: this.renderer.info.render.calls,
    };
  }

  render(feedRects) {
    const r = this.renderer, W = window.innerWidth, H = window.innerHeight;
    r.setScissorTest(false); r.setViewport(0, 0, W, H); r.clear();
    r.render(this.scene, this.camera);
    r.setScissorTest(true);
    const overlaysWere = this.overlay.visible; this.overlay.visible = false;
    for (const [name, rect] of Object.entries(feedRects)) {
      const cam = this.feedCams[name]; if (!cam || rect.width < 10) continue;
      const x = rect.left, y = H - rect.bottom;
      r.setViewport(x, y, rect.width, rect.height); r.setScissor(x, y, rect.width, rect.height);
      cam.aspect = rect.width / rect.height;
      cam.fov = 2 * Math.atan(Math.tan(cam.userData.hfov * Math.PI / 360) / cam.aspect) * 180 / Math.PI; cam.updateProjectionMatrix();
      r.clear(); r.render(this.scene, cam);
    }
    this.overlay.visible = overlaysWere;
    r.setScissorTest(false);
  }
}
