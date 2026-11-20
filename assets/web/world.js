// The 3D world. app.js sends it the game state after every step and calls frame() every frame.
// Grid cell (x, y) sits at world (x, 0, y) times CELL, centred on the origin. The river (row 0) is north (-z).

import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Water } from 'three/addons/objects/Water.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

const CELL = 2;
const FLOOD = 2, CROP = 3, HOUSE = 4;
const WATER_ROWS = 2;

// extra turn (radians) for model files in assets/meshes that do not face +z
const MESH_TURN = { guardian: 0, crocodile: 0, hippo: 0 };
// colour for model files that come without their own colours
const MESH_COLOUR = { crocodile: 0x46502f, hippo: 0x6d5d66 };

// small helpers

function seeded(seed) {
  // tiny repeatable random generator, so trees and grass are always in the same place
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(x, z) {
  const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

function noise(x, z) {
  // smooth value noise between 0 and 1
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
  const a = hash(ix, iz), b = hash(ix + 1, iz), c = hash(ix, iz + 1), d = hash(ix + 1, iz + 1);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}

function fbm(x, z) {
  return noise(x, z) * 0.5 + noise(x * 2.1, z * 2.1) * 0.3 + noise(x * 4.3, z * 4.3) * 0.2;
}

function mesh(geometry, material, scale, position, shadow = true) {
  const m = new THREE.Mesh(geometry, material);
  if (scale) m.scale.set(...scale);
  if (position) m.position.set(...position);
  m.castShadow = shadow;
  m.receiveShadow = true;
  return m;
}

function walkToward(position, target, speed, dt) {
  // a steady walking pace, but hurry up when far behind (at high simulation speeds)
  const d = target.clone().sub(position);
  const dist = d.length();
  if (dist < 0.0001) return;
  position.addScaledVector(d, Math.min(dist, Math.max(speed, dist * 3) * dt) / dist);
}

function turnToward(object, angle, amount) {
  let d = angle - object.rotation.y;
  d = Math.atan2(Math.sin(d), Math.cos(d)); // shortest way round
  object.rotation.y += d * amount;
}

function canvasTexture(width, height, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d'));
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

const SPHERE = new THREE.SphereGeometry(1, 14, 10);
const X_AXIS = new THREE.Vector3(1, 0, 0);
const CYLINDER = new THREE.CylinderGeometry(1, 1, 1, 8);

const GRASS_DRY = new THREE.Color(0xa89a58);
const GRASS_GREEN = new THREE.Color(0x6f8a3a);
const MUD = new THREE.Color(0x7a6040);
const RIVERBED = new THREE.Color(0x3d3224);
const LATERITE = new THREE.Color(0xa45f3c);
const SOIL = new THREE.Color(0x4f3522);
const TRAMPLED = new THREE.Color(0x77613f);

function grassColour(x, z, out) {
  const n = fbm(x * 0.08, z * 0.08);
  out.copy(GRASS_GREEN).lerp(GRASS_DRY, n * 0.9);
  return out.multiplyScalar(0.92 + hash(x * 3.1, z * 2.7) * 0.12);
}

export class World {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.6;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0xc9d3dc, 110, 380);
    this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 3000);
    this.camera.position.set(0, 14, 26);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enabled = false;
    this.controls.maxPolarAngle = Math.PI * 0.48;

    this.cameraMode = 0;
    this.camLook = new THREE.Vector3();
    this.shake = 0;
    this.time = 0;
    this.animals = new Map();
    this.effects = [];
    this.episode = null;
    this.models = {};

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  cellPos(x, y) {
    const mid = (this.size - 1) / 2;
    return new THREE.Vector3((x - mid) * CELL, 0, (y - mid) * CELL);
  }

  async init(state) {
    this.size = state.size;
    this.extent = (this.size / 2) * CELL;                   // half width of the play area
    this.riverNear = (WATER_ROWS - this.size / 2) * CELL;    // the bank the village is on
    this.riverFar = this.riverNear - 14;                     // the opposite bank

    this.buildSkyAndLight();
    this.buildTerrain();
    this.buildPlayArea(state.grid);
    this.buildRiver();
    this.buildVegetation();
    this.buildVillage(state.grid);
    this.buildCrops(state.grid);
    await this.loadMeshes(state.meshes || []);
    this.guardian = this.makeGuardian();
    this.scene.add(this.guardian.root);
    this.update(state);
  }

  // sky, sun and fog

  buildSkyAndLight() {
    const elevation = THREE.MathUtils.degToRad(24);
    const azimuth = THREE.MathUtils.degToRad(215);
    this.sunDir = new THREE.Vector3().setFromSphericalCoords(1, Math.PI / 2 - elevation, azimuth);

    const makeSky = () => {
      const sky = new Sky();
      sky.scale.setScalar(10000);
      const u = sky.material.uniforms;
      u.turbidity.value = 7;
      u.rayleigh.value = 1.6;
      u.mieCoefficient.value = 0.005;
      u.mieDirectionalG.value = 0.8;
      u.sunPosition.value.copy(this.sunDir);
      return sky;
    };
    this.scene.add(makeSky());

    // light the scene with the same sky, so materials pick up its colour
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const envScene = new THREE.Scene();
    envScene.add(makeSky());
    this.scene.environment = pmrem.fromScene(envScene).texture;
    this.scene.environmentIntensity = 0.55;

    const sun = new THREE.DirectionalLight(0xfff0d8, 3.2);
    sun.position.copy(this.sunDir).multiplyScalar(80);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const s = sun.shadow.camera;
    s.left = -34; s.right = 34; s.top = 34; s.bottom = -34; s.near = 1; s.far = 220;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.03;
    this.scene.add(sun, sun.target);
    this.scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x8a6b45, 0.9));
  }

  // ground

  terrainHeight(x, z) {
    let h = 0;
    if (z < this.riverNear && z > this.riverFar) {
      const fromBank = Math.min(this.riverNear - z, z - this.riverFar);
      h = -Math.min(1.8, 0.35 + fromBank * 0.6);
    }
    // gentle hills that grow away from the village
    const dx = Math.max(0, Math.abs(x) - this.extent - 8);
    const dz = z > 0 ? Math.max(0, z - this.extent - 8) : Math.max(0, this.riverFar - 4 - z);
    const away = Math.min(1, Math.hypot(dx, dz) / 40);
    // no hills in the river valley, so the river keeps flowing past the edges of the map
    const inRiver = z < this.riverNear && z > this.riverFar;
    const fromRiver = inRiver ? 0 : Math.min(Math.abs(z - this.riverNear), Math.abs(z - this.riverFar));
    h += away * away * (3 + fbm(x * 0.025, z * 0.025) * 16) * Math.min(1, fromRiver / 12);
    return h;
  }

  buildTerrain() {
    const geometry = new THREE.PlaneGeometry(700, 700, 220, 220);
    geometry.rotateX(-Math.PI / 2);
    const pos = geometry.attributes.position;
    const colours = new Float32Array(pos.count * 3);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      const h = this.terrainHeight(x, z);
      pos.setY(i, h);
      if (h < -0.3) c.copy(RIVERBED);
      else if (z < this.riverNear + 0.8 && z > this.riverFar - 2.5) c.copy(MUD).lerp(GRASS_DRY, hash(x, z) * 0.3);
      else grassColour(x, z, c);
      colours.set([c.r, c.g, c.b], i * 3);
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3));
    geometry.computeVertexNormals();
    const ground = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95 }));
    ground.receiveShadow = true;
    this.scene.add(ground);
  }

  buildPlayArea(grid) {
    // a finer ground patch over the grid (flood edge down to the bottom), coloured by cell type
    const rows = this.size - WATER_ROWS;
    const geometry = new THREE.PlaneGeometry(this.size * CELL, rows * CELL, this.size * 8, rows * 8);
    geometry.rotateX(-Math.PI / 2);
    geometry.translate(0, 0.02, this.riverNear + (rows * CELL) / 2);
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(geometry.attributes.position.count * 3), 3));
    this.playArea = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95 }));
    this.playArea.receiveShadow = true;
    this.scene.add(this.playArea);
    this.eatenCells = new Set();
    this.colourPlayArea(grid);

    // faint cell lines, toggled with G
    const points = [];
    const top = this.riverNear, bottom = this.extent, left = -this.extent, right = this.extent;
    for (let i = 0; i <= this.size; i++) {
      const x = left + i * CELL;
      points.push(x, 0.04, top, x, 0.04, bottom);
    }
    for (let j = 0; j <= rows; j++) {
      const z = top + j * CELL;
      points.push(left, 0.04, z, right, 0.04, z);
    }
    const lines = new THREE.BufferGeometry();
    lines.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    this.gridLines = new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 }));
    this.gridLines.visible = false;
    this.scene.add(this.gridLines);
  }

  colourPlayArea(grid) {
    const pos = this.playArea.geometry.attributes.position;
    const colours = this.playArea.geometry.attributes.color;
    const mid = (this.size - 1) / 2;
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      // which cell this vertex belongs to (vertices on an edge go to the lower cell)
      const gx = Math.min(this.size - 1, Math.max(0, Math.round(x / CELL + mid - 0.001)));
      const gy = Math.min(this.size - 1, Math.max(WATER_ROWS, Math.round(z / CELL + mid - 0.001)));
      const cell = grid[gy][gx];
      const grain = hash(x * 5.3, z * 4.1) * 0.12;
      if (cell === FLOOD) c.copy(MUD).lerp(GRASS_DRY, fbm(x * 0.6, z * 0.6) * 0.35);
      else if (cell === HOUSE) c.copy(LATERITE).lerp(GRASS_DRY, fbm(x * 0.7, z * 0.7) * 0.25);
      else if (cell === CROP) c.copy(SOIL);
      else if (this.eatenCells.has(`${gx},${gy}`)) c.copy(TRAMPLED);
      else grassColour(x, z, c);
      c.multiplyScalar(0.94 + grain);
      colours.setXYZ(i, c.r, c.g, c.b);
    }
    colours.needsUpdate = true;
  }

  // river

  buildRiver() {
    const width = this.riverNear - this.riverFar + 6;
    const geometry = new THREE.PlaneGeometry(900, width);
    const normals = new THREE.TextureLoader().load('textures/waternormals.jpg', (t) => {
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
    });
    this.water = new Water(geometry, {
      textureWidth: 512,
      textureHeight: 512,
      waterNormals: normals,
      sunDirection: this.sunDir.clone(),
      sunColor: 0xfff2dd,
      waterColor: 0x1d3a35,
      distortionScale: 2.2,
      fog: true,
    });
    this.water.rotation.x = -Math.PI / 2;
    this.water.position.set(0, -0.28, (this.riverNear + this.riverFar) / 2);
    this.water.material.uniforms.size.value = 3;
    this.scene.add(this.water);
  }

  // plants and trees

  outsidePlayArea(x, z, margin) {
    const inRiver = z < this.riverNear + 1 && z > this.riverFar - 1;
    const inGrid = Math.abs(x) < this.extent + margin && z > this.riverNear - 1 && z < this.extent + margin;
    return !inRiver && !inGrid;
  }

  buildVegetation() {
    const rand = seeded(7);
    const dummy = new THREE.Object3D();

    // reeds along both banks, in small clumps
    const reedGeometry = new THREE.ConeGeometry(0.035, 1.3, 4);
    reedGeometry.translate(0, 0.65, 0);
    const reeds = new THREE.InstancedMesh(reedGeometry, new THREE.MeshStandardMaterial({ color: 0x7d8a3a, roughness: 0.9 }), 1400);
    let n = 0;
    while (n < reeds.count) {
      const nearSide = rand() < 0.55;
      const cx = (rand() - 0.5) * 260;
      const cz = nearSide ? this.riverNear - 0.4 - rand() * 1.4 : this.riverFar + 0.3 + rand() * 1.2;
      for (let k = 0; k < 10 && n < reeds.count; k++, n++) {
        const x = cx + (rand() - 0.5) * 1.6, z = cz + (rand() - 0.5) * 0.8;
        dummy.position.set(x, this.terrainHeight(x, z) * 0.4 - 0.15, z);
        dummy.rotation.set((rand() - 0.5) * 0.4, rand() * 6, (rand() - 0.5) * 0.4);
        dummy.scale.setScalar(0.6 + rand() * 0.8);
        dummy.updateMatrix();
        reeds.setMatrixAt(n, dummy.matrix);
      }
    }
    reeds.castShadow = true;
    this.scene.add(reeds);

    // grass tufts outside the play area
    const tuftGeometry = new THREE.ConeGeometry(0.07, 0.55, 3);
    tuftGeometry.translate(0, 0.27, 0);
    const tufts = new THREE.InstancedMesh(tuftGeometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 }), 5000);
    const c = new THREE.Color();
    for (let i = 0; i < tufts.count; i++) {
      let x, z;
      do { x = (rand() - 0.5) * 160; z = (rand() - 0.5) * 160; } while (!this.outsidePlayArea(x, z, 0.5));
      dummy.position.set(x, this.terrainHeight(x, z), z);
      dummy.rotation.set((rand() - 0.5) * 0.5, rand() * 6, (rand() - 0.5) * 0.5);
      dummy.scale.set(1, 0.6 + rand(), 1);
      dummy.updateMatrix();
      tufts.setMatrixAt(i, dummy.matrix);
      tufts.setColorAt(i, grassColour(x, z, c).multiplyScalar(0.85));
    }
    this.scene.add(tufts);

    // acacia trees, the flat topped trees of the Eastern Province savanna
    for (let i = 0; i < 55; i++) {
      let x, z;
      do { x = (rand() - 0.5) * 170; z = (rand() - 0.5) * 150 - 10; } while (!this.outsidePlayArea(x, z, 9));
      const tree = this.makeAcacia(rand);
      tree.position.set(x, this.terrainHeight(x, z) - 0.05, z);
      tree.rotation.y = rand() * 6.28;
      tree.scale.setScalar(0.8 + rand() * 0.6);
      this.scene.add(tree);
    }

    // a few rocks on the banks
    const rockGeometry = new THREE.DodecahedronGeometry(0.5, 0);
    const rockMaterial = new THREE.MeshStandardMaterial({ color: 0x7a7066, roughness: 0.9, flatShading: true });
    for (let i = 0; i < 40; i++) {
      const x = (rand() - 0.5) * 120;
      const z = rand() < 0.5 ? this.riverNear - 0.6 - rand() : this.riverFar + rand() * 1.5;
      if (Math.abs(x) < this.extent && z > this.riverNear - 1) continue;
      const rock = mesh(rockGeometry, rockMaterial, [0.5 + rand(), 0.35 + rand() * 0.4, 0.5 + rand()],
        [x, this.terrainHeight(x, z) * 0.5 - 0.1, z]);
      rock.rotation.y = rand() * 6;
      this.scene.add(rock);
    }
  }

  makeAcacia(rand) {
    if (!this.acacia) {
      this.acacia = {
        bark: new THREE.MeshStandardMaterial({ color: 0x4a3828, roughness: 0.95 }),
        leaves: new THREE.MeshStandardMaterial({ color: 0x55702c, roughness: 0.9, flatShading: true }),
        trunk: new THREE.CylinderGeometry(0.1, 0.2, 1, 6),
        canopy: new THREE.IcosahedronGeometry(1, 1),
      };
    }
    const a = this.acacia;
    const tree = new THREE.Group();
    const trunk = mesh(a.trunk, a.bark, [1, 2.6, 1], [0, 1.3, 0]);
    tree.add(trunk);
    // two or three branches spreading out, each with a flat crown on top
    const branches = 2 + Math.floor(rand() * 2);
    for (let b = 0; b < branches; b++) {
      const angle = (b / branches) * Math.PI * 2 + rand();
      const lean = 0.5 + rand() * 0.3;
      const branch = mesh(a.trunk, a.bark, [0.6, 1.8, 0.6]);
      branch.position.set(Math.sin(angle) * 0.4, 3.2, Math.cos(angle) * 0.4);
      branch.rotation.set(Math.cos(angle) * lean, 0, -Math.sin(angle) * lean);
      tree.add(branch);
      const crown = mesh(a.canopy, a.leaves, [1.6 + rand() * 0.6, 0.35, 1.6 + rand() * 0.6],
        [Math.sin(angle) * 1.2, 4.1 + rand() * 0.3, Math.cos(angle) * 1.2]);
      tree.add(crown);
    }
    tree.add(mesh(a.canopy, a.leaves, [2.4, 0.4, 2.4], [0, 4.35, 0]));
    return tree;
  }

  makeBanana(rand) {
    const group = new THREE.Group();
    const stem = new THREE.MeshStandardMaterial({ color: 0x6b7a3a, roughness: 0.8 });
    const leafMaterial = new THREE.MeshStandardMaterial({ color: 0x4f8a2b, roughness: 0.7, side: THREE.DoubleSide });
    group.add(mesh(CYLINDER, stem, [0.12, 1.6, 0.12], [0, 0.8, 0]));
    for (let i = 0; i < 7; i++) {
      const leaf = new THREE.PlaneGeometry(0.45, 1.8, 1, 6);
      const p = leaf.attributes.position;
      for (let v = 0; v < p.count; v++) {
        const t = (p.getY(v) + 0.9) / 1.8;
        p.setZ(v, -t * t * 0.7); // droop toward the tip
      }
      leaf.translate(0, 0.9, 0);
      leaf.computeVertexNormals();
      const m = mesh(leaf, leafMaterial);
      m.position.y = 1.5;
      m.rotation.set(-0.7 - rand() * 0.4, (i / 7) * Math.PI * 2 + rand() * 0.3, 0, 'YXZ');
      group.add(m);
    }
    return group;
  }

  // village

  buildVillage(grid) {
    const walls = canvasTexture(128, 128, (g) => {
      g.fillStyle = '#b06a42';
      g.fillRect(0, 0, 128, 128);
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 4; col++) {
          const shade = 150 + Math.floor(hash(row, col) * 40);
          g.fillStyle = `rgb(${shade + 20}, ${shade - 50}, ${shade - 90})`;
          g.fillRect(col * 32 + (row % 2) * 16 + 1, row * 16 + 1, 30, 14);
        }
      }
    });
    walls.repeat.set(1.5, 1);
    const roof = canvasTexture(64, 8, (g) => {
      const grad = g.createLinearGradient(0, 0, 8, 0);
      grad.addColorStop(0, '#8e9499');
      grad.addColorStop(0.5, '#d5d9dc');
      grad.addColorStop(1, '#8e9499');
      g.fillStyle = grad;
      for (let i = 0; i < 8; i++) g.fillRect(i * 8, 0, 8, 8);
      g.fillStyle = 'rgba(140, 80, 40, 0.25)'; // a bit of rust
      g.fillRect(10, 0, 9, 8);
      g.fillRect(41, 0, 6, 8);
    });
    roof.repeat.set(3, 1);

    const materials = {
      wall: new THREE.MeshStandardMaterial({ map: walls, roughness: 0.9 }),
      roof: new THREE.MeshStandardMaterial({ map: roof, metalness: 0.25, roughness: 0.55, side: THREE.DoubleSide }),
      base: new THREE.MeshStandardMaterial({ color: 0x5b5048, roughness: 0.9 }),
      door: new THREE.MeshStandardMaterial({ color: 0x4a2f1c, roughness: 0.7 }),
      glass: new THREE.MeshStandardMaterial({ color: 0x1b2430, roughness: 0.2, metalness: 0.3 }),
      frame: new THREE.MeshStandardMaterial({ color: 0xd8c8a6, roughness: 0.8 }),
      can: new THREE.MeshStandardMaterial({ color: 0xf2c418, roughness: 0.5 }),
    };

    const rand = seeded(11);
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        if (grid[y][x] !== HOUSE) continue;
        const house = this.makeHouse(materials, rand);
        house.position.copy(this.cellPos(x, y));
        house.rotation.y = (rand() - 0.5) * 0.12;
        this.scene.add(house);
      }
    }

    // banana plants behind the village and along its sides
    for (let i = 0; i < 14; i++) {
      const x = (rand() - 0.5) * (this.extent * 2 + 10);
      const z = this.extent + 3 + rand() * 9;
      const banana = this.makeBanana(rand);
      banana.position.set(x, this.terrainHeight(x, z), z);
      banana.scale.setScalar(0.9 + rand() * 0.5);
      this.scene.add(banana);
    }

    // wooden posts marking the edge of the guarded land
    const post = new THREE.MeshStandardMaterial({ color: 0x5c4630, roughness: 0.9 });
    const addPost = (x, z) => this.scene.add(mesh(CYLINDER, post, [0.07, 0.9, 0.07], [x, 0.45, z]));
    for (let z = this.riverNear + CELL; z <= this.extent; z += CELL) {
      addPost(-this.extent - 0.3, z);
      addPost(this.extent + 0.3, z);
    }
    for (let x = -this.extent; x <= this.extent; x += CELL) addPost(x, this.extent + 0.3);
  }

  makeHouse(m, rand) {
    // a mud brick house with a corrugated iron roof, the door faces the camera (+z)
    const house = new THREE.Group();
    const w = 1.55, d = 1.35, h = 1.2, rise = 0.55, overhang = 0.18;
    house.add(mesh(new THREE.BoxGeometry(w + 0.12, 0.12, d + 0.12), m.base, null, [0, 0.06, 0]));
    house.add(mesh(new THREE.BoxGeometry(w, h, d), m.wall, null, [0, 0.12 + h / 2, 0]));

    const top = 0.12 + h;
    const half = d / 2 + overhang;
    const slope = Math.hypot(half, rise);
    const angle = Math.atan2(rise, half);
    for (const side of [1, -1]) {
      const panel = mesh(new THREE.BoxGeometry(w + overhang * 2, 0.03, slope), m.roof);
      panel.position.set(0, top + rise / 2, (side * half) / 2);
      panel.rotation.x = side * angle;
      house.add(panel);
    }
    // gable ends
    const shape = new THREE.Shape([new THREE.Vector2(-d / 2, 0), new THREE.Vector2(d / 2, 0), new THREE.Vector2(0, rise - 0.02)]);
    for (const side of [1, -1]) {
      const gable = mesh(new THREE.ShapeGeometry(shape), m.wall);
      gable.rotation.y = Math.PI / 2;
      gable.position.set((side * w) / 2, top, 0);
      house.add(gable);
    }
    house.add(mesh(new THREE.BoxGeometry(0.38, 0.78, 0.04), m.door, null, [0, 0.12 + 0.39, d / 2 + 0.01]));
    for (const side of [-1, 1]) {
      house.add(mesh(new THREE.BoxGeometry(0.36, 0.36, 0.03), m.frame, null, [side * 0.5, 0.85, d / 2 + 0.005]));
      house.add(mesh(new THREE.BoxGeometry(0.28, 0.28, 0.04), m.glass, null, [side * 0.5, 0.85, d / 2 + 0.01]));
    }
    if (rand() < 0.7) {
      house.add(mesh(new THREE.BoxGeometry(0.16, 0.24, 0.1), m.can, null, [0.32, 0.12, d / 2 + 0.2]));
    }
    house.scale.setScalar(0.95 + rand() * 0.1);
    return house;
  }

  // crops

  buildCrops(grid) {
    // one maize plant shape, drawn many times with instancing
    const parts = [];
    const stalk = new THREE.CylinderGeometry(0.022, 0.035, 1.7, 5);
    stalk.translate(0, 0.85, 0);
    parts.push(stalk);
    for (let k = 0; k < 6; k++) {
      const leaf = new THREE.PlaneGeometry(0.09, 0.8, 1, 4);
      const p = leaf.attributes.position;
      for (let v = 0; v < p.count; v++) {
        const t = (p.getY(v) + 0.4) / 0.8;
        p.setZ(v, t * t * 0.35);
      }
      leaf.translate(0, 0.4, 0);
      leaf.rotateX(0.7);
      leaf.rotateY((k / 6) * Math.PI * 2 + k);
      leaf.translate(0, 0.35 + k * 0.2, 0);
      parts.push(leaf);
    }
    const plant = mergeGeometries(parts.map((g) => g.toNonIndexed()));
    plant.computeVertexNormals();
    const tassel = new THREE.ConeGeometry(0.05, 0.28, 4);
    tassel.translate(0, 1.82, 0);

    this.cropCells = [];
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) if (grid[y][x] === CROP) this.cropCells.push([x, y]);
    }
    const perCell = 9;
    const count = this.cropCells.length * perCell;
    this.maize = new THREE.InstancedMesh(plant,
      new THREE.MeshStandardMaterial({ color: 0x6f9a32, roughness: 0.75, side: THREE.DoubleSide }), count);
    this.tassels = new THREE.InstancedMesh(tassel, new THREE.MeshStandardMaterial({ color: 0xd8c27a, roughness: 0.8 }), count);
    this.maize.castShadow = true;
    this.maize.receiveShadow = true;

    // soil ridges under the plants
    const ridge = new THREE.BoxGeometry(CELL * 0.92, 0.1, 0.22);
    const ridges = new THREE.InstancedMesh(ridge, new THREE.MeshStandardMaterial({ color: 0x3f2a1b, roughness: 1 }), this.cropCells.length * 3);

    const rand = seeded(5);
    const dummy = new THREE.Object3D();
    this.cropPlants = new Map();
    let i = 0;
    this.cropCells.forEach(([x, y], cellIndex) => {
      const centre = this.cellPos(x, y);
      const plants = [];
      for (let r = 0; r < 3; r++) {
        dummy.position.set(centre.x, 0.05, centre.z + (r - 1) * 0.6);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        ridges.setMatrixAt(cellIndex * 3 + r, dummy.matrix);
        for (let col = 0; col < 3; col++, i++) {
          dummy.position.set(centre.x + (col - 1) * 0.6 + (rand() - 0.5) * 0.15, 0.05,
            centre.z + (r - 1) * 0.6 + (rand() - 0.5) * 0.15);
          dummy.rotation.set((rand() - 0.5) * 0.12, rand() * 6.28, (rand() - 0.5) * 0.12);
          dummy.scale.setScalar(0.85 + rand() * 0.3);
          dummy.updateMatrix();
          plants.push({ index: i, matrix: dummy.matrix.clone() });
        }
      }
      this.cropPlants.set(`${x},${y}`, { plants, eatenAt: null });
    });
    ridges.receiveShadow = true;
    this.scene.add(this.maize, this.tassels, ridges);
    this.resetCrops();
  }

  resetCrops() {
    for (const cell of this.cropPlants.values()) {
      cell.eatenAt = null;
      for (const p of cell.plants) {
        this.maize.setMatrixAt(p.index, p.matrix);
        this.tassels.setMatrixAt(p.index, p.matrix);
      }
    }
    this.maize.instanceMatrix.needsUpdate = true;
    this.tassels.instanceMatrix.needsUpdate = true;
  }

  animateCrops() {
    // eaten plants tip over and shrink into the ground
    const m = new THREE.Matrix4(), squash = new THREE.Matrix4();
    let changed = false;
    for (const cell of this.cropPlants.values()) {
      if (cell.eatenAt === null) continue;
      const t = Math.min(1, (this.time - cell.eatenAt) / 0.8);
      if (cell.done) continue;
      for (const p of cell.plants) {
        squash.makeScale(1, Math.max(0.001, 1 - t), 1);
        m.copy(p.matrix).multiply(squash);
        this.maize.setMatrixAt(p.index, m);
        this.tassels.setMatrixAt(p.index, m);
      }
      cell.done = t >= 1;
      changed = true;
    }
    if (changed) {
      this.maize.instanceMatrix.needsUpdate = true;
      this.tassels.instanceMatrix.needsUpdate = true;
    }
  }

  // characters

  async loadMeshes(names) {
    const loader = new GLTFLoader();
    for (const name of names) {
      try {
        const gltf = await loader.loadAsync(`meshes/${name}.glb`);
        this.models[name] = gltf;
      } catch (error) {
        console.warn(`Could not load meshes/${name}.glb, using the built in shape`, error);
      }
    }
  }

  fromMesh(name, size, byHeight) {
    // copy a loaded model, scale it to the wanted size and stand it on the ground
    const gltf = this.models[name];
    const model = SkeletonUtils.clone(gltf.scene);
    const box = new THREE.Box3().setFromObject(model);
    const dims = box.getSize(new THREE.Vector3());
    const scale = size / (byHeight ? dims.y : Math.max(dims.x, dims.z));
    model.scale.setScalar(scale);
    const centre = box.getCenter(new THREE.Vector3());
    model.position.set(-centre.x * scale, -box.min.y * scale, -centre.z * scale);
    const legBones = [];
    model.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
        o.frustumCulled = false;
        if (MESH_COLOUR[name]) {
          o.material = new THREE.MeshStandardMaterial({ color: MESH_COLOUR[name], roughness: 0.7, flatShading: true });
        }
      }
      // upper leg bones (like FrontLeg_L) are swung by hand while walking, if the model has them
      if (o.isBone && /Leg_[LR]$/.test(o.name)) legBones.push({ bone: o, rest: o.quaternion.clone() });
    });
    const holder = new THREE.Group();
    holder.add(model);
    holder.rotation.y = MESH_TURN[name] || 0;
    let mixer = null, action = null;
    if (gltf.animations.length) {
      mixer = new THREE.AnimationMixer(model);
      action = mixer.clipAction(gltf.animations[0]);
      action.play();
    }
    return { holder, mixer, action, legBones };
  }

  makeGuardian() {
    const root = new THREE.Group();
    const g = { root, target: new THREE.Vector3(), phase: 0, hop: 0, legs: [], arms: [] };
    if (this.models.guardian) {
      const m = this.fromMesh('guardian', 1.75, true);
      root.add(m.holder);
      g.mixer = m.mixer;
      g.action = m.action;
      return g;
    }
    // built in ranger: khaki shirt, dark trousers, wide hat and a long stick
    const shirt = new THREE.MeshStandardMaterial({ color: 0x8b7d4b, roughness: 0.8 });
    const trousers = new THREE.MeshStandardMaterial({ color: 0x2f3a2a, roughness: 0.8 });
    const skin = new THREE.MeshStandardMaterial({ color: 0x4a2e1f, roughness: 0.6 });
    const hat = new THREE.MeshStandardMaterial({ color: 0x6e5d34, roughness: 0.9 });
    const boots = new THREE.MeshStandardMaterial({ color: 0x241811, roughness: 0.7 });
    for (const side of [-1, 1]) {
      const leg = new THREE.Group();
      leg.position.set(side * 0.11, 0.92, 0);
      leg.add(mesh(CYLINDER, trousers, [0.085, 0.8, 0.085], [0, -0.42, 0]));
      leg.add(mesh(new THREE.BoxGeometry(0.14, 0.1, 0.26), boots, null, [0, -0.85, 0.04]));
      root.add(leg);
      g.legs.push(leg);
      const arm = new THREE.Group();
      arm.position.set(side * 0.25, 1.45, 0);
      arm.add(mesh(CYLINDER, shirt, [0.065, 0.55, 0.065], [0, -0.27, 0]));
      arm.add(mesh(SPHERE, skin, [0.06, 0.07, 0.06], [0, -0.58, 0]));
      if (side === 1) arm.add(mesh(CYLINDER, new THREE.MeshStandardMaterial({ color: 0x7a5a36 }), [0.025, 1.7, 0.025], [0, -0.45, 0.1]));
      root.add(arm);
      g.arms.push(arm);
    }
    root.add(mesh(new THREE.CapsuleGeometry(0.2, 0.42, 4, 10), shirt, [1, 1, 0.75], [0, 1.22, 0]));
    root.add(mesh(SPHERE, skin, [0.13, 0.15, 0.13], [0, 1.68, 0]));
    root.add(mesh(CYLINDER, hat, [0.25, 0.02, 0.25], [0, 1.78, 0]));
    root.add(mesh(CYLINDER, hat, [0.13, 0.12, 0.13], [0, 1.84, 0]));
    return g;
  }

  makeCrocodile() {
    const model = new THREE.Group();
    const skin = new THREE.MeshStandardMaterial({ color: 0x3c4729, roughness: 0.75, flatShading: true });
    const belly = new THREE.MeshStandardMaterial({ color: 0x8d8a5a, roughness: 0.9, flatShading: true });
    const eye = new THREE.MeshStandardMaterial({ color: 0xd9b81c, roughness: 0.3 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x111111 });
    const tooth = new THREE.MeshStandardMaterial({ color: 0xeeeedd });

    model.add(mesh(SPHERE, skin, [0.42, 0.2, 0.95], [0, 0.24, 0]));
    model.add(mesh(SPHERE, belly, [0.38, 0.12, 0.88], [0, 0.17, 0]));
    for (let i = 0; i < 6; i++) {
      model.add(mesh(new THREE.ConeGeometry(0.05, 0.12, 4), skin, null, [0.1, 0.44 - Math.abs(i - 2.5) * 0.02, 0.6 - i * 0.24]));
      model.add(mesh(new THREE.ConeGeometry(0.05, 0.12, 4), skin, null, [-0.1, 0.44 - Math.abs(i - 2.5) * 0.02, 0.6 - i * 0.24]));
    }
    const head = new THREE.Group();
    head.position.set(0, 0.26, 0.85);
    head.add(mesh(SPHERE, skin, [0.27, 0.15, 0.32], [0, 0, 0.12]));
    head.add(mesh(SPHERE, skin, [0.17, 0.085, 0.55], [0, -0.03, 0.6]));
    for (const side of [-1, 1]) {
      head.add(mesh(SPHERE, skin, [0.08, 0.07, 0.08], [side * 0.12, 0.12, 0.1]));
      head.add(mesh(SPHERE, eye, [0.045, 0.045, 0.045], [side * 0.13, 0.16, 0.14]));
      head.add(mesh(SPHERE, dark, [0.015, 0.035, 0.02], [side * 0.13, 0.165, 0.18]));
      for (let t = 0; t < 4; t++) {
        const fang = mesh(new THREE.ConeGeometry(0.018, 0.06, 4), tooth, null, [side * 0.13, -0.08, 0.35 + t * 0.15]);
        fang.rotation.x = Math.PI;
        head.add(fang);
      }
    }
    model.add(head);

    const tail = [];
    let parent = model;
    for (let i = 0; i < 5; i++) {
      const seg = new THREE.Group();
      seg.position.set(0, i === 0 ? 0.22 : 0, i === 0 ? -0.85 : -0.36);
      const k = 1 - i * 0.17;
      seg.add(mesh(SPHERE, skin, [0.27 * k, 0.13 * k, 0.3], [0, 0, -0.18]));
      seg.add(mesh(new THREE.ConeGeometry(0.04, 0.12, 4), skin, null, [0, 0.13 * k, -0.18]));
      parent.add(seg);
      tail.push(seg);
      parent = seg;
    }
    const legs = [];
    for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      const leg = new THREE.Group();
      leg.position.set(sx * 0.3, 0.2, sz * 0.48);
      const limb = mesh(SPHERE, skin, [0.16, 0.07, 0.1], [sx * 0.1, -0.06, 0]);
      limb.rotation.y = sx * 0.6;
      leg.add(limb);
      model.add(leg);
      legs.push(leg);
    }
    model.scale.setScalar(0.75);
    return { model, tail, legs };
  }

  makeHippo() {
    const model = new THREE.Group();
    const skin = new THREE.MeshStandardMaterial({ color: 0x67585f, roughness: 0.55 });
    const pink = new THREE.MeshStandardMaterial({ color: 0x9d7479, roughness: 0.6 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x15100f, roughness: 0.4 });

    const body = mesh(SPHERE, skin, [0.8, 0.68, 1.15], [0, 1.0, 0]);
    model.add(body);
    model.add(mesh(SPHERE, pink, [0.7, 0.45, 1.0], [0, 0.82, 0]));
    const head = new THREE.Group();
    head.position.set(0, 1.12, 1.05);
    head.add(mesh(SPHERE, skin, [0.5, 0.44, 0.5], [0, 0.05, 0.15]));
    head.add(mesh(SPHERE, skin, [0.56, 0.38, 0.42], [0, -0.14, 0.6]));
    head.add(mesh(SPHERE, pink, [0.5, 0.2, 0.36], [0, -0.32, 0.6]));
    for (const side of [-1, 1]) {
      head.add(mesh(SPHERE, dark, [0.06, 0.035, 0.05], [side * 0.18, 0.17, 0.92]));
      head.add(mesh(SPHERE, skin, [0.12, 0.1, 0.12], [side * 0.26, 0.4, 0.22]));
      head.add(mesh(SPHERE, dark, [0.05, 0.05, 0.05], [side * 0.29, 0.43, 0.31]));
      head.add(mesh(SPHERE, skin, [0.07, 0.1, 0.05], [side * 0.3, 0.52, 0.02]));
    }
    model.add(head);
    const legs = [];
    for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      const leg = new THREE.Group();
      leg.position.set(sx * 0.45, 0.6, sz * 0.62);
      leg.add(mesh(CYLINDER, skin, [0.24, 0.62, 0.24], [0, -0.3, 0]));
      model.add(leg);
      legs.push(leg);
    }
    model.add(mesh(new THREE.ConeGeometry(0.06, 0.3, 5), skin, null, [0, 1.1, -1.18]));
    model.scale.setScalar(0.9);
    return { model, tail: [], legs, body, head };
  }

  makeAnimal(a) {
    const root = new THREE.Group();
    let parts;
    if (this.models[a.kind]) {
      const m = this.fromMesh(a.kind, a.kind === 'hippo' ? 2.3 : 2.8, false);
      parts = { model: m.holder, tail: [], legs: [], mixer: m.mixer, legBones: m.legBones, fromFile: true };
    } else {
      parts = a.kind === 'crocodile' ? this.makeCrocodile() : this.makeHippo();
    }
    root.add(parts.model);
    // the "!" above an animal that is running away
    const marker = new THREE.Sprite(this.alertMaterial());
    marker.position.y = a.kind === 'hippo' ? 2.3 : 1.4;
    marker.scale.setScalar(0.65);
    marker.visible = false;
    root.add(marker);
    this.scene.add(root);
    return { ...parts, root, marker, kind: a.kind, target: new THREE.Vector3(), phase: Math.random() * 6, fleeing: false, leaving: null };
  }

  alertMaterial() {
    if (!this.alert) {
      const texture = canvasTexture(64, 64, (g) => {
        g.fillStyle = '#facc15';
        g.beginPath();
        g.arc(32, 32, 28, 0, Math.PI * 2);
        g.fill();
        g.fillStyle = '#1f2937';
        g.font = 'bold 44px sans-serif';
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText('!', 32, 34);
      });
      this.alert = new THREE.SpriteMaterial({ map: texture, depthTest: false });
    }
    return this.alert;
  }

  // effects

  addEffect(object, duration, step) {
    this.scene.add(object);
    this.effects.push({ object, start: this.time, duration, step });
  }

  scareWave(position, radius) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 64),
      new THREE.MeshBasicMaterial({ color: 0xfde047, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(position.x, 0.12, position.z);
    const size = (radius + 0.5) * CELL;
    this.addEffect(ring, 0.55, (t) => {
      ring.scale.setScalar(0.3 + t * size * 1.2);
      ring.material.opacity = 1 - t;
    });
    const area = new THREE.Mesh(new THREE.PlaneGeometry(size * 2, size * 2),
      new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.22, depthWrite: false }));
    area.rotation.x = -Math.PI / 2;
    area.position.set(position.x, 0.06, position.z);
    this.addEffect(area, 0.8, (t) => { area.material.opacity = 0.22 * (1 - t); });
  }

  ripple(position) {
    for (let k = 0; k < 3; k++) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 48),
        new THREE.MeshBasicMaterial({ color: 0xe0f2fe, transparent: true, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(position.x, -0.2, position.z);
      this.addEffect(ring, 1.4 + k * 0.3, (t) => {
        ring.scale.setScalar(0.2 + t * (2 + k));
        ring.material.opacity = 0.8 * (1 - t);
      });
    }
  }

  puff(position, colour) {
    for (let k = 0; k < 10; k++) {
      const bit = new THREE.Mesh(SPHERE, new THREE.MeshStandardMaterial({ color: colour, transparent: true }));
      bit.scale.setScalar(0.08 + Math.random() * 0.08);
      const v = new THREE.Vector3((Math.random() - 0.5) * 2, 1.5 + Math.random() * 1.5, (Math.random() - 0.5) * 2);
      bit.position.set(position.x, 0.3, position.z);
      this.addEffect(bit, 0.9, (t, dt) => {
        v.y -= 6 * dt;
        bit.position.addScaledVector(v, dt);
        bit.material.opacity = 1 - t;
      });
    }
  }

  // state updates

  update(s) {
    const newEpisode = s.episode !== this.episode;
    this.episode = s.episode;

    // crops: reset on a new episode, otherwise look for cells that were just eaten
    if (newEpisode) {
      this.eatenCells.clear();
      this.resetCrops();
      this.colourPlayArea(s.grid);
    } else {
      let eaten = false;
      for (const [x, y] of this.cropCells) {
        const cell = this.cropPlants.get(`${x},${y}`);
        if (s.grid[y][x] !== CROP && cell.eatenAt === null) {
          cell.eatenAt = this.time;
          cell.done = false;
          this.eatenCells.add(`${x},${y}`);
          this.puff(this.cellPos(x, y), 0x6f9a32);
          eaten = true;
        }
      }
      if (eaten) this.colourPlayArea(s.grid);
    }

    // guardian
    const [gx, gy] = s.guardian;
    this.guardian.target.copy(this.cellPos(gx, gy));
    if (newEpisode) this.guardian.root.position.copy(this.guardian.target);

    // animals: ids start again each episode, so clear everything on a new one
    if (newEpisode) {
      for (const a of this.animals.values()) this.scene.remove(a.root);
      this.animals.clear();
    }
    const seen = new Set();
    for (const a of s.animals) {
      seen.add(a.id);
      let obj = this.animals.get(a.id);
      const target = this.cellPos(a.x, a.y);
      if (obj && obj.leaving !== null) {
        // the same animal is back from the river before its old body finished sinking
        this.scene.remove(obj.root);
        obj = null;
      }
      if (!obj) {
        obj = this.makeAnimal(a);
        this.animals.set(a.id, obj);
        if (newEpisode) {
          obj.root.position.copy(target);
          obj.root.rotation.y = 0; // facing the village (+z)
        } else {
          // climbs out of the river
          obj.root.position.set(target.x, -0.9, target.z - 3);
          obj.root.rotation.y = 0;
          this.ripple(obj.root.position);
        }
      }
      obj.target.copy(target);
      obj.fleeing = a.fleeing;
      obj.marker.visible = a.fleeing;
    }
    for (const [id, obj] of this.animals) {
      if (seen.has(id) || obj.leaving !== null) continue;
      obj.leaving = this.time; // it went back into the river, swim off and sink
      obj.marker.visible = false;
      obj.target.set(obj.root.position.x, -1.4, this.riverNear - 4);
      this.ripple(new THREE.Vector3(obj.root.position.x, 0, this.riverNear - 1));
    }

    if (s.events.includes('scare')) {
      this.guardian.hop = 1;
      this.scareWave(this.guardian.target, s.scare_radius);
    }
    if (s.events.includes('breach')) this.shake = 1;
  }

  setCamera(mode) {
    this.cameraMode = mode;
    this.controls.enabled = mode === 4;
    if (mode === 4) this.controls.target.copy(this.camLook);
  }

  toggleGrid() {
    this.gridLines.visible = !this.gridLines.visible;
  }

  // every frame

  frame(dt) {
    this.time += dt;
    this.water.material.uniforms.time.value += dt * 0.5;

    // guardian walks to its cell and faces where it goes
    const g = this.guardian;
    const toTarget = g.target.clone().sub(g.root.position);
    const moving = toTarget.length() > 0.05;
    if (moving) turnToward(g.root, Math.atan2(toTarget.x, toTarget.z), 1 - Math.exp(-dt * 12));
    walkToward(g.root.position, g.target, 4.5, dt);
    g.hop = Math.max(0, g.hop - dt * 3);
    g.root.position.y = Math.sin(g.hop * Math.PI) * 0.35;
    g.phase += dt * (moving ? 11 : 0);
    if (g.mixer) {
      g.action.timeScale = moving ? 1.6 : 0;
      g.mixer.update(dt);
    } else {
      const swing = moving ? Math.sin(g.phase) * 0.6 : 0;
      g.legs.forEach((leg, i) => { leg.rotation.x += ((i ? swing : -swing) - leg.rotation.x) * 0.3; });
      g.arms.forEach((arm, i) => {
        const raised = g.hop > 0 && i === 1 ? -2.4 : (i ? -swing : swing) * 0.7;
        arm.rotation.x += (raised - arm.rotation.x) * 0.3;
      });
    }

    // animals
    for (const [id, a] of this.animals) {
      const d = a.target.clone().sub(a.root.position);
      const walking = Math.hypot(d.x, d.z) > 0.05;
      if (walking) turnToward(a.root, Math.atan2(d.x, d.z), 1 - Math.exp(-dt * 8));
      if (a.leaving !== null) a.root.position.addScaledVector(d, 1 - Math.exp(-dt * 2)); // swim off and sink
      else walkToward(a.root.position, a.target, a.fleeing ? 5 : 3, dt);
      a.phase += dt * (walking ? (a.fleeing ? 14 : 8) : 1.5);
      a.tail.forEach((seg, i) => { seg.rotation.y = Math.sin(a.phase - i * 0.7) * (walking ? 0.3 : 0.08); });
      a.legs.forEach((leg, i) => { leg.rotation.x = walking ? Math.sin(a.phase + (i % 2) * Math.PI) * 0.5 : 0; });
      if (a.body) a.model.position.y = walking ? Math.abs(Math.sin(a.phase)) * 0.05 : Math.sin(a.phase) * 0.01;
      if (a.fromFile) {
        // model files have no walk animation, so sway the body and swing the legs a little
        a.model.position.y = walking ? Math.abs(Math.sin(a.phase)) * 0.06 : 0;
        a.model.rotation.z = walking ? Math.sin(a.phase) * 0.05 : 0;
        a.legBones.forEach((leg, i) => {
          const swing = walking ? Math.sin(a.phase + (i % 2) * Math.PI) * 0.35 : 0;
          leg.bone.quaternion.copy(leg.rest).multiply(new THREE.Quaternion().setFromAxisAngle(X_AXIS, swing));
        });
      }
      if (a.mixer) a.mixer.update(dt * (walking ? 1 : 0.2));
      a.marker.position.y = (a.kind === 'hippo' ? 2.3 : 1.4) + Math.sin(this.time * 6) * 0.08;
      if (a.leaving !== null && this.time - a.leaving > 1.6) {
        this.scene.remove(a.root);
        this.animals.delete(id);
      }
    }

    this.animateCrops();

    for (let i = this.effects.length - 1; i >= 0; i--) {
      const e = this.effects[i];
      const t = Math.min(1, (this.time - e.start) / e.duration);
      e.step(t, dt);
      if (t >= 1) {
        this.scene.remove(e.object);
        this.effects.splice(i, 1);
      }
    }

    this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
  }

  updateCamera(dt) {
    if (this.cameraMode === 4) {
      this.controls.update();
      this.camLook.copy(this.controls.target);
      return;
    }
    const g = this.guardian.root.position;
    let pos, look;
    if (this.cameraMode === 0) {          // chase: behind the guardian, looking at the river
      pos = new THREE.Vector3(g.x, 7.5, g.z + 9.5);
      look = new THREE.Vector3(g.x, 0.8, g.z - 5);
    } else if (this.cameraMode === 1) {   // over the guardian's shoulder
      pos = new THREE.Vector3(g.x + 0.6, 2.6, g.z + 3.4);
      look = new THREE.Vector3(g.x, 1.0, g.z - 8);
    } else if (this.cameraMode === 2) {   // straight down on the whole grid
      pos = new THREE.Vector3(0, 44, 2);
      look = new THREE.Vector3(0, 0, 0);
    } else {                              // slow orbit around the village
      const angle = this.time * 0.07;
      pos = new THREE.Vector3(Math.sin(angle) * 34, 22, Math.cos(angle) * 34);
      look = new THREE.Vector3(0, 0, -2);
    }
    const k = 1 - Math.exp(-dt * 3);
    this.camera.position.lerp(pos, k);
    this.camLook.lerp(look, k);
    if (this.shake > 0) {
      this.shake = Math.max(0, this.shake - dt * 1.5);
      this.camera.position.x += (Math.random() - 0.5) * this.shake * 0.6;
      this.camera.position.y += (Math.random() - 0.5) * this.shake * 0.6;
    }
    this.camera.lookAt(this.camLook);
  }
}
