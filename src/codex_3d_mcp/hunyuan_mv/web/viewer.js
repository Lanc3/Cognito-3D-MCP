import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { RoomEnvironment } from "./vendor/RoomEnvironment.js";

const $ = (id) => document.getElementById(id);
const count = (value) => Number.isFinite(value) ? value.toLocaleString() : "—";

function modelURL(value) {
  if (typeof value !== "string" || !value.startsWith("/")) throw new Error("Invalid model path.");
  const url = new URL(value, location.origin);
  const parts = value.split("/").slice(1).map((part) => decodeURIComponent(part));
  if (url.origin !== location.origin || url.search || url.hash || url.username || url.password ||
      parts.length < 4 || !["artifacts", "models"].includes(parts[0]) ||
      !parts.slice(1, 3).every((part) => /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(part)) ||
      parts.some((part) => !part || /[\\/:\x00-\x1f\x7f%]/.test(part) || /[. ]$/.test(part)) ||
      !/\.glb$/i.test(parts.at(-1)) ||
      (parts[0] === "models" && (parts.length !== 4 || parts[3] !== "original-shape.glb"))) {
    throw new Error("This model is outside the asset's local viewer routes.");
  }
  return url.href;
}

function embeddedGLB(buffer) {
  const header = new DataView(buffer);
  if (buffer.byteLength < 20 || header.getUint32(0, true) !== 0x46546c67 ||
      header.getUint32(4, true) !== 2 || header.getUint32(8, true) !== buffer.byteLength ||
      header.getUint32(16, true) !== 0x4e4f534a) throw new Error("The file is not a valid GLB model.");
  const length = header.getUint32(12, true);
  if (length + 20 > buffer.byteLength) throw new Error("The GLB file is incomplete.");
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, length)));
  if ([...(json.buffers || []), ...(json.images || [])].some((entry) => entry.uri !== undefined)) {
    throw new Error("The viewer requires textures and buffers embedded in the GLB.");
  }
}

function disposeObject(object) {
  if (!object) return;
  const geometries = new Set(), materials = new Set(), textures = new Set(), images = new Set();
  object.traverse((item) => {
    if (item.geometry) geometries.add(item.geometry);
    for (const material of (Array.isArray(item.material) ? item.material : [item.material])) {
      if (!material) continue;
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  for (const texture of textures) {
    if (texture.source?.data) images.add(texture.source.data);
    texture.dispose();
  }
  for (const image of images) if (typeof image.close === "function") image.close();
  for (const material of materials) material.dispose();
  for (const geometry of geometries) geometry.dispose();
  object.removeFromParent();
}

export class ModelViewer {
  constructor() {
    this.dialog = $("model-dialog");
    this.viewport = $("model-viewport");
    this.generation = 0;
    this.loadChain = Promise.resolve();
    this.frame = null;
    $("model-close").addEventListener("click", () => this.dialog.close());
    this.dialog.addEventListener("close", () => this.close());
    $("model-select").addEventListener("change", () => this.select($("model-select").value));
    $("model-reset").addEventListener("click", () => this.resetView());
    $("model-wireframe").addEventListener("change", () => this.setWireframe());
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) this.requestRender();
    });
    window.addEventListener("pagehide", () => this.close());
  }

  open(asset) {
    this.asset = asset;
    this.models = (asset.models || []).filter((model) => {
      try { modelURL(model.url); return true; } catch { return false; }
    });
    if (!this.models.length) throw new Error("No generated model is available yet.");
    $("model-title").textContent = asset.name || "Generated asset";
    $("model-select").replaceChildren(...this.models.map((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.label}${model.textured ? " · Textured" : " · Untextured"}${model.stale ? " · Previous version" : ""}`;
      return option;
    }));
    $("model-wireframe").checked = false;
    if (!this.dialog.open) this.dialog.showModal();
    this.initialView = true;
    try {
      this.createRenderer();
      this.select((this.models.find((m) => m.id === "full_game" && m.textured && !m.stale) ||
        this.models.find((m) => m.textured && !m.stale) ||
        this.models.find((m) => m.id.startsWith("repair_attempt_") && m.current && !m.approved) ||
        this.models.find((m) => m.id === "accepted_master" && !m.stale) || this.models[0]).id);
    } catch (error) {
      this.message("3D rendering could not start. Check that hardware acceleration is available; the GLB downloads still work.", true);
      this.disposeRenderer();
      console.error(error);
    }
  }

  createRenderer() {
    if (this.renderer) return;
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false, powerPreference: "high-performance"});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(getComputedStyle(this.viewport).getPropertyValue("--viewer-bg").trim() || "#181818");
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    const room = new RoomEnvironment();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.environment = pmrem.fromScene(room, 0.04);
    this.scene.environment = this.environment.texture;
    this.scene.environmentIntensity = 1.2;
    room.dispose();
    pmrem.dispose();
    this.scene.add(new THREE.HemisphereLight(0xf5f5f5, 0x737373, 1.5));
    const light = new THREE.DirectionalLight(0xffffff, 2);
    light.position.set(3, 5, 4);
    this.scene.add(light);
    const canvas = this.renderer.domElement;
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", "3D model: drag to rotate, scroll to zoom, right-drag to pan. Arrow keys pan.");
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      if (this.dialog.open && this.renderer) this.message("The graphics context was lost. Close and reopen this viewer to retry.", true);
    });
    this.viewport.prepend(canvas);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.autoRotate = false;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 14;
    this.controls.listenToKeyEvents(canvas);
    this.controls.addEventListener("change", () => this.requestRender());
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(this.viewport);
    this.resize();
  }

  resize() {
    if (!this.renderer || !this.dialog.open) return;
    const width = Math.max(1, this.viewport.clientWidth);
    const height = Math.max(1, this.viewport.clientHeight);
    const previousAspect = this.camera.aspect;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    if (this.model && this.controls && previousAspect !== this.camera.aspect) {
      // Preserve the user's orbit and relative zoom when the inspector narrows.
      const ratio = this.fitDistance(this.camera.aspect) / this.fitDistance(previousAspect);
      this.camera.position.sub(this.controls.target).multiplyScalar(ratio).add(this.controls.target);
      this.controls.maxDistance = Math.max(14, this.fitDistance() * 3);
      this.controls.update();
    }
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  fitDistance(aspect = this.camera.aspect) {
    const vertical = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * Math.max(aspect, 0.01));
    return (this.modelRadius || Math.sqrt(3)) / Math.sin(Math.min(vertical, horizontal)) * 1.12;
  }

  requestRender() {
    if (this.frame !== null || !this.renderer || !this.dialog.open || document.hidden) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      if (this.renderer && this.dialog.open && !document.hidden) {
        this.renderer.render(this.scene, this.camera);
        this.viewport.dataset.renderCount = String(Number(this.viewport.dataset.renderCount || 0) + 1);
      }
    });
  }

  resetView() {
    if (!this.controls) return;
    this.controls.target.set(0, 0, 0);
    // +Y is glTF up and +Z is the Hunyuan front. Frame a useful three-quarter view.
    const distance = this.fitDistance();
    this.controls.maxDistance = Math.max(14, distance * 3);
    this.camera.position.set(0.65, 0.43, 0.85).normalize().multiplyScalar(distance);
    this.camera.lookAt(0, 0, 0);
    this.controls.update();
    this.requestRender();
  }

  message(text, failed = false) {
    $("model-message").textContent = text;
    $("model-message").hidden = !text;
    $("model-message").classList.toggle("failed", failed);
    this.viewport.dataset.state = failed ? "error" : text ? "loading" : "ready";
  }

  select(id) {
    const descriptor = this.models.find((model) => model.id === id);
    if (!descriptor) return;
    $("model-select").value = id;
    const generation = ++this.generation;
    this.controller?.abort();
    this.removeModel();
    $("model-profile-name").textContent = descriptor.label;
    $("model-quads").textContent = count(descriptor.quad_count);
    $("model-target").textContent = count(descriptor.target_quads);
    $("model-triangles").textContent = count(descriptor.triangle_count);
    $("model-texture-state").textContent = descriptor.textured ? "Textures on" : "Untextured source";
    $("model-texture-state").className = `badge ${descriptor.textured ? "completed" : ""}`;
    $("model-topology-note").textContent = Number.isFinite(descriptor.quad_count)
      ? "Quad count is measured on the cleaned AutoRemesher mesh before GLB triangulation."
      : "This source contains triangles. No native quad count is recorded.";
    $("model-stage-note").textContent = descriptor.stale
      ? "Previous attempt, retained for comparison. This model is not the current accepted output."
      : descriptor.id === "shape"
      ? "Original Hunyuan geometry, before remeshing and painting. It has no texture; neutral clay shading helps show its shape."
      : descriptor.id.startsWith("repair_attempt_")
      ? "Repair candidate awaiting geometry checks and agent review. Compare its underside and small parts with the original before acceptance."
      : descriptor.id === "accepted_master"
      ? "Accepted master, bound to its geometry report and agent review. All three remesh profiles use this version."
      : `${descriptor.approved ? "Accepted" : "Awaiting review"} ${descriptor.stage || "generated"} output. ${descriptor.textured ? "Saved textures and materials are displayed." : "Textures become available after painting and baking."}`;
    $("model-download").href = modelURL(descriptor.url);
    this.message("Loading model…");
    this.requestRender();
    // Wait for an interrupted parse to dispose before loading another model.
    this.loadChain = this.loadChain.catch(() => {}).then(() => this.load(descriptor, generation));
  }

  async load(descriptor, generation) {
    if (generation !== this.generation || !this.dialog.open) return;
    this.controller = new AbortController();
    let object = null;
    try {
      const response = await fetch(modelURL(descriptor.url), {signal: this.controller.signal, cache: "no-store"});
      if (!response.ok) throw new Error("This model is no longer available. Close the viewer and refresh the queue.");
      const buffer = await response.arrayBuffer();
      if (generation !== this.generation || !this.dialog.open) return;
      embeddedGLB(buffer);
      const manager = new THREE.LoadingManager();
      manager.setURLModifier((url) => {
        if (!url.startsWith(`blob:${location.origin}/`)) throw new Error("External model resources are unsupported.");
        return url;
      });
      const gltf = await new GLTFLoader(manager).parseAsync(buffer, "");
      object = gltf.scene;
      if (generation !== this.generation || !this.dialog.open) { disposeObject(object); return; }
      let triangles = 0;
      const meshes = [];
      object.traverse((item) => {
        if (!item.isMesh) return;
        meshes.push(item);
        triangles += (item.geometry.index?.count ?? item.geometry.attributes.position?.count ?? 0) / 3;
        if (!descriptor.textured) {
          // Display-only clay shading for raw and repaired meshes; GLB bytes stay unchanged.
          for (const material of (Array.isArray(item.material) ? item.material : [item.material])) {
            if (!material) continue;
            material.color?.set(0x858585);
            if ("metalness" in material) material.metalness = 0;
            if ("roughness" in material) material.roughness = 0.82;
          }
        }
      });
      if (!meshes.length || !Number.isFinite(triangles)) throw new Error("The model has no displayable mesh.");
      const bounds = new THREE.Box3().setFromObject(object);
      const size = bounds.getSize(new THREE.Vector3());
      const span = Math.max(size.x, size.y, size.z);
      if (!Number.isFinite(span) || span <= 0) throw new Error("The model has invalid bounds.");
      const center = bounds.getCenter(new THREE.Vector3());
      const wrapper = new THREE.Group();
      wrapper.add(object);
      object.position.sub(center);
      wrapper.scale.setScalar(2 / span);
      this.model = wrapper;
      this.modelRadius = bounds.getBoundingSphere(new THREE.Sphere()).radius * (2 / span);
      this.meshes = meshes;
      this.scene.add(wrapper);
      $("model-triangles").textContent = count(Math.round(triangles));
      this.viewport.dataset.modelId = descriptor.id;
      this.viewport.dataset.textured = String(descriptor.textured);
      if (this.initialView) { this.resetView(); this.initialView = false; }
      this.setWireframe();
      this.message("");
      this.requestRender();
    } catch (error) {
      if (object && !this.model) disposeObject(object);
      if (generation !== this.generation || error.name === "AbortError") return;
      this.removeModel();
      this.message(error.message || "The model could not be loaded.", true);
    }
  }

  setWireframe() {
    for (const mesh of this.meshes || []) {
      let wire = mesh.children.find((child) => child.userData.viewerOverlay);
      if (!wire && $("model-wireframe").checked) {
        wire = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({
          color: 0x111111, wireframe: true, transparent: true, opacity: 0.72, depthWrite: false,
        }));
        wire.userData.viewerOverlay = true;
        mesh.add(wire);
      }
      if (wire) wire.visible = $("model-wireframe").checked;
    }
    this.requestRender();
  }

  removeModel() {
    disposeObject(this.model);
    this.model = null;
    this.modelRadius = null;
    this.meshes = [];
    if (this.renderer) this.renderer.renderLists.dispose();
    delete this.viewport.dataset.modelId;
  }

  disposeRenderer() {
    this.observer?.disconnect();
    this.controls?.dispose();
    this.environment?.dispose();
    if (this.renderer) {
      const renderer = this.renderer;
      this.renderer = null;
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    }
    this.controls = this.environment = this.observer = this.scene = this.camera = null;
  }

  close() {
    ++this.generation;
    this.controller?.abort();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.removeModel();
    this.disposeRenderer();
    this.viewport.dataset.state = "closed";
  }
}
