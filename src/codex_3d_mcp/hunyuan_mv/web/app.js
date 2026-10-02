"use strict";

const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="queue-session-token"]').content;
const stageNames = ["references", "shape", "shape_repair", "remesh", "paint", "finish"];
const stageLabels = ["References", "Shape", "Repair & review", "Remesh ×3", "Paint", "Bake & finish"];
const profileNames = ["full_game", "mobile", "browser"];
const profileLabels = {full_game: "Full game", mobile: "Mobile", browser: "Browser"};
const canonicalViews = ["front", "back", "left", "right"];
const attentionStates = new Set(["awaiting_agent", "awaiting_review", "needs_repair", "awaiting_images", "interrupted", "failed", "failed_quality", "blocked"]);
const completeStates = new Set(["completed", "accepted", "finished"]);
const stateLabels = {awaiting_agent: "Agent review", awaiting_review: "Agent review", needs_repair: "Needs repair", needs_evidence: "Preview pending", awaiting_images: "Needs references", interrupted: "Interrupted", failed_quality: "Quality check failed", pause_requested: "Pausing", running: "In progress", completed: "Completed", approved: "Phase accepted", accepted: "Accepted"};
const preferenceKey = "cognito-3d-dashboard-preferences-v1";
const preferences = readPreferences();
const dialogReturnTargets = new Map();
const mobileSidebar = matchMedia("(max-width: 760px)");
let snapshot = {batches: []};
let selectedBatchId = preferences.batchId || null;
let currentSignature = "";
let busy = false;
let activeRefresh = Promise.resolve();
let commandPending = false;
let pendingAction = "";
let hasSnapshot = false;
let loadAttempted = false;
let noticeSource = "";
let sessionExpired = false;
let modelViewer = null;
let viewerModule = null;
let cancelBatchId = null;

async function showModel(asset) {
  const returnTarget = focusTarget();
  try {
    viewerModule ||= import("/viewer.js");
    const { ModelViewer } = await viewerModule;
    modelViewer ||= new ModelViewer();
    dialogReturnTargets.set("model-dialog", returnTarget);
    modelViewer.open(asset);
  } catch (error) {
    viewerModule = null;
    notice(error.message || "The 3D viewer could not open. The model downloads remain available.");
  }
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function icon(name) {
  const paths = {
    cube: ["m12 3 9 5v8l-9 5-9-5V8l9-5Z", "m3 8 9 5 9-5M12 13v8M7.5 5.5l9 5"],
    check: ["m5 12 4 4L19 6"],
    download: ["M12 3v12m-5-5 5 5 5-5", "M5 16v5h14v-5"],
    image: ["M3 3h18v18H3z", "m3 16 5-5 4 4 3-3 6 6", "M16 7h.01"],
  };
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [key, value] of Object.entries({viewBox: "0 0 24 24", width: "18", height: "18", fill: "none", stroke: "currentColor", "stroke-width": "1.6", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false", class: "ui-icon"})) svg.setAttribute(key, value);
  for (const data of paths[name] || paths.cube) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", data);
    svg.append(path);
  }
  return svg;
}

function readPreferences() {
  try {
    const value = JSON.parse(localStorage.getItem(preferenceKey) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}

function savePreferences() {
  try {
    localStorage.setItem(preferenceKey, JSON.stringify({
      batchId: selectedBatchId,
      filter: $("asset-filter").value,
      layout: $("assets").classList.contains("list-layout") ? "list" : "grid",
    }));
  } catch { /* Storage may be unavailable in private or restricted sessions. */ }
}

function focusTarget(element = document.activeElement) {
  if (!element || element === document.body) return null;
  if (element.id) return {id: element.id};
  const batch = element.closest("[data-batch-id]");
  if (batch) return {batchId: batch.dataset.batchId};
  const asset = element.closest("[data-asset-id]");
  const keyed = element.closest("[data-focus-key]");
  return asset && keyed ? {assetId: asset.dataset.assetId, key: keyed.dataset.focusKey} : null;
}

function restoreFocus(target, fallback = false) {
  if (!target) return;
  let element = target.id ? $(target.id) : null;
  if (target.batchId) element = [...$("batches").querySelectorAll("[data-batch-id]")].find((item) => item.dataset.batchId === target.batchId);
  if (target.assetId) {
    const asset = [...$("assets").querySelectorAll("[data-asset-id]")].find((item) => item.dataset.assetId === target.assetId);
    element = asset && [...asset.querySelectorAll("[data-focus-key]")].find((item) => item.dataset.focusKey === target.key);
  }
  if (element && !element.disabled && !element.closest("[hidden], [inert]") && element.getClientRects().length) element.focus({preventScroll: true});
  else if (document.body.classList.contains("sidebar-open")) sidebarFocusables()[0]?.focus({preventScroll: true});
  else if (fallback && !$("workspace").hidden) $("asset-filter").focus({preventScroll: true});
}

function setSidebarOpen(open, returnFocus = false) {
  const sidebar = $("sidebar");
  const focusWasInSidebar = sidebar?.contains(document.activeElement);
  const wasOpen = document.body.classList.contains("sidebar-open");
  open = Boolean(open && mobileSidebar.matches);
  document.body.classList.toggle("sidebar-open", open);
  $("sidebar-toggle")?.setAttribute("aria-expanded", String(open));
  if ($("sidebar-backdrop")) $("sidebar-backdrop").hidden = !open;
  if (sidebar) {
    sidebar.inert = mobileSidebar.matches && !open;
    if (open) { sidebar.setAttribute("role", "dialog"); sidebar.setAttribute("aria-modal", "true"); }
    else { sidebar.removeAttribute("role"); sidebar.removeAttribute("aria-modal"); }
  }
  const main = $("main-content") || document.querySelector("main");
  if (main) main.inert = open;
  const skip = document.querySelector(".skip-link");
  if (skip) skip.inert = open;
  if (open && (!wasOpen || !focusWasInSidebar)) ($("batches").querySelector(".batch-link.active, .batch-link") || $("sidebar-close") || sidebar?.querySelector("a[href]"))?.focus({preventScroll: true});
  else if (!open && (returnFocus || (mobileSidebar.matches && focusWasInSidebar))) $("sidebar-toggle")?.focus({preventScroll: true});
}

function sidebarFocusables() {
  return [...$("sidebar").querySelectorAll("a[href], button, input, select, textarea, [tabindex]")].filter((element) =>
    element.tabIndex >= 0 && !element.disabled && !element.closest("[hidden], [inert]") && element.getClientRects().length);
}

function renderBatches(batches) {
  const navigation = $("batches");
  const existing = new Map([...navigation.querySelectorAll("[data-batch-id]")].map((element) => [element.dataset.batchId, element]));
  const links = batches.map((batch) => {
    const id = batch.batch_id;
    let link = existing.get(id);
    if (!link) {
      link = node("button", "batch-link");
      link.type = "button";
      link.dataset.batchId = id;
      const label = node("span");
      label.append(node("span", "batch-label"), node("span", "batch-meta"));
      link.append(node("span", "batch-dot"), label);
      link.addEventListener("click", () => {
        selectedBatchId = id;
        render();
        if (mobileSidebar.matches) setSidebarOpen(false, true);
      });
    }
    const name = batch.name || "Untitled batch";
    const assetTotal = (batch.assets || []).length;
    const metadata = `${assetTotal} ${assetTotal === 1 ? "asset" : "assets"} · ${stateLabel(batch.state)}`;
    link.classList.toggle("active", id === selectedBatchId);
    link.setAttribute("aria-current", id === selectedBatchId ? "page" : "false");
    link.title = name;
    if (link.querySelector(".batch-label").textContent !== name) link.querySelector(".batch-label").textContent = name;
    if (link.querySelector(".batch-meta").textContent !== metadata) link.querySelector(".batch-meta").textContent = metadata;
    return link;
  });
  if (!links.length) links.push(navigation.querySelector(".sidebar-empty") || node("p", "sidebar-empty", "No batches yet"));
  // Retain button identity and focus when only status or timestamps change.
  links.forEach((link, index) => {
    if (navigation.children[index] !== link) navigation.insertBefore(link, navigation.children[index] || null);
  });
  const retained = new Set(links);
  for (const child of [...navigation.children]) if (!retained.has(child)) child.remove();
}

function controlLabel(button, label) {
  const span = button.querySelector("[data-button-label], .button-label");
  if (span) { span.textContent = label; return; }
  const text = [...button.childNodes].find((child) => child.nodeType === Node.TEXT_NODE && child.textContent.trim());
  if (text) text.textContent = label;
  else button.append(document.createTextNode(label));
}

function readable(value) {
  return String(value || "queued").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function stateLabel(state) { return stateLabels[state] || readable(state); }
function stageIndex(stage) {
  const aliases = {validating_views: "references", awaiting_images: "references", generating_shape: "shape", remeshing: "remesh", texturing: "paint", postprocessing: "finish", finishing: "finish", validating: "finish", completed: "finish"};
  return stageNames.indexOf(aliases[stage] || stage);
}
function tone(state) {
  if (completeStates.has(state) || state === "approved") return "completed";
  if (["failed", "failed_quality", "cancelled"].includes(state)) return "failed";
  if (attentionStates.has(state) || ["paused", "pause_requested"].includes(state)) return "attention";
  return state === "running" ? "running" : "";
}
function badge(state) { return node("span", `badge ${tone(state)}`, stateLabel(state)); }
function selectedBatch() { return snapshot.batches.find((batch) => batch.batch_id === selectedBatchId); }
function stringify(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

// Every image/link is constrained to this server's scoped artifact route. User
// strings are never HTML, CSS, script, or arbitrary URL input.
function artifactURL(value) {
  if (typeof value !== "string" || !value.startsWith("/artifacts/")) return null;
  try {
    const url = new URL(value, location.origin);
    if (url.origin !== location.origin || !url.pathname.startsWith("/artifacts/")) return null;
    if (url.username || url.password || url.search || url.hash) return null;
    // Validate the original path before URL normalization can hide traversal.
    const parts = value.split("/").slice(1).map((part) => decodeURIComponent(part));
    if (parts.length < 4 || !parts.slice(1, 3).every((part) => /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(part))) return null;
    if (parts.some((part) => !part || /[\\/:\x00-\x1f\x7f%]/.test(part) || /[. ]$/.test(part))) return null;
    return url.href;
  } catch { return null; }
}

function assetFileURL(batch, asset, file) {
  if (typeof file !== "string" && (!file || typeof file !== "object")) return null;
  const explicit = artifactURL(typeof file === "string" ? file : file.url);
  if (explicit) return explicit;
  const relative = typeof file === "string" ? file : file.relative_path;
  if (typeof relative !== "string" || /[\\:\x00%]/.test(relative)) return null;
  const parts = relative.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) return null;
  return artifactURL(`/artifacts/${encodeURIComponent(batch.batch_id)}/${encodeURIComponent(asset.asset_id)}/${parts.map(encodeURIComponent).join("/")}`);
}

function previewEntries(asset) {
  return Object.entries(asset.previews || {}).flatMap(([key, value]) => {
    const url = artifactURL(value);
    if (!url) return [];
    const profile = profileNames.find((name) => key.startsWith(`${name}_`));
    const view = profile ? key.slice(profile.length + 1) : key;
    return [{key, view, url, profile, label: readable(view),
      group: profile ? profileLabels[profile] : "Other views"}];
  }).sort((a, b) => {
    const groupA = a.profile ? profileNames.indexOf(a.profile) : profileNames.length;
    const groupB = b.profile ? profileNames.indexOf(b.profile) : profileNames.length;
    const viewA = canonicalViews.includes(a.view) ? canonicalViews.indexOf(a.view) : canonicalViews.length;
    const viewB = canonicalViews.includes(b.view) ? canonicalViews.indexOf(b.view) : canonicalViews.length;
    return groupA - groupB || viewA - viewB || a.key.localeCompare(b.key);
  });
}

function showPreview(asset, entry) {
  dialogReturnTargets.set("preview-dialog", focusTarget());
  $("preview-title").textContent = `${asset.name || asset.asset_id} · ${entry.profile ? `${entry.group} · ` : ""}${entry.label}`;
  const image = $("preview-image");
  const message = $("preview-message");
  image.hidden = true;
  if (message) { message.textContent = "Loading preview…"; message.hidden = false; }
  image.onload = () => { image.hidden = false; if (message) message.hidden = true; };
  image.onerror = () => {
    image.hidden = true;
    if (message) { message.textContent = "This preview is unavailable. Close this window and try another view."; message.hidden = false; }
  };
  image.src = entry.url;
  image.alt = `${asset.name || asset.asset_id}, ${entry.profile ? `${entry.group}, ` : ""}${entry.label} view`;
  $("preview-dialog").showModal();
}

function previewPlaceholder(title, caption) {
  const placeholder = node("div", "visual-placeholder");
  const mark = node("span", "placeholder-icon");
  mark.append(icon("cube"));
  placeholder.append(mark, node("span", "placeholder-title", title), node("span", "placeholder-caption", caption));
  return placeholder;
}

function meshCount(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "—";
}

function renderProfiles(asset) {
  const profiles = asset.profiles || {};
  const names = profileNames.filter((name) => profiles[name]);
  if (!names.length) return null;
  const section = node("section", "profile-metrics");
  section.setAttribute("aria-label", "Mesh targets and measured results");
  const table = node("table");
  table.append(node("caption", "", "Mesh targets & results"));
  const head = node("thead");
  const headings = node("tr");
  for (const title of ["Profile", "Target quads", "Actual quads", "Triangles / limit"]) {
    const heading = node("th", "", title);
    heading.scope = "col";
    headings.append(heading);
  }
  head.append(headings);
  const body = node("tbody");
  for (const name of names) {
    const profile = profiles[name];
    const row = node("tr");
    const label = node("th", "profile-label", profileLabels[name]);
    label.scope = "row";
    label.append(node("small", `profile-state ${tone(profile.state)}`, stateLabel(profile.state || "pending")));
    const target = node("td", "", meshCount(profile.target_quads));
    const actual = node("td", "", meshCount(profile.actual_quads));
    if (typeof profile.native_quads === "number" && profile.native_quads !== profile.actual_quads) {
      actual.append(node("small", "", `${meshCount(profile.native_quads)} raw`));
    }
    const triangles = node("td", "", meshCount(profile.actual_triangles));
    triangles.append(node("small", "", `/ ${meshCount(profile.triangle_budget)} max`));
    if (typeof profile.actual_triangles === "number" && typeof profile.triangle_budget === "number" && profile.actual_triangles > profile.triangle_budget) {
      triangles.classList.add("over-budget");
      triangles.title = "Exported triangle count exceeds this profile's limit";
    }
    row.append(label, target, actual, triangles);
    body.append(row);
  }
  table.append(head, body);
  section.append(table, node("p", "profile-note", "Quad targets guide remeshing. Actual quads are measured after cleanup; triangle limits apply to exported meshes."));
  return section;
}

function buildStages(batch) {
  const position = stageIndex(batch.stage);
  const complete = completeStates.has(batch.state);
  const waiting = attentionStates.has(batch.state);
  const elements = stageNames.map((name, index) => {
    let state = "pending";
    if (complete || (position >= 0 && index < position)) state = "complete";
    else if (index === position) state = waiting ? "waiting" : "current";
    const item = node("li", `stage ${state}`);
    const number = node("span", "stage-number", state === "complete" ? null : String(index + 1));
    if (state === "complete") { number.append(icon("check")); number.setAttribute("aria-label", "Completed"); }
    item.append(number);
    item.append(node("span", "stage-name", stageLabels[index]));
    let detail = state === "complete" ? "Accepted" : state === "pending" ? "Pending" : "In progress";
    if (state === "waiting") detail = "Agent action";
    else if (state === "current" && ["queued", "paused", "interrupted"].includes(batch.state)) detail = stateLabel(batch.state);
    const approvedCount = batch.counts?.[name];
    if (typeof approvedCount === "number" && (batch.assets || []).length) detail = `${approvedCount}/${batch.assets.length} ${name === "shape" ? "generated" : "accepted"}`;
    item.append(node("span", "stage-detail", detail));
    if (["current", "waiting"].includes(state)) item.setAttribute("aria-current", "step");
    return item;
  });
  $("stages").replaceChildren(...elements);
}

function updateGate(batch) {
  const assets = batch.assets || [];
  const attention = attentionStates.has(batch.state) || assets.some((asset) => attentionStates.has(asset.state));
  $("gate").classList.toggle("needs-attention", attention);
  let title = "Agent review controls the next phase";
  let description = "All assets clear this phase before the next begins. References → shape → repair & review → three meshes → paint full game → bake mobile/browser. Work stays serial.";
  if (completeStates.has(batch.state)) {
    title = "This batch has cleared its final review";
    description = "Inspect the model previews and download the finished assets below.";
  } else if (batch.state === "cancelled") {
    title = "Batch cancelled";
    description = "Existing evidence and artifacts remain available below.";
  } else if (["paused", "pause_requested"].includes(batch.state)) {
    title = batch.state === "pause_requested" ? "Queue pause requested" : "Queue paused";
    description = "Resume continues eligible work. Stage approval still belongs to the agent.";
  } else if (batch.state === "awaiting_images" || assets.some((asset) => asset.state === "awaiting_images")) {
    title = "Waiting for reference images";
    description = "The agent prepares and checks the image set before shape generation can begin.";
  } else if (batch.state === "needs_repair" || assets.some((asset) => asset.state === "needs_repair")) {
    title = "A repair is needed before this phase can advance";
    description = "The agent inspects the evidence, corrects the affected asset, and submits another attempt.";
  } else if (batch.state === "interrupted") {
    title = "Work was interrupted";
    description = "Review the last attempt and its evidence. Resume restores eligible work from saved checkpoints.";
  } else if (attention) {
    title = "Waiting for agent review";
    description = "The next phase stays on hold until the agent accepts every asset in this phase.";
  }
  $("gate-title").textContent = title;
  $("gate-description").textContent = description;
}

function renderAsset(batch, asset) {
  const state = asset.state || "queued";
  const card = node("article", "asset-card");
  card.dataset.assetId = asset.asset_id;
  const activeId = snapshot.active_asset_id || batch.active_asset_id;
  if (asset.asset_id === activeId || state === "running") card.classList.add("is-active");
  const previews = previewEntries(asset);
  const visual = node("div", `asset-visual${previews.length ? " has-image" : ""}`);
  if (previews.length) {
    const interactive = Array.isArray(asset.models) && asset.models.length > 0;
    const img = node("img");
    img.src = previews[0].url;
    img.alt = `${asset.name || asset.asset_id}, ${previews[0].view} model preview`;
    img.loading = "lazy";
    img.addEventListener("error", () => {
      img.hidden = true;
      visual.classList.add("preview-unavailable");
      visual.dataset.previewFailed = "true";
      visual.append(previewPlaceholder("Preview unavailable", interactive ? "Open the 3D model to inspect this asset." : "The saved image could not be loaded."));
      visual.querySelector(".visual-tag").textContent = "PREVIEW UNAVAILABLE";
      if (!interactive) {
        visual.removeAttribute("role"); visual.removeAttribute("tabindex"); visual.removeAttribute("aria-label");
        visual.classList.remove("has-image");
        visual.querySelector(".preview-count")?.remove();
      }
    }, {once: true});
    visual.append(img, node("span", "visual-tag", "MODEL PREVIEW"));
    visual.append(node("span", "preview-count", interactive ? "Rotate in 3D ↗" : `${previews.length} ${previews.length === 1 ? "view" : "views"} ↗`));
    visual.dataset.focusKey = "visual";
    visual.setAttribute("role", "button");
    visual.setAttribute("tabindex", "0");
    visual.setAttribute("aria-label", `${interactive ? "View in 3D" : "Enlarge preview"}: ${asset.name || asset.asset_id}`);
    const openVisual = () => {
      if (interactive) showModel(asset);
      else if (visual.dataset.previewFailed !== "true") showPreview(asset, previews[0]);
    };
    visual.addEventListener("click", openVisual);
    visual.addEventListener("keydown", (event) => {
      if (["Enter", " "].includes(event.key)) { event.preventDefault(); openVisual(); }
    });
  } else {
    visual.append(previewPlaceholder(state === "running" ? "Creating this asset" : "Preview will appear here", "Model views appear as evidence becomes available."), node("span", "visual-tag", "AWAITING PREVIEW"));
  }
  const body = node("div", "card-body");
  const heading = node("div", "card-heading");
  const title = node("div");
  title.append(node("h3", "", asset.name || "Untitled asset"), node("p", "asset-id", asset.asset_id));
  heading.append(title, badge(state));
  body.append(heading);
  if (Array.isArray(asset.models) && asset.models.length) {
    const viewerButton = node("button", "button primary open-model", "View in 3D");
    viewerButton.type = "button";
    viewerButton.dataset.focusKey = "model";
    viewerButton.addEventListener("click", () => showModel(asset));
    const viewerAction = node("div", "model-card-action");
    viewerAction.append(viewerButton, node("span", "", `${asset.models.length} models / LODs · Drag to rotate`));
    body.append(viewerAction);
  }
  const position = stageIndex(asset.stage);
  const mini = node("div", "mini-stages");
  mini.setAttribute("aria-label", `Current phase: ${readable(asset.stage)}`);
  for (let index = 0; index < stageNames.length; index++) {
    let status = "";
    if (completeStates.has(state) || (position >= 0 && index < position) || (state === "approved" && index === position)) status = "complete";
    else if (index === position) status = attentionStates.has(state) ? "waiting" : "current";
    const mark = node("span", `mini-stage ${status}`);
    mark.title = stageLabels[index];
    mini.append(mark);
  }
  const meta = node("div", "card-meta");
  const attempt = asset.attempt ?? (Array.isArray(asset.attempts) ? asset.attempts.length : 0);
  meta.append(node("span", "stage-text", position >= 0 ? stageLabels[position] : readable(asset.stage)));
  meta.append(node("span", "", `Attempt ${attempt}`));
  body.append(mini, meta);
  const disclosure = node("details", "asset-details");
  const disclosureToggle = node("summary", "asset-details-toggle", "Details and files");
  disclosureToggle.dataset.focusKey = "details";
  disclosureToggle.setAttribute("aria-label", `Details and files for ${asset.name || asset.asset_id}`);
  const detailBody = node("div", "asset-details-body");
  disclosure.append(disclosureToggle, detailBody);
  if (asset.shape_repair) {
    const repair = node("details", "evidence repair-evidence");
    const summary = node("summary", "", "Repair diagnosis & acceptance");
    summary.dataset.focusKey = "repair-evidence";
    repair.append(summary, node("pre", "", stringify(asset.shape_repair)));
    detailBody.append(repair);
  }
  const profiles = renderProfiles(asset);
  if (profiles) detailBody.append(profiles);
  if (previews.length > 1) {
    const groups = new Map();
    for (const entry of previews) {
      if (!groups.has(entry.group)) groups.set(entry.group, []);
      groups.get(entry.group).push(entry);
    }
    const previewGroups = node("div", "preview-groups");
    for (const [groupName, entries] of groups) {
      const group = node("section", "preview-group");
      group.setAttribute("aria-label", `${groupName} previews`);
      if (groups.size > 1 || entries[0].profile) group.append(node("h4", "", groupName));
      const strip = node("div", "preview-strip");
      for (const entry of entries) {
        const button = node("button", "preview-thumb");
        button.type = "button";
        button.dataset.focusKey = `preview:${entry.key}`;
        button.setAttribute("aria-label", `${asset.name || asset.asset_id}: ${groupName}, ${entry.label} preview`);
        const img = node("img");
        img.src = entry.url;
        img.alt = "";
        img.loading = "lazy";
        img.addEventListener("error", () => {
          img.hidden = true;
          button.classList.add("image-failed");
          button.disabled = true;
          button.title = "This preview image is unavailable";
          button.setAttribute("aria-label", `${asset.name || asset.asset_id}: ${groupName}, ${entry.label} preview unavailable`);
          button.prepend(icon("image"));
        }, {once: true});
        button.append(img, node("span", "", entry.label));
        button.addEventListener("click", () => showPreview(asset, entry));
        strip.append(button);
      }
      group.append(strip);
      previewGroups.append(group);
    }
    detailBody.append(previewGroups);
  }
  if (asset.error) {
    const errorText = typeof asset.error === "string" ? asset.error : asset.error.message || asset.error.reason || readable(asset.error.code || "Asset needs attention");
    const errorSummary = node("div", "asset-error", errorText);
    errorSummary.title = "Open Details and files for the complete error report";
    body.append(errorSummary);
    const diagnostic = node("details", "error-details evidence");
    const diagnosticToggle = node("summary", "", "Error details");
    diagnosticToggle.dataset.focusKey = "error";
    diagnostic.append(diagnosticToggle, node("pre", "", stringify(asset.error)));
    detailBody.append(diagnostic);
  }
  const evidence = asset.evidence || asset.review || asset.attempts || asset.history;
  if (evidence && (typeof evidence !== "object" || Object.keys(evidence).length)) {
    const details = node("details", "evidence");
    const summary = node("summary", "", "Review evidence & attempts");
    summary.dataset.focusKey = "evidence";
    details.append(summary, node("pre", "", stringify(evidence)));
    detailBody.append(details);
  }
  const footer = node("div", "card-footer");
  for (const file of (asset.artifacts || [])) {
    const url = assetFileURL(batch, asset, file);
    if (!url) continue;
    const name = typeof file === "string" ? file.split("/").pop() : (file.name || file.relative_path?.split("/").pop() || "Artifact");
    const link = node("a", "artifact-link");
    link.append(node("span", "", name), icon("download"));
    link.href = url;
    link.dataset.focusKey = `artifact:${url}`;
    link.setAttribute("download", "");
    footer.append(link);
  }
  if (!footer.children.length) footer.append(node("span", "no-artifacts", "Exports appear when available"));
  detailBody.append(footer);
  card.append(visual, body, disclosure);
  return card;
}

function renderAssets(batch) {
  const focused = document.activeElement;
  const target = focusTarget(focused);
  const filter = $("asset-filter").value;
  const allAssets = batch.assets || [];
  const assets = allAssets.filter((asset) => filter === "all" || (filter === "attention" ? attentionStates.has(asset.state) : completeStates.has(asset.state)));
  const openDetails = new Set([...$("assets").querySelectorAll("details[open]")].map((detail) => `${detail.closest("article").dataset.assetId}:${detail.className}`));
  const cards = assets.map((asset) => renderAsset(batch, asset));
  for (const card of cards) {
    for (const detail of card.querySelectorAll("details")) {
      if (openDetails.has(`${card.dataset.assetId}:${detail.className}`)) detail.open = true;
    }
  }
  $("assets").replaceChildren(...cards);
  $("filtered-empty").hidden = assets.length > 0;
  $("filtered-empty").textContent = !allAssets.length ? "This batch has no assets yet. New assets will appear here when they are added." : filter === "attention" ? "No assets currently need attention." : "No finished assets yet. Accepted exports will appear here.";
  $("asset-list-count").textContent = String(assets.length);
  $("asset-list-description").textContent = filter === "all" ? "Open an asset to inspect its progress and evidence." : `Showing ${assets.length} of ${allAssets.length} ${allAssets.length === 1 ? "asset" : "assets"}${filter === "attention" ? " that need attention" : " that are finished"}.`;
  if (focused && !focused.isConnected && !document.querySelector("dialog[open]")) restoreFocus(target);
}

function render() {
  const focused = document.activeElement;
  const target = focusTarget(focused);
  const batches = Array.isArray(snapshot.batches) ? snapshot.batches : [];
  if (!batches.some((batch) => batch.batch_id === selectedBatchId)) selectedBatchId = snapshot.active_batch_id || batches[0]?.batch_id || null;
  if (!batches.some((batch) => batch.batch_id === selectedBatchId)) selectedBatchId = batches[0]?.batch_id || null;
  $("batch-total").textContent = String(batches.length);
  renderBatches(batches);
  const batch = selectedBatch();
  $("empty").hidden = Boolean(batch);
  $("workspace").hidden = !batch;
  if (!batch) {
    if (focused && !focused.isConnected && !document.querySelector("dialog[open]")) restoreFocus(target);
    return;
  }
  savePreferences();
  const assets = batch.assets || [];
  $("batch-name").textContent = batch.name || "Untitled batch";
  $("batch-subtitle").textContent = `${batch.batch_id} · ${assets.length} ${assets.length === 1 ? "asset" : "assets"} · Sequential production`;
  $("batch-status").className = `badge ${tone(batch.state)}`;
  $("batch-status").textContent = stateLabel(batch.state);
  const terminal = completeStates.has(batch.state) || batch.state === "cancelled";
  const paused = ["paused", "interrupted"].includes(batch.state);
  $("pause").hidden = paused || terminal;
  $("resume").hidden = !paused || terminal;
  $("cancel").hidden = terminal;
  for (const [id, label, pendingLabel] of [["pause", "Pause queue", "Pausing…"], ["resume", "Resume queue", "Resuming…"], ["cancel", "Cancel", "Cancelling…"]]) {
    const button = $(id);
    const pending = commandPending && pendingAction === id;
    button.disabled = commandPending || (id === "pause" && batch.state === "pause_requested");
    button.setAttribute("aria-busy", String(pending));
    controlLabel(button, pending ? pendingLabel : label);
  }
  $("asset-count").textContent = String(assets.length);
  $("finished-count").textContent = String(assets.filter((asset) => completeStates.has(asset.state)).length);
  $("attention-count").textContent = String(assets.filter((asset) => attentionStates.has(asset.state)).length);
  const activeResource = snapshot.active_resource || batch.active_resource || "Idle";
  $("active-resource").textContent = activeResource === "idle" ? "Idle" : activeResource;
  const activeId = snapshot.active_asset_id || batch.active_asset_id;
  const active = batches.flatMap((item) => item.assets || []).find((asset) => asset.asset_id === activeId);
  $("active-asset").textContent = active?.name || activeId || "Waiting for eligible work";
  buildStages(batch);
  updateGate(batch);
  renderAssets(batch);
  const stamp = batch.updated_at || batch.created_at;
  const date = stamp ? new Date(stamp) : null;
  $("updated-at").textContent = date && !Number.isNaN(date.getTime()) ? `Last change ${date.toLocaleString()}` : "Watching for changes";
  if (focused && !focused.isConnected && !document.querySelector("dialog[open]")) restoreFocus(target);
}

function notice(message, retry = false, source = "") {
  const content = $("notice-message") || $("notice");
  if (content.textContent !== (message || "")) content.textContent = message || "";
  $("notice").hidden = !message;
  noticeSource = source;
  if ($("retry")) {
    $("retry").hidden = !message || !retry;
    $("retry").textContent = sessionExpired ? "Reload dashboard" : "Retry connection";
  }
}

async function refresh(force = false) {
  if (busy) {
    // A command needs a new snapshot taken after its POST, not an older poll.
    if (force) { await activeRefresh; return refresh(true); }
    return;
  }
  if ((document.hidden && !force) || (sessionExpired && !force)) return;
  busy = true;
  let finishRefresh;
  activeRefresh = new Promise((resolve) => { finishRefresh = resolve; });
  if ($("retry")) $("retry").disabled = true;
  if ($("loading") && !hasSnapshot && (!loadAttempted || force)) $("loading").hidden = false;
  loadAttempted = true;
  try {
    const response = await fetch("/api/queue", {headers: {"X-Queue-Token": token}, cache: "no-store", signal: AbortSignal.timeout(8000)});
    if (!response.ok) {
      sessionExpired = response.status === 403;
      throw new Error(sessionExpired ? "This dashboard session expired. Reload the page to reconnect." : "The queue is temporarily unavailable. Your local worker keeps its saved progress.");
    }
    const next = await response.json();
    if (!Array.isArray(next.batches)) throw new Error("The queue returned an invalid snapshot.");
    const signature = JSON.stringify(next);
    snapshot = next;
    hasSnapshot = true;
    sessionExpired = false;
    if (force || signature !== currentSignature) { currentSignature = signature; render(); }
    $("connection").classList.remove("offline");
    if ($("connection-text").textContent !== "Live queue") $("connection-text").textContent = "Live queue";
    if (noticeSource === "connection") notice("");
  } catch (error) {
    $("connection").classList.add("offline");
    const connectionText = sessionExpired ? "Session expired" : "Reconnecting…";
    if ($("connection-text").textContent !== connectionText) $("connection-text").textContent = connectionText;
    const message = ["TimeoutError", "AbortError", "TypeError"].includes(error.name) ? "Cannot reach the local queue. Check that the worker is running, then try again." : error.message;
    notice(message || "Connection interrupted. Your work continues in the local worker.", true, "connection");
  } finally {
    busy = false;
    if ($("loading")) $("loading").hidden = true;
    if ($("retry")) $("retry").disabled = false;
    finishRefresh();
  }
}

async function control(action, batchId = selectedBatchId) {
  const batch = snapshot.batches.find((item) => item.batch_id === batchId);
  if (!batch || commandPending) return;
  const origin = document.activeElement;
  const focusWasOnControl = origin === $(action);
  commandPending = true;
  pendingAction = action;
  render();
  try {
    const response = await fetch(`/api/batches/${encodeURIComponent(batch.batch_id)}/${action}`, {
      method: "POST", headers: {"Content-Type": "application/json", "X-Queue-Token": token},
      body: "{}", signal: AbortSignal.timeout(10000)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The command could not be applied.");
    notice("");
    await refresh(true);
  } catch (error) { notice(error.message || "The command could not be applied."); }
  finally {
    commandPending = false;
    pendingAction = "";
    render();
    if (focusWasOnControl && (document.activeElement === document.body || document.activeElement === origin)) {
      const next = action === "pause" && !$("resume").hidden ? $("resume") : action === "resume" && !$("pause").hidden ? $("pause") : origin;
      if (!next.hidden && !next.disabled) next.focus({preventScroll: true});
    }
  }
}

$("pause").addEventListener("click", () => control("pause"));
$("resume").addEventListener("click", () => control("resume"));
$("cancel").addEventListener("click", () => {
  cancelBatchId = selectedBatchId;
  dialogReturnTargets.set("cancel-dialog", focusTarget());
  $("cancel-dialog").showModal();
});
$("cancel-back").addEventListener("click", () => $("cancel-dialog").close());
$("cancel-confirm").addEventListener("click", () => { $("cancel-dialog").close(); control("cancel", cancelBatchId); });
$("preview-close").addEventListener("click", () => $("preview-dialog").close());
if (["all", "attention", "finished"].includes(preferences.filter)) $("asset-filter").value = preferences.filter;
function setLayout(list) {
  $("assets").classList.toggle("list-layout", list);
  $("grid-view").setAttribute("aria-pressed", String(!list));
  $("list-view").setAttribute("aria-pressed", String(list));
}
setLayout(preferences.layout !== "grid");
$("asset-filter").addEventListener("change", () => { const batch = selectedBatch(); if (batch) renderAssets(batch); savePreferences(); });
for (const [id, list] of [["grid-view", false], ["list-view", true]]) {
  $(id).addEventListener("click", () => {
    setLayout(list);
    savePreferences();
  });
}
for (const id of ["preview-dialog", "model-dialog", "cancel-dialog"]) {
  $(id).addEventListener("close", () => {
    const target = dialogReturnTargets.get(id);
    dialogReturnTargets.delete(id);
    if (id === "cancel-dialog") cancelBatchId = null;
    if (!document.querySelector("dialog[open]")) restoreFocus(target, true);
  });
}
$("retry")?.addEventListener("click", () => sessionExpired ? location.reload() : refresh(true));
$("sidebar-toggle")?.addEventListener("click", () => setSidebarOpen(!document.body.classList.contains("sidebar-open"), document.body.classList.contains("sidebar-open")));
$("sidebar-backdrop")?.addEventListener("click", () => setSidebarOpen(false, true));
$("sidebar-close")?.addEventListener("click", () => setSidebarOpen(false, true));
document.addEventListener("keydown", (event) => {
  if (!document.body.classList.contains("sidebar-open") || document.querySelector("dialog[open]")) return;
  if (event.key === "Escape") {
    event.preventDefault();
    setSidebarOpen(false, true);
  } else if (event.key === "Tab") {
    const focusable = sidebarFocusables();
    const current = focusable.indexOf(document.activeElement);
    const next = current < 0 ? (event.shiftKey ? focusable.length - 1 : 0) : (current + (event.shiftKey ? -1 : 1) + focusable.length) % focusable.length;
    event.preventDefault();
    focusable[next]?.focus({preventScroll: true});
  }
});
document.addEventListener("focusin", (event) => {
  if (document.body.classList.contains("sidebar-open") && !document.querySelector("dialog[open]") && !$("sidebar").contains(event.target)) sidebarFocusables()[0]?.focus({preventScroll: true});
});
mobileSidebar.addEventListener("change", () => {
  // A queued viewport event must not close navigation the user just opened.
  setSidebarOpen(mobileSidebar.matches && document.body.classList.contains("sidebar-open"));
});
setSidebarOpen(false);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(true); });
refresh(true);
setInterval(() => refresh(), 2500);
