"use strict";

// Run through scripts/dashboard-ui-fixture.py --test, or pass a fixture URL here.
// No worker or model-generation API is called. Chromium uses software WebGL.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const bundled = path.join(process.env.USERPROFILE || "", ".cache", "codex-runtimes",
  "codex-primary-runtime", "dependencies", "node", "node_modules", "playwright");
let playwright;
try { playwright = require(process.env.PLAYWRIGHT_MODULE || "playwright"); }
catch { playwright = require(bundled); }
const url = process.argv[2];
if (!url) throw new Error("Pass the URL of scripts/dashboard-ui-fixture.py --serve");
const origin = new URL(url).origin;
const output = path.resolve(process.argv[3] || path.join(__dirname, "..", ".tools", "dashboard-ui"));
fs.mkdirSync(output, {recursive: true});
const captureOnly = process.argv.includes("--capture");
const widths = [{width: 1440, height: 1080}, {width: 768, height: 1024},
  {width: 390, height: 844}, {width: 320, height: 740}];

async function noOverflow(page, label) {
  const result = await page.evaluate(() => {
    const width = document.documentElement.clientWidth;
    return {width, scrollWidth: document.documentElement.scrollWidth,
      offenders: [...document.querySelectorAll("body *")].filter((element) => {
        const box = element.getBoundingClientRect();
        return box.width && (box.right > width + 1 || box.left < -1)
          && getComputedStyle(element).position !== "fixed"
          && !element.closest("[hidden], dialog:not([open])");
      }).slice(0, 12).map((element) => `${element.tagName}.${element.className}`)};
  });
  assert.ok(result.scrollWidth <= result.width + 1,
    `${label} has horizontal overflow: ${JSON.stringify(result)}`);
  const openDialog = page.locator("dialog[open]");
  if (await openDialog.count()) {
    const sizes = await openDialog.evaluate((element) => ({scroll: element.scrollWidth, width: element.clientWidth}));
    assert.ok(sizes.scroll <= sizes.width + 1, `${label} dialog has horizontal overflow: ${JSON.stringify(sizes)}`);
  }
}

async function waitForChangedSnapshot(page) {
  const before = await page.locator("#updated-at").textContent();
  await page.waitForFunction((previous) => document.querySelector("#updated-at").textContent !== previous, before);
}

async function focused(locator, description) {
  // Native dialog close events are queued after Escape returns to Playwright.
  // Poll briefly so we assert the settled focus, not that browser event timing.
  for (let attempt = 0; attempt < 10; attempt++) {
    if (await locator.evaluate((element) => element === document.activeElement)) return;
    await locator.page().waitForTimeout(50);
  }
  const active = await locator.page().evaluate(() => document.activeElement.outerHTML.slice(0, 400));
  assert.fail(`${description}; active element: ${active}`);
}

async function assertDialogName(page, id, name) {
  assert.equal(await page.getByRole("dialog", {name}).getAttribute("id"), id,
    `${id} needs a useful accessible name`);
}

function jsonGLB(document) {
  const raw = Buffer.from(JSON.stringify(document));
  const json = Buffer.concat([raw, Buffer.alloc((4 - raw.length % 4) % 4, 0x20)]);
  const header = Buffer.alloc(20);
  [0x46546c67, 2, 20 + json.length, json.length, 0x4e4f534a]
    .forEach((value, index) => header.writeUInt32LE(value, index * 4));
  return Buffer.concat([header, json]);
}

(async () => {
  const browser = await playwright.chromium.launch({
    executablePath: process.env.CHROME_EXECUTABLE
      || path.join(process.env.ProgramFiles || "C:/Program Files", "Google/Chrome/Application/chrome.exe"),
    headless: true,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const context = await browser.newContext({viewport: widths[0], deviceScaleFactor: 1});
  const errors = [];
  const externalRequests = [];
  context.on("page", (page) => page.on("pageerror", (error) => errors.push(error.message)));
  await context.route("**/*", (route) => {
    if (new URL(route.request().url()).origin === origin) return route.continue();
    externalRequests.push(route.request().url());
    return route.abort();
  });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const checks = [];
  const screenshots = [];
  const shot = async (name, target = page, fullPageOverride) => {
    const file = path.join(output, `${name}.png`);
    const hasDialog = await target.locator("dialog[open], [role='dialog'][aria-modal='true']").count() > 0;
    // Capture fixed headers from the top after responsive drawer transitions settle.
    // Dialog backdrops only cover the viewport, so do not composite them full-page.
    await target.evaluate(() => {
      window.scrollTo(0, 0);
      const dialog = document.querySelector("dialog[open], [role='dialog'][aria-modal='true']");
      if (dialog) dialog.scrollTop = 0;
    });
    await target.waitForTimeout(240);
    await target.screenshot({path: file, fullPage: fullPageOverride ?? !hasDialog, animations: "disabled"});
    screenshots.push(file);
  };
  try {
    await page.goto(url, {waitUntil: "networkidle"});
    assert.equal(await page.title(), "Cognito-3D-mcp · Asset queue");
    assert.equal(await page.getByRole("link", {name: "Cognito-3D-mcp queue home"}).count(), 1);
    await page.locator("#workspace:not([hidden])").waitFor();
    assert.equal(await page.locator(".asset-card").count(), 3);
    assert.equal(await page.locator("#connection-text").textContent(), "Live queue");
    assert.equal(await page.locator("#attention-count").textContent(), "2");
    assert.equal(await page.locator("#finished-count").textContent(), "0");
    assert.equal(await page.locator("#stages .stage").count(), 6);
    assert.equal(await page.locator("#stages .stage-name").nth(2).textContent(), "Repair & review");
    assert.equal(await page.locator(".stage.waiting .stage-detail").textContent(), "1/3 accepted");
    assert.equal(await page.locator("[data-asset-id='vessel'] .badge").textContent(), "Phase accepted");
    assert.equal(await page.locator(".asset-visual > img").evaluateAll((images) =>
      images.every((image) => image.complete && image.naturalWidth > 0)), true);
    assert.equal(await page.locator("#list-view").getAttribute("aria-pressed"), "true",
      "The initial workspace should use the compact list layout");
    await shot("dashboard-hero-1440", page, false);
    await page.locator("#grid-view").click();
    for (const viewport of widths) {
      await page.setViewportSize(viewport);
      await noOverflow(page, `grid at ${viewport.width}px`);
      await shot(`dashboard-${viewport.width}`);
    }
    checks.push("responsive grid at 1440, 768, 390, and 320px");
    if (captureOnly) {
      console.log(JSON.stringify({passed: true, captureOnly, screenshots}, null, 2));
      return;
    }

    await page.locator("#list-view").click();
    assert.equal(await page.locator("#list-view").getAttribute("aria-pressed"), "true");
    for (const viewport of widths) {
      await page.setViewportSize(viewport);
      await noOverflow(page, `list at ${viewport.width}px`);
      await shot(`dashboard-list-${viewport.width}`);
    }
    await page.locator("#asset-filter").selectOption("attention");
    assert.equal(await page.locator(".asset-card").count(), 2);
    assert.equal(await page.locator("#asset-list-count").textContent(), "2");
    await page.reload({waitUntil: "networkidle"});
    assert.equal(await page.locator("#asset-filter").inputValue(), "attention");
    assert.equal(await page.locator("#list-view").getAttribute("aria-pressed"), "true");
    assert.equal(await page.locator(".asset-card").count(), 2);
    await page.locator("#asset-filter").selectOption("finished");
    assert.equal(await page.locator(".asset-card").count(), 0);
    assert.equal(await page.locator("#filtered-empty").isVisible(), true);
    await shot("dashboard-filter-empty");
    await page.locator("#asset-filter").selectOption("all");
    await page.locator("#grid-view").click();
    checks.push("list layout, attention/finished filters, shown count, and saved preferences");

    await page.setViewportSize(widths[2]);
    await page.locator("#sidebar-toggle").click();
    assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "true");
    assert.equal(await page.locator("#sidebar").getAttribute("aria-modal"), "true");
    assert.equal(await page.locator("#main-content").evaluate((element) => element.inert), true);
    for (let tab = 0; tab < 8; tab++) {
      await page.keyboard.press(tab < 4 ? "Tab" : "Shift+Tab");
      if (tab === 0) await waitForChangedSnapshot(page);
      const focus = await page.evaluate(() => ({
        contained: document.querySelector("#sidebar").contains(document.activeElement),
        active: document.activeElement.outerHTML.slice(0, 500),
        sidebarOpen: document.body.classList.contains("sidebar-open"),
        sidebarInert: document.querySelector("#sidebar").inert,
        mainInert: document.querySelector("#main-content").inert,
        updated: document.querySelector("#updated-at").textContent,
      }));
      assert.equal(focus.contained, true,
        `Keyboard focus escaped the mobile navigation drawer at key ${tab}: ${JSON.stringify(focus)}`);
    }
    await shot("dashboard-navigation-mobile");
    await page.locator("[data-batch-id='completed-fixture']").click();
    assert.equal(await page.locator("#batch-name").textContent(), "Coastal props");
    assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "false");
    await focused(page.locator("#sidebar-toggle"), "Selecting a mobile batch loses navigation focus");
    await page.locator("#sidebar-toggle").click();
    await page.locator("[data-batch-id='ui-fixture']").click();
    await page.locator("#sidebar-toggle").click();
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "false");
    await focused(page.locator("#sidebar-toggle"), "Dismissing mobile navigation loses toggle focus");
    await page.locator("#sidebar-toggle").click();
    await page.locator("#sidebar-close").click();
    await focused(page.locator("#sidebar-toggle"), "Closing mobile navigation loses toggle focus");
    checks.push("mobile navigation focus trap, batch selection, close, and Escape focus return");

    await page.setViewportSize(widths[0]);
    await page.locator("#list-view").click();
    const detailToggle = page.locator("[data-asset-id='tower'] details.asset-details > summary");
    await detailToggle.click();
    await detailToggle.focus();
    await waitForChangedSnapshot(page);
    await focused(detailToggle, "Asset details toggle loses keyboard focus after queue refresh");
    assert.equal(await page.locator("[data-asset-id='tower'] details.asset-details").getAttribute("open"), "");
    await shot("dashboard-list-expanded-1440");
    for (const viewport of widths.slice(1)) {
      await page.setViewportSize(viewport);
      await noOverflow(page, `expanded list details at ${viewport.width}px`);
      if (viewport.width === 390) await shot("dashboard-list-expanded-390");
    }
    await page.setViewportSize(widths[0]);
    const summary = page.locator("[data-asset-id='tower'] details.evidence summary");
    await summary.click();
    await summary.focus();
    await waitForChangedSnapshot(page);
    await focused(summary, "Evidence summary loses keyboard focus after queue refresh");
    assert.equal(await page.locator("[data-asset-id='tower'] details.evidence").getAttribute("open"), "");
    assert.equal(await page.locator("[data-asset-id='tower'] details.asset-details").getAttribute("open"), "");
    assert.match(await page.locator("[data-asset-id='tower'] details.evidence pre").textContent(), /rear pillar/);
    await summary.click();

    const thumb = page.locator("[data-asset-id='tower'] .preview-thumb").first();
    await thumb.focus();
    await page.keyboard.press("Enter");
    await page.locator("#preview-dialog[open]").waitFor();
    await assertDialogName(page, "preview-dialog", "Sanctuary watchtower · Front");
    await waitForChangedSnapshot(page);
    await page.setViewportSize(widths[2]);
    await noOverflow(page, "preview at 390px");
    await shot("dashboard-preview-mobile");
    await page.keyboard.press("Escape");
    await focused(thumb, "Preview close does not restore focus to the refreshed thumbnail");
    checks.push("evidence and preview focus survive changed snapshots");

    await page.setViewportSize(widths[0]);
    const modelTrigger = page.locator("[data-asset-id='tower'] .open-model");
    await modelTrigger.focus();
    await page.keyboard.press("Enter");
    await page.locator("#model-dialog[open]").waitFor();
    await assertDialogName(page, "model-dialog", "Sanctuary watchtower");
    await page.waitForFunction(() => document.querySelector("#model-viewport").dataset.state === "ready");
    assert.equal(await page.locator("#model-triangles").textContent(), "12");
    assert.equal(await page.locator("#model-viewport canvas").count(), 1);
    await page.locator("#model-select").selectOption("mobile");
    await page.waitForFunction(() => document.querySelector("#model-viewport").dataset.modelId === "mobile"
      && document.querySelector("#model-viewport").dataset.state === "ready");
    assert.equal(await page.locator("#model-profile-name").textContent(), "Mobile");
    await page.locator("#model-wireframe").check();
    const renderCount = await page.locator("#model-viewport").getAttribute("data-render-count");
    await page.locator("#model-reset").click();
    await page.waitForFunction((count) => Number(document.querySelector("#model-viewport").dataset.renderCount) > Number(count), renderCount);
    await shot("dashboard-model-desktop");
    for (const viewport of widths.slice(1)) {
      await page.setViewportSize(viewport);
      await page.locator("#model-reset").click();
      await noOverflow(page, `model inspector at ${viewport.width}px`);
    }
    await shot("dashboard-model-mobile");
    await waitForChangedSnapshot(page);
    await page.keyboard.press("Escape");
    await focused(modelTrigger, "Model close does not restore focus to the refreshed trigger");
    assert.equal(await page.locator("#model-viewport canvas").count(), 0);
    checks.push("real GLB loading, profile selection, wireframe, reset, cleanup, and focus return");

    await page.setViewportSize(widths[2]);
    await page.locator("#pause").click();
    await page.waitForFunction(() => document.querySelector("#batch-status").textContent === "Paused");
    assert.equal(await page.locator("#gate-title").textContent(), "Queue paused");
    await focused(page.locator("#resume"), "Pause should move focus to Resume queue");
    await page.locator("#resume").click();
    await page.waitForFunction(() => document.querySelector("#batch-status").textContent === "Agent review");
    await focused(page.locator("#pause"), "Resume should move focus to Pause queue");
    await page.locator("#cancel").click();
    await assertDialogName(page, "cancel-dialog", "Stop this batch?");
    await focused(page.locator("#cancel-back"), "Cancel dialog should initially focus Keep working");
    await noOverflow(page, "cancel dialog at 390px");
    await shot("dashboard-cancel-mobile");
    await page.keyboard.press("Escape");
    await focused(page.locator("#cancel"), "Cancel dismissal loses its trigger focus");
    assert.equal(await page.locator("#batch-status").textContent(), "Agent review");

    await page.route("**/api/queue", (route) => route.abort("failed"));
    await page.waitForFunction(() => document.querySelector("#connection").classList.contains("offline"));
    assert.equal(await page.locator("#workspace").isVisible(), true, "An outage should retain the last usable snapshot");
    assert.equal(await page.locator("#notice").isVisible(), true);
    await shot("dashboard-offline");
    await page.unroute("**/api/queue");
    const retry = page.getByRole("button", {name: /retry|reconnect/i});
    if (await retry.isVisible()) await retry.click();
    await page.waitForFunction(() => document.querySelector("#connection-text").textContent === "Live queue");
    checks.push("pause, resume, safe cancel dismissal, and offline recovery");

    const states = await context.newPage();
    let queueCaptured;
    const capturedQueue = new Promise((resolve) => { queueCaptured = resolve; });
    await states.route("**/api/queue", (route) => queueCaptured(route));
    await states.goto(url, {waitUntil: "domcontentloaded"});
    await states.waitForFunction(() => document.querySelector("#connection-text").textContent.includes("Connecting"));
    assert.equal(await states.locator("#workspace").isVisible(), false);
    const loading = states.locator("#loading, #loading-state");
    assert.equal(await loading.isVisible(), true, "The initial request needs a visible loading state");
    await shot("dashboard-loading", states);
    await (await capturedQueue).fulfill({json: {batches: []}});
    await states.locator("#empty:not([hidden])").waitFor();
    assert.equal(await loading.isVisible(), false);
    await shot("dashboard-empty", states);
    await states.close();
    checks.push("explicit initial loading and empty queue");

    const denied = await context.newPage();
    await denied.addInitScript(() => {
      Object.defineProperty(window, "localStorage", {get() { throw new DOMException("Storage denied", "SecurityError"); }});
    });
    await denied.goto(url, {waitUntil: "networkidle"});
    await denied.locator("#workspace:not([hidden])").waitFor();
    await denied.locator("#list-view").click();
    await denied.locator("#asset-filter").selectOption("attention");
    assert.equal(await denied.locator(".asset-card").count(), 2);
    await denied.close();
    checks.push("storage unavailable remains usable");

    const broken = await context.newPage();
    await broken.route("**/previews/front.png", (route) => route.abort("failed"));
    await broken.goto(url, {waitUntil: "networkidle"});
    await broken.locator("[data-asset-id='tower'] .preview-unavailable").waitFor();
    await broken.locator("[data-asset-id='vessel'] .preview-unavailable").waitFor();
    assert.equal(await broken.locator("[data-asset-id='tower'] .asset-visual").getAttribute("role"), "button",
      "A missing preview must leave the valid 3D viewer available");
    assert.equal(await broken.locator("[data-asset-id='vessel'] .asset-visual").getAttribute("role"), null,
      "A missing image without a 3D model must not remain a broken button");
    await broken.setViewportSize(widths[2]);
    await noOverflow(broken, "missing previews at 390px");
    await shot("dashboard-preview-unavailable", broken);
    await broken.close();
    checks.push("missing preview has a usable fallback");

    const repairView = await context.newPage();
    await repairView.route("**/api/queue", async (route) => {
      const response = await route.fetch();
      const queue = await response.json();
      queue.batches[0].stage = "shape_repair";
      const asset = queue.batches[0].assets[0];
      asset.stage = "shape_repair";
      asset.shape_repair = {diagnosis: "Close the accidental underside opening", current_attempt: 2};
      const model = asset.models[0];
      asset.models = [
        {...model, id: "full_game", textured: true, stale: true},
        {...model, id: "shape", label: "Original Hunyuan", quad_count: null},
        {...model, id: "accepted_master", label: "Accepted master", quad_count: null},
        {...model, id: "repair_attempt_2", label: "Repair attempt 2", stage: "shape_repair",
          approved: false, current: true, quad_count: null},
      ];
      await route.fulfill({json: queue});
    });
    await repairView.goto(url, {waitUntil: "networkidle"});
    await repairView.locator("#workspace:not([hidden])").waitFor();
    assert.equal(await repairView.locator(".stage.waiting .stage-name").textContent(), "Repair & review");
    await repairView.locator("[data-asset-id='tower'] .asset-details-toggle").click();
    await repairView.locator("[data-asset-id='tower'] .repair-evidence summary").click();
    assert.match(await repairView.locator("[data-asset-id='tower'] .repair-evidence pre").textContent(), /underside opening/);
    await repairView.locator("[data-asset-id='tower'] .open-model").click();
    await repairView.waitForFunction(() => document.querySelector("#model-viewport").dataset.state === "ready");
    assert.equal(await repairView.locator("#model-select").inputValue(), "repair_attempt_2",
      "The current repair candidate should open before stale textured outputs");
    assert.match(await repairView.locator("#model-stage-note").textContent(), /Repair candidate/);
    assert.match(await repairView.locator("#model-select option").first().textContent(), /Previous version/);
    await repairView.locator("#model-select").selectOption("accepted_master");
    await repairView.waitForFunction(() => document.querySelector("#model-viewport").dataset.modelId === "accepted_master");
    assert.match(await repairView.locator("#model-stage-note").textContent(), /All three remesh profiles/);
    await shot("dashboard-repair-review", repairView);
    await repairView.locator("#model-select").selectOption("full_game");
    assert.match(await repairView.locator("#model-stage-note").textContent(), /Previous attempt/);
    await repairView.close();
    checks.push("repair stage, diagnosis evidence, current candidate selection, accepted master, and previous-version labels");

    const hostile = await context.newPage();
    const hostileText = '<img src=x onerror="window.dashboardXss = true">';
    let modelDocument = {asset: {version: "2.0"}, images: [{uri: "https://attacker.example/texture.png"}]};
    await hostile.route("**/api/queue", async (route) => {
      const response = await route.fetch();
      const queue = await response.json();
      const asset = queue.batches[0].assets[0];
      asset.name = hostileText;
      asset.evidence = {review: hostileText};
      asset.previews = {
        front: "/artifacts/ui-fixture/tower/previews/front.png",
        traversal: "/artifacts/ui-fixture/tower/../tower/previews/front.png",
        external: "https://attacker.example/preview.png",
        query: "/artifacts/ui-fixture/tower/previews/front.png?unsafe=true",
      };
      asset.artifacts = [null, {url: "javascript:window.dashboardXss=true"},
        {url: "/artifacts/ui-fixture/tower/../tower/game.glb"}];
      asset.models = [
        {...asset.models[0], label: hostileText},
        {id: "traversal", label: "Traversal", url: "/artifacts/ui-fixture/tower/../tower/game.glb"},
        {id: "alias", label: "Arbitrary alias", url: "/models/ui-fixture/tower/other.glb"},
        {id: "remote", label: "Remote", url: "https://attacker.example/model.glb"},
      ];
      await route.fulfill({json: queue});
    });
    await hostile.route("**/game.glb", (route) => route.fulfill({
      contentType: "application/octet-stream", body: jsonGLB(modelDocument),
    }));
    await hostile.goto(url, {waitUntil: "networkidle"});
    await hostile.locator("#workspace:not([hidden])").waitFor();
    assert.equal(await hostile.locator("[data-asset-id='tower'] h3").textContent(), hostileText);
    assert.equal(await hostile.locator("[data-asset-id='tower'] .preview-thumb").count(), 0,
      "Unsafe preview URLs must be omitted");
    assert.equal(await hostile.locator("[data-asset-id='tower'] .artifact-link").count(), 0,
      "Unsafe download URLs must be omitted");
    await hostile.locator("[data-asset-id='tower'] .open-model").click();
    await hostile.waitForFunction(() => document.querySelector("#model-viewport").dataset.state === "error");
    assert.equal(await hostile.locator("#model-select option").count(), 1,
      "Only scoped canonical GLB URLs belong in the model selector");
    assert.match(await hostile.locator("#model-message").textContent(), /embedded in the GLB/);
    for (const uri of ["https://attacker.example/mesh.bin", "", null]) {
      await hostile.locator("#model-close").click();
      modelDocument = {asset: {version: "2.0"}, buffers: [{uri, byteLength: 0}]};
      await hostile.locator("[data-asset-id='tower'] .open-model").click();
      await hostile.waitForFunction(() => document.querySelector("#model-viewport").dataset.state === "error");
      assert.match(await hostile.locator("#model-message").textContent(), /embedded in the GLB/);
    }
    assert.equal(await hostile.evaluate(() => window.dashboardXss), undefined);
    await hostile.close();
    checks.push("untrusted metadata stays text, invalid asset paths are omitted, and external GLB resources are rejected");

    await page.locator("#cancel").click();
    await page.locator("#cancel-confirm").click();
    await page.waitForFunction(() => document.querySelector("#batch-status").textContent === "Cancelled");
    assert.equal(await page.locator("#gate-title").textContent(), "Batch cancelled");
    assert.equal(await page.locator("#cancel").isVisible(), false);
    checks.push("confirmed cancel preserves artifact cards");
    assert.deepEqual(errors, [], "Dashboard raised browser errors");
    assert.deepEqual(externalRequests, [], "Dashboard attempted external requests");
    const report = {passed: true, fixtureOnly: true, checks, screenshots};
    fs.writeFileSync(path.join(output, "report.json"), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    await shot("dashboard-failure").catch(() => {});
    console.error(JSON.stringify({passed: false, checks, errors, externalRequests}));
    throw error;
  } finally {
    await context.close();
    await browser.close();
  }
})().catch((error) => {console.error(error); process.exitCode = 1;});
