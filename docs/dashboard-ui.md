# Asset workspace UI

The dashboard follows the Codex desktop dark theme. Core color, typography,
radius, spacing and interaction values were checked against the installed
Codex 26.908.4834 app. The purple sidebar tint follows its observed Windows
appearance; operating-system translucency is approximated with an opaque fill.

`style.css` owns shared tokens and dashboard layout. `viewer.css` extends those
tokens for the inspector; the renderer reads `--viewer-bg`. Fonts, SVG icons,
styles, scripts, previews and models stay local. No external font or icon CDN
is required.

The default asset layout is a compact list. Grid mode exposes larger previews.
Details and files expand inline, preserving both disclosure state and keyboard
focus during queue updates. Batch selection, filter and layout are remembered
when local storage is available. Queue controls keep their existing APIs.

The mobile navigation drawer traps focus, makes the background inert, and closes
using its close button, Escape or the backdrop. Native dialogs expose accessible
names and return focus to refreshed asset controls. Initial loading, connection
retry, session expiry, empty results and missing preview images have explicit
states. Motion respects the operating-system reduced-motion setting.

## Verification

Run the existing HTTP and static-resource suite:

```powershell
python -m pytest tests/test_hunyuan_mv_dashboard.py -q
```

The isolated browser suite serves the real dashboard against a synthetic manager.
It creates tiny preview and embedded GLB fixtures and never runs generation jobs:

```powershell
python scripts/dashboard-ui-fixture.py --test --node node
```

Install Playwright for the selected Node runtime and provide a Chrome executable
if it is not at the standard Windows location. `PLAYWRIGHT_MODULE` and
`CHROME_EXECUTABLE` override those paths. The Python fixture needs Pillow.
Reports and screenshots are written to `.tools/dashboard-ui/`.

For manual visual review:

```powershell
python scripts/dashboard-ui-fixture.py --serve
```

Browser coverage includes 320, 390, 768 and 1440 pixel layouts, list/grid controls,
disclosures and focus across changed snapshots, mobile navigation, preferences,
loading/empty/offline states, missing images, pause/resume/cancel, and real GLB
loading, model selection, wireframe, camera fit and viewer cleanup.
