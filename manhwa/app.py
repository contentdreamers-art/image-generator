import os
import time
import traceback
import importlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import gradio as gr

STARTUP_LOG = []
_IMPORTED = {}


def _log(msg: str) -> str:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    STARTUP_LOG.append(line)
    if len(STARTUP_LOG) > 500:
        del STARTUP_LOG[:-500]
    try:
        with open('/tmp/manhwa_startup.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass
    return '\n'.join(STARTUP_LOG)


def _safe_import(module_name: str, attr_name: str = None):
    key = f"{module_name}:{attr_name or '*'}"
    if key in _IMPORTED:
        return _IMPORTED[key]
    _log(f"import start -> {key}")
    t0 = time.time()
    mod = importlib.import_module(module_name)
    result = getattr(mod, attr_name) if attr_name else mod
    _IMPORTED[key] = result
    _log(f"import ok -> {key} ({time.time() - t0:.2f}s)")
    return result


CSS = """
/* ===== GRADIO 6 CSS VARIABLE OVERRIDES ===== */
:root, .dark {
    --body-background-fill: #0e0e14 !important;
    --background-fill-primary: #17171f !important;
    --background-fill-secondary: #0d0d12 !important;
    --border-color-primary: #27273e !important;
    --border-color-accent: #f97316 !important;
    --color-accent: #f97316 !important;
    --color-accent-soft: rgba(249,115,22,0.12) !important;
    --button-primary-background-fill: linear-gradient(135deg, #f97316, #ea580c) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #fb923c, #f97316) !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: #1c1c2e !important;
    --button-secondary-background-fill-hover: #252540 !important;
    --button-secondary-border-color: #35355a !important;
    --button-secondary-text-color: #9090c0 !important;
    --input-background-fill: #0d0d12 !important;
    --input-border-color: #2a2a3e !important;
    --input-border-color-focus: #f97316 !important;
    --input-shadow-focus: 0 0 0 2px rgba(249,115,22,0.15) !important;
    --block-background-fill: #17171f !important;
    --block-border-color: #27273e !important;
    --block-label-background-fill: #1a1a28 !important;
    --block-label-text-color: #6060a0 !important;
    --panel-background-fill: #17171f !important;
    --panel-border-color: #27273e !important;
    --checkbox-background-color-selected: #f97316 !important;
    --color-accent-base: 249 115 22 !important;
    --body-text-color: #d8d8f0 !important;
    --body-text-color-subdued: #6868a0 !important;
    --block-label-text-size: 10.5px !important;
    --block-label-font-weight: 700 !important;
    --table-even-background-fill: #131320 !important;
    --table-odd-background-fill: #17171f !important;
    --table-row-focus: rgba(249,115,22,0.08) !important;
    --section-header-text-weight: 700 !important;
}

/* ===== FORCE BODY DARK ===== */
body, html { background: #0e0e14 !important; color: #d8d8f0 !important; }
.gradio-container { max-width: 1700px !important; background: #0e0e14 !important; }
footer { display: none !important; }

/* ===== STICKY TABS ===== */
.tabs > div:first-child {
    position: sticky !important;
    top: 0 !important;
    z-index: 500 !important;
    background: #0e0e14 !important;
    border-bottom: 2px solid #22223a !important;
    padding-bottom: 0 !important;
}
.tabs button[role=tab] {
    padding: 10px 28px !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    color: #4a4a72 !important;
    background: transparent !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.tabs button[role=tab][aria-selected=true] {
    color: #f97316 !important;
    border-bottom-color: #f97316 !important;
    background: transparent !important;
}
.tabs button[role=tab]:hover:not([aria-selected=true]) { color: #9090c0 !important; }

/* ===== GENERATION BAR ===== */
.gen-bar {
    background: linear-gradient(135deg, #191926, #1d1928) !important;
    border: 1px solid #3a2a50 !important;
    border-left: 4px solid #f97316 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
}

/* ===== BUTTONS EXTRA ===== */
button.primary { box-shadow: 0 2px 12px rgba(249,115,22,0.35) !important; font-weight: 700 !important; }
button.primary:hover { box-shadow: 0 4px 18px rgba(249,115,22,0.55) !important; transform: translateY(-1px) !important; }
button.stop { background: #1e0e0e !important; border-color: #662020 !important; color: #e08080 !important; }

/* ===== COST BOX ===== */
.cost-box textarea { color: #50e896 !important; font-family: 'Menlo', monospace !important; font-size: 11px !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0e0e14; }
::-webkit-scrollbar-thumb { background: #2e2e4e; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #f97316; }

/* ===== ACCORDION HEADERS ===== */
details > summary { background: #1a1a28 !important; border: 1px solid #27273e !important; border-radius: 8px !important; font-weight: 700 !important; color: #c0c0e0 !important; }
details[open] > summary { border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important; border-bottom-color: transparent !important; }

/* ===== MARKDOWN ACCENT ===== */
.prose h3 { color: #f97316 !important; }
input[type=checkbox]:checked { accent-color: #f97316 !important; }

/* ===== COMPACT TABLES (max ~5 rows + scrollable) ===== */
.compact-table .table-wrap { max-height: 220px !important; overflow-y: auto !important; }
.compact-table .table-wrap table { font-size: 12px !important; }

/* ===== BUILD MODE BIG TOGGLE ===== */
#build-mode-toggle { margin: 0 0 20px 0 !important; }
#build-mode-toggle fieldset,
#build-mode-toggle .form { border: none !important; padding: 0 !important; background: transparent !important; }
#build-mode-toggle .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 12px !important;
    padding: 4px 0 !important;
    background: transparent !important;
}
#build-mode-toggle .wrap label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 22px 24px !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: 2px solid #35355a !important;
    border-radius: 14px !important;
    cursor: pointer !important;
    background: #131320 !important;
    color: #4a4a72 !important;
    transition: all 0.15s ease !important;
    min-height: 68px !important;
    gap: 10px !important;
}
#build-mode-toggle input[type=radio] {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important; height: 0 !important;
}
#build-mode-toggle .wrap label:has(input[type=radio]:checked) {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    border-color: #f97316 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 24px rgba(249,115,22,0.45) !important;
}
#build-mode-toggle .wrap label:hover:not(:has(input[type=radio]:checked)) {
    border-color: #6060a0 !important;
    color: #c0c0e0 !important;
    background: #1c1c2e !important;
}
"""


def _render_tab_fallback(tab_title: str, module_name: str, err: Exception):
    with gr.Tab(tab_title):
        gr.Markdown(f"## {tab_title}")
        gr.Textbox(
            label=f"{module_name}.py error",
            value=f"{type(err).__name__}: {err}\n\n{traceback.format_exc()}",
            lines=20,
            interactive=False,
        )


MANHWA_THEME = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0e0e14",
    body_text_color="#d8d8f0",
    body_text_color_subdued="#6868a0",
    background_fill_primary="#17171f",
    background_fill_secondary="#0d0d12",
    border_color_primary="#27273e",
    border_color_accent="#f97316",
    block_background_fill="#17171f",
    block_border_color="#27273e",
    block_label_background_fill="#1a1a28",
    block_label_text_color="#6868a0",
    block_label_text_size="10.5px",
    block_label_text_weight="700",
    input_background_fill="#0d0d12",
    input_border_color="#2a2a3e",
    input_border_color_focus="#f97316",
    button_primary_background_fill="linear-gradient(135deg, #f97316, #ea580c)",
    button_primary_background_fill_hover="linear-gradient(135deg, #fb923c, #f97316)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1c1c2e",
    button_secondary_background_fill_hover="#252540",
    button_secondary_border_color="#35355a",
    button_secondary_text_color="#9090c0",
    panel_background_fill="#17171f",
    panel_border_color="#27273e",
    checkbox_background_color_selected="#f97316",
    table_even_background_fill="#131320",
    table_odd_background_fill="#17171f",
    shadow_drop="none",
    shadow_drop_lg="0 4px 24px rgba(0,0,0,0.6)",
    slider_color="#f97316",
    color_accent="#f97316",
)

_log('app.py import start')
with gr.Blocks(title='Manhwa Tool') as demo:
    gr.Markdown('# Manhwa Tool')
    with gr.Accordion('Startup / Import Diagnostics', open=False):
        startup_diag = gr.Textbox(label='Diagnostics', lines=20, value='\n'.join(STARTUP_LOG), interactive=False)
        diag_refresh_btn = gr.Button("Refresh Diagnostics", size="sm")

    state = gr.State(None)
    sync_token = gr.State(0)

    _proj_load_fn = None
    _proj_load_outputs = None
    try:
        build_projects_tab = _safe_import('build', 'build_projects_tab')
        _log('render start -> build_projects_tab')
        _proj_result = build_projects_tab(state, sync_token)
        if _proj_result:
            _proj_load_fn, _proj_load_outputs = _proj_result
        _log('render ok -> build_projects_tab')
    except Exception as e:
        _log(f'render fail -> build_projects_tab: {type(e).__name__}: {e}')
        _log(traceback.format_exc())
        _render_tab_fallback('📁 Projects', 'build', e)

    _build_mode_radio = None
    try:
        build_tab1 = _safe_import('build', 'build_tab1')
        _log('render start -> build_tab1')
        _build_mode_radio = build_tab1(state, sync_token)
        _log('render ok -> build_tab1')
    except Exception as e:
        _log(f'render fail -> build_tab1: {type(e).__name__}: {e}')
        _log(traceback.format_exc())
        _render_tab_fallback('Tab 1 — Build', 'build', e)

    _prompt_format_dd = None
    _tab2_gallery = None
    try:
        try:
            build_tab2 = _safe_import('director', 'build_tab2')
            tab2_module = 'director'
        except Exception:
            build_tab2 = _safe_import('make', 'build_tab2')
            tab2_module = 'make'
        _log(f'render start -> {tab2_module}.build_tab2')
        _tab2_result = build_tab2(state, sync_token)
        if isinstance(_tab2_result, tuple):
            _prompt_format_dd, _tab2_gallery = _tab2_result
        else:
            _prompt_format_dd = _tab2_result
        _log(f'render ok -> {tab2_module}.build_tab2')
    except Exception as e:
        _log(f'render fail -> tab2: {type(e).__name__}: {e}')
        _log(traceback.format_exc())
        _render_tab_fallback('Tab 2 — Director', 'director', e)

    try:
        build_storyboard_tab = _safe_import('storyboard_tab', 'build_storyboard_tab')
        _log('render start -> storyboard_tab.build_storyboard_tab')
        build_storyboard_tab(state)
        _log('render ok -> storyboard_tab.build_storyboard_tab')
    except Exception as e:
        _log(f'render fail -> storyboard_tab: {type(e).__name__}: {e}')
        _log(traceback.format_exc())
        _render_tab_fallback('🎬 Storyboard', 'storyboard_tab', e)

    _lib_load_fn = None
    _lib_load_outputs = None
    try:
        build_library_tab = _safe_import('library_tab', 'build_library_tab')
        _log('render start -> library_tab.build_library_tab')
        _lib_result = build_library_tab(state)
        if _lib_result:
            _lib_load_fn, _lib_load_outputs = _lib_result
        _log('render ok -> library_tab.build_library_tab')
    except Exception as e:
        _log(f'render fail -> library_tab: {type(e).__name__}: {e}')
        _log(traceback.format_exc())
        _render_tab_fallback('🎭 Library', 'library_tab', e)

    # ── Cross-tab sync: Tab 1 Panel/Normal radio → Tab 2 prompt format ─────
    if _build_mode_radio is not None and _prompt_format_dd is not None:
        def _sync_build_mode_to_format(mode):
            if mode in ("Panel", "Shorts"):
                return gr.update(value="panel")
            return gr.update(value="normal")
        _build_mode_radio.change(
            _sync_build_mode_to_format,
            inputs=[_build_mode_radio],
            outputs=[_prompt_format_dd],
        )

    # Library tab stats/gallery: NOT loaded on page connect — too expensive to
    # push 200 image paths over SSE while the browser is initializing 700+ components.
    # Users click "🔄 Refresh Stats" inside the Library tab to populate it.
    # (_lib_load_fn and _lib_load_outputs are kept for future use but not registered here.)

    # Auto-load REMOVED: demo.load updating `state` triggers state.change handlers
    # (age_char_refresh_cb → Tab 1, _maybe_refresh_tab2 → Tab 2) which force-render
    # those tabs' Svelte components (150 + 166 = 316 components) simultaneously with
    # page load — this is what caused the persistent "Page Unresponsive" freeze.
    # Users load their project via the Projects tab → Resume Project button.

    def refresh_diag():
        return '\n'.join(STARTUP_LOG)

    diag_refresh_btn.click(refresh_diag, outputs=[startup_diag])


_log('app.py import complete')

# Queue must be called before mounting
demo.queue()

# ── FastAPI app with custom zip-download endpoint ──────────────────────────
fastapi_app = FastAPI()

# Disable proxy buffering so Gradio's SSE stream reaches the browser live.
# Without this Replit's nginx proxy buffers chunks and the UI only updates on refresh.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _Request

class _NoBufMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: _Request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # SSE streams must not be buffered by nginx and must not be cached
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
        # All other responses (JS bundle, CSS, config JSON, etc.) keep their
        # original cache headers so the browser can cache them across reloads.
        return response

fastapi_app.add_middleware(_NoBufMiddleware)


@fastapi_app.get("/zip-download/{filename}")
async def _serve_zip(filename: str):
    """Serve a zip: local /tmp/ first, then fall back to GCS (for autoscale)."""
    if ".." in filename or "/" in filename or not filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = f"/tmp/{filename}"
    # ── fast path: file is already on this instance ───────────────────────────
    if os.path.isfile(path):
        return FileResponse(
            path, filename=filename, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    # ── fallback: fetch from GCS (different autoscale instance built it) ──────
    try:
        import cloud_storage as _cs
        if _cs.is_available():
            ok = _cs.download_zip_to_file(filename, path)
            if ok and os.path.isfile(path):
                return FileResponse(
                    path, filename=filename, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="File not found — it may have expired. Please re-zip.")


# Mount Gradio at root — theme and css must be passed here in Gradio 6
# allowed_paths must include all local directories that gallery/image components
# need to serve.  The projects/ folder is relative to the app CWD (manhwa/).
_PROJECTS_DIR = os.path.abspath("projects")
_LIB_IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "character_library", "images"))
fastapi_app = gr.mount_gradio_app(
    fastapi_app, demo, path="/",
    theme=MANHWA_THEME,
    css=CSS,
    ssr_mode=False,
    allowed_paths=["/tmp", _PROJECTS_DIR, _LIB_IMAGES_DIR],
    show_error=True,
)

# On startup: sync any existing local projects to cloud (background, non-blocking)
try:
    import cloud_storage as _cs
    import threading as _t
    def _startup_sync():
        if _cs.is_available():
            _cs.sync_all_projects_to_cloud_bg("projects")
    _t.Thread(target=_startup_sync, daemon=True).start()
except Exception:
    pass

# Startup bulk-upload disabled — it hammered FAL with 6 parallel workers (957 images)
# which saturated the API and caused generation calls to time out.
# Images are now uploaded on-demand in _get_fal_url() as each ref is needed.

# On startup: delete local images for projects not touched in 7+ days
# Metadata (project.json, manifest.jsonl, refs) is kept so projects still appear in the list.
# Images are already backed up to cloud storage after every generation.
try:
    import shutil as _shutil
    import threading as _t2

    def _cleanup_old_local_images(projects_root: str = "projects", max_age_days: int = 7):
        import time as _time
        cutoff = _time.time() - max_age_days * 86400
        freed = 0
        try:
            base = os.path.join(os.path.dirname(__file__), projects_root)
            if not os.path.isdir(base):
                return  # A fresh source checkout has no saved projects yet.
            for pid in os.listdir(base):
                proj_dir = os.path.join(base, pid)
                if not os.path.isdir(proj_dir):
                    continue
                images_dir = os.path.join(proj_dir, "images")
                if not os.path.isdir(images_dir):
                    continue
                # Use the most recent file mtime inside images/ as the last-used time
                try:
                    mtimes = [os.path.getmtime(os.path.join(images_dir, f))
                              for f in os.listdir(images_dir)]
                    last_used = max(mtimes) if mtimes else os.path.getmtime(proj_dir)
                except Exception:
                    last_used = os.path.getmtime(proj_dir)
                if last_used < cutoff:
                    try:
                        size = sum(
                            os.path.getsize(os.path.join(images_dir, f))
                            for f in os.listdir(images_dir)
                            if os.path.isfile(os.path.join(images_dir, f))
                        )
                        _shutil.rmtree(images_dir, ignore_errors=True)
                        freed += size
                        _log(f"[cleanup] Removed local images for {pid} ({size//1024//1024}MB freed, last used {int((_time.time()-last_used)/86400)}d ago)")
                    except Exception as _e:
                        _log(f"[cleanup] Failed to clean {pid}: {_e}")
        except Exception as _e:
            _log(f"[cleanup] Error during local image cleanup: {_e}")
        if freed:
            _log(f"[cleanup] Total freed: {freed//1024//1024}MB")

    _t2.Thread(target=_cleanup_old_local_images, daemon=True).start()
except Exception:
    pass

if __name__ == '__main__':
    _log('launch start')
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="warning")
