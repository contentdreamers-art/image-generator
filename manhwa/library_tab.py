"""
library_tab.py — Character Library & Story Cast tab for the Manhwa Tool.

Sub-tabs:
  📚 Templates  — add/edit templates, AI auto-analysis, FAL upload
  🎭 Story Cast — assign templates to story characters
  🔍 Gap Analysis — find scene types underrepresented in the library
  📥 Import — bulk URL import + (future) Pinterest sync
"""

import os
import threading
from typing import Any, Dict, List, Optional

import gradio as gr

import character_library as lib_mod


# ── Helpers ───────────────────────────────────────────────────────────────────

def _template_choices(category: str) -> List[str]:
    return [
        f"{t['template_id']} — {t.get('name', t['template_id'])}"
        for t in lib_mod.list_templates(category)
    ]


def _all_template_choices() -> List[str]:
    lib = lib_mod.load_library()
    templates = sorted(lib.get("templates", {}).values(), key=lambda t: t.get("template_id", ""))
    return [f"{t['template_id']} — {t.get('name', t['template_id'])}" for t in templates]


def _parse_tid(choice: str) -> str:
    return (choice or "").split(" — ")[0].strip()


def _ai_analysis_html(analysis: Optional[Dict]) -> str:
    if not analysis:
        return "<div style='color:#555;font-size:12px'>No AI analysis yet — click 🧠 Analyze to run.</div>"
    sev_colors = {"high": "#f87171", "medium": "#fbbf24", "low": "#34d399", "ok": "#6ee7b7"}
    tags = " ".join(
        f"<span style='background:#1e293b;border:1px solid #334155;border-radius:4px;"
        f"padding:1px 6px;font-size:11px;color:#94a3b8;margin:2px'>{t}</span>"
        for t in (analysis.get("tags") or [])
    )
    scenes = ", ".join(analysis.get("scene_suitability") or [])
    features = " · ".join(analysis.get("distinctive_features") or []) or "—"
    return (
        f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:6px;"
        f"padding:10px 14px;font-size:12.5px;line-height:1.8;color:#cbd5e1'>"
        f"<b style='color:#f97316'>{analysis.get('auto_name','—')}</b>"
        f" &nbsp;·&nbsp; {analysis.get('perceived_gender','—')} &nbsp;·&nbsp; "
        f"{analysis.get('age_group','—')} &nbsp;·&nbsp; {analysis.get('body_type','—')}<br>"
        f"<b>Archetype:</b> {analysis.get('archetype','—')} &nbsp; "
        f"<b>Mood:</b> {analysis.get('mood','—')}<br>"
        f"<b>Scene fit:</b> {scenes or '—'}<br>"
        f"<b>Hair:</b> {analysis.get('hair','—')} &nbsp; "
        f"<b>Skin:</b> {analysis.get('skin_tone','—')}<br>"
        f"<b>Features:</b> {features}<br>"
        f"<b>Clothing:</b> {analysis.get('clothing_description','—')}<br>"
        f"<div style='margin-top:6px'>{tags}</div>"
        f"</div>"
    )


def _gap_rows(gaps: List[Dict]) -> List[List[str]]:
    SEV = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low", "ok": "✅ OK"}
    return [
        [
            g["label"],
            str(g["beats_using_it"]),
            str(g["template_count"]),
            SEV.get(g["severity"], g["severity"]),
            ", ".join(g["tags_needed"]),
        ]
        for g in gaps
    ]


# ── Main tab builder ──────────────────────────────────────────────────────────

def build_library_tab(state: gr.State):
    with gr.Tab("🎭 Library"):

        lib_status = gr.Textbox(label="Status", interactive=False, lines=1, value="")

        with gr.Tabs():
            load_fn, load_outputs = _build_templates_subtab(lib_status)
            _build_cast_subtab(state, lib_status)
            _build_clothing_classes_subtab(lib_status)
            _build_gaps_subtab(lib_status)
            _build_import_subtab(lib_status)

    return load_fn, load_outputs


# ── Templates sub-tab ─────────────────────────────────────────────────────────

def _build_templates_subtab(lib_status: gr.Textbox) -> None:
    with gr.Tab("📚 Templates"):

        # ── Stats header ──────────────────────────────────────────────────────
        def _stats_text():
            s = lib_mod.get_library_stats()
            pct = f"{s['analyzed'] / s['total'] * 100:.0f}%" if s["total"] else "0%"
            return (
                f"🗂  Total: {s['total']}   ✅ Analyzed: {s['analyzed']}   "
                f"⏳ Needs analysis: {s['needs_analysis']}   ({pct} complete)"
            )

        def _activity_log_text():
            s = lib_mod.get_library_stats()
            if not s["activity_log"]:
                return "No activity yet."
            lines = []
            for entry in s["activity_log"]:
                lines.append(f"{entry['label']}  —  {entry['count']} image(s) added")
            return "\n".join(lines)

        with gr.Row():
            stats_box = gr.Textbox(
                value="",
                label="Library Overview",
                interactive=False,
                lines=1,
                scale=5,
                placeholder="Click 🔄 Refresh Stats to load counts",
            )
            refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary", scale=1, size="sm")

        # ── Category bar ─────────────────────────────────────────────────────
        lib0 = lib_mod.load_library()
        cats0 = lib_mod.get_categories(lib0)
        # Prefer "other" (where bulk imports land), then first non-empty, then fallback
        if "other" in cats0 and lib_mod.list_templates("other"):
            _default_cat = "other"
        else:
            _default_cat = next(
                (c for c in cats0 if lib_mod.list_templates(c)), cats0[0] if cats0 else "other"
            )

        _CAT_ALL = "✦ All Categories"
        _cat_choices = [_CAT_ALL] + cats0
        _default_cat = _CAT_ALL   # always open on the full library

        with gr.Row():
            category_dd = gr.Dropdown(
                label="Category", choices=_cat_choices,
                value=_default_cat, scale=2,
            )
            new_cat_input = gr.Textbox(
                label="New category name",
                placeholder="e.g. villain, elderly, beast...", scale=2,
            )
            add_cat_btn = gr.Button("+ Add Category", variant="secondary", scale=1, size="sm")

        # ── Tag filter ────────────────────────────────────────────────────────
        def _get_all_tags():
            """Sorted list of (tag, count) for all templates that have been analyzed."""
            from collections import Counter as _Counter
            counts = _Counter()
            for t in lib_mod.load_library().get("templates", {}).values():
                for tag in (t.get("tags") or []):
                    counts[tag] += 1
            # Sort by count desc, then alpha
            return [f"{tag}  ({n})" for tag, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

        # Tags + sort in one row
        with gr.Row():
            tag_filter = gr.Dropdown(
                label="Filter by tags  (AND logic: image must have ALL selected)",
                choices=[],
                value=[],
                multiselect=True,
                scale=4,
            )
            sort_dd = gr.Dropdown(
                label="Sort",
                choices=["Newest → Oldest", "Oldest → Newest", "A-Z by name"],
                value="Newest → Oldest",
                scale=1,
                min_width=180,
            )

        # ── Semantic search bar ───────────────────────────────────────────────
        with gr.Row():
            lib_search_box = gr.Textbox(
                label="🔍 Describe what you're looking for",
                placeholder="e.g. 'angry crowd watching someone', 'lone warrior at sunset', 'snake coiled'…",
                scale=5,
            )
            lib_search_btn  = gr.Button("Search",       variant="primary",    scale=1)
            lib_build_btn   = gr.Button("📐 Build Index", variant="secondary", scale=1)
        lib_search_status = gr.Textbox(label="", interactive=False, visible=True,
                                       placeholder="Type a description and hit Search · Build Index once to enable semantic search")

        # State: template IDs currently shown in gallery (populated by search)
        lib_search_ids = gr.State([])

        # ── Gallery ───────────────────────────────────────────────────────────
        def _get_filtered_templates(cat, selected_tags, sort="Newest → Oldest"):
            """
            Return templates matching category + ALL selected tags.
            cat=_CAT_ALL means search every category.
            When tags are selected, always search ALL categories — the counts shown
            in the tag dropdown are global, so filtering by category would mislead.
            selected_tags is a list of strings like ["angry  (32)", "outdoor  (15)"].
            Uses field presence (no os.path.exists) for speed with large libraries.
            """
            import datetime as _dt

            # Parse bare tag names from "tag  (count)" display format
            required = set()
            for entry in (selected_tags or []):
                bare = entry.split("  (")[0].strip()
                if bare:
                    required.add(bare)

            # Any active tag filter → ignore category, search everything
            if required or cat == _CAT_ALL:
                all_t = list(lib_mod.load_library().get("templates", {}).values())
            else:
                all_t = lib_mod.list_templates(cat)

            result = []
            for t in all_t:
                # Check field presence instead of os.path.exists (no filesystem calls)
                if not t.get("local_face"):
                    continue
                if required:
                    t_tags = set(t.get("tags") or [])
                    if not required.issubset(t_tags):
                        continue
                result.append(t)

            # Apply sort
            def _ts(t):
                raw = t.get("created_at") or ""
                try:
                    return _dt.datetime.fromisoformat(raw.rstrip("Z")).timestamp()
                except Exception:
                    return 0.0

            def _tid_num(t):
                raw = (t.get("template_id") or "").lstrip("OoQq")
                try:
                    return int(raw)
                except Exception:
                    return 0

            if sort == "Newest → Oldest":
                result.sort(key=lambda t: (-_ts(t), -_tid_num(t)))
            elif sort == "Oldest → Newest":
                result.sort(key=lambda t: (_ts(t), _tid_num(t)))
            else:  # A-Z by name
                result.sort(key=lambda t: (t.get("name") or t.get("template_id") or "").lower())

            return result

        _GALLERY_PAGE_SIZE = 200  # max images shown at once in All Categories

        def _gallery_images(cat, selected_tags=None, sort="Newest → Oldest"):
            """Return (file_path, label) pairs for the filtered template set.
            Caps at _GALLERY_PAGE_SIZE for All Categories to avoid slow loads."""
            templates = _get_filtered_templates(cat, selected_tags or [], sort)
            if cat == _CAT_ALL and not selected_tags:
                templates = templates[:_GALLERY_PAGE_SIZE]
            return [
                (lib_mod.image_path(t["template_id"], "face"),
                 t.get("name") or t["template_id"])
                for t in templates
            ]

        # Start empty — gallery loads when the user selects a category or clicks Refresh Stats
        gallery = gr.Gallery(
            label="📸 Select a category above to browse images",
            value=[],
            columns=6, rows=3, height=340,
            object_fit="cover", show_label=True,
            allow_preview=True,
        )

        # ── Primary bulk actions ──────────────────────────────────────────────
        with gr.Row():
            analyze_all_btn    = gr.Button("🧠 Analyze All Unanalyzed", variant="primary",   scale=2)
            auto_tag_free_btn  = gr.Button("🏷️ Auto-Tag All (free)",    variant="secondary", scale=2)
            upload_all_fal_btn = gr.Button("☁️ Upload All to FAL",      variant="secondary", scale=2)
            analyze_all_out = gr.Textbox(
                label="Progress", interactive=False, scale=4,
                placeholder="Click Analyze to tag images with AI · Click Upload All to FAL to enable NB2 Edit refs…",
            )

        # ── Activity log ──────────────────────────────────────────────────────
        activity_log_box = gr.Textbox(
            value="",
            label="📋 Import History (grouped by hour, UTC — newest first)",
            interactive=False,
            lines=6,
            placeholder="Click 🔄 Refresh Stats to load import history",
        )

        # ── Template selector ─────────────────────────────────────────────────
        with gr.Row():
            template_dd = gr.Dropdown(
                label="Template",
                choices=[],
                value=None, scale=3,
            )
            add_tmpl_btn  = gr.Button("➕ New Template",      variant="secondary", scale=1, size="sm")
            del_tmpl_btn  = gr.Button("🗑️ Delete",           variant="stop",      scale=1, size="sm")
            del_confirm   = gr.Button("⚠️ Confirm delete?",  variant="stop",      scale=1, size="sm", visible=False)

        # ── Per-template editor ───────────────────────────────────────────────
        with gr.Row():
            face_img = gr.Image(label="Face Reference", type="pil", scale=1, height=260)
            body_img = gr.Image(label="Body Reference", type="pil", scale=1, height=260)

        with gr.Row():
            tmpl_id_box = gr.Textbox(label="ID", interactive=False, scale=1)
            tmpl_name   = gr.Textbox(label="Name", scale=2)
            tmpl_tags   = gr.Textbox(label="Tags (comma-separated)", scale=3,
                                     placeholder="young, warrior, sad...")

        with gr.Row():
            save_imgs_btn    = gr.Button("💾 Save Images",       variant="primary",   scale=1)
            reanalyze_btn    = gr.Button("🔬 Re-analyze",        variant="secondary", scale=1)
            upload_fal_btn   = gr.Button("☁️ Upload to FAL",     variant="secondary", scale=1)
            refresh_tmpl_btn = gr.Button("🔄 Refresh",           variant="secondary", scale=1, size="sm")

        # keep analyze_btn alias so wiring below compiles (reanalyze does same job)
        analyze_btn = reanalyze_btn

        with gr.Row():
            fal_face_box  = gr.Textbox(label="FAL Face URL", interactive=False, scale=3, placeholder="Not uploaded yet")
            fal_body_box  = gr.Textbox(label="FAL Body URL", interactive=False, scale=3, placeholder="Not uploaded yet")
            usage_count_box = gr.Textbox(label="Times Used (global)", interactive=False, scale=1,
                                         placeholder="0", info="How many times these images have been sent to the model across all projects and generations.")

        ai_html = gr.HTML(value="")

        # ── Wiring ────────────────────────────────────────────────────────────

        def _refresh_gallery(cat, selected_tags, sort="Newest → Oldest"):
            """Recompute gallery for any category/tag/sort change."""
            imgs  = _gallery_images(cat, selected_tags, sort)
            count = len(imgs)
            tag_count = len(selected_tags or [])
            if tag_count:
                label = f"📸 {count} results matching {tag_count} tag(s) — click to edit"
            else:
                label = f"📸 {count} images — click any to edit"
            # Template dropdown only makes sense for a single category
            if cat == _CAT_ALL:
                choices = []
            else:
                choices = _template_choices(cat)
            return (
                gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                gr.Gallery(value=imgs, label=label),
            )

        # Keep old name as alias so category_dd.change can call it with one arg
        def _refresh_category(cat):
            return _refresh_gallery(cat, [])

        def _gallery_select(evt: gr.SelectData, cat, selected_tags, sort, search_ids):
            """When user clicks a gallery thumbnail, load that template into the editor."""
            if search_ids:
                # Search mode: IDs are stored in state
                if evt.index >= len(search_ids):
                    return (gr.Dropdown(), "") + ("", "", None, None, "", "", "")
                tid = search_ids[evt.index]
                t   = lib_mod.get_template(tid)
                if not t:
                    return (gr.Dropdown(), "") + ("", "", None, None, "", "", "")
            else:
                # Normal mode: use filtered list index
                templates = _get_filtered_templates(cat, selected_tags or [], sort or "Newest → Oldest")
                if evt.index >= len(templates):
                    return (gr.Dropdown(), "") + ("", "", None, None, "", "", "")
                t   = templates[evt.index]
                tid = t["template_id"]
            # Template dropdown: use the template's own category if we're in "All"
            eff_cat = t.get("category", cat) if cat == _CAT_ALL else cat
            choices = _template_choices(eff_cat)
            sel = next((c for c in choices if c.startswith(tid)), None)
            tags_str  = ", ".join(t.get("tags") or [])
            face      = lib_mod.get_face_image(tid)
            body      = lib_mod.get_body_image(tid)
            fal_face  = t.get("fal_face_url", "")
            fal_body  = t.get("fal_body_url", "")
            ai_markup = _ai_analysis_html(t.get("ai_analysis"))
            usage_str = _template_usage_str(t)
            return (
                gr.Dropdown(choices=choices, value=sel),
                tid, t.get("name", tid), tags_str, face, body,
                fal_face, fal_body, ai_markup, usage_str,
            )

        def _do_lib_search(query):
            """Semantic search over the library. Returns (gallery, search_ids, status)."""
            import semantic_search as sem
            if not (query or "").strip():
                # Clear search — restore default gallery
                imgs = _gallery_images(_default_cat, [])
                return gr.Gallery(value=imgs, label=f"📸 {len(imgs)} images"), [], ""

            if not sem.is_ready():
                # Fall back to tag search
                try:
                    from storyboard_tab import _search_library
                    imgs, ids = _search_library(query)
                    return (
                        gr.Gallery(value=imgs, label=f"📸 {len(imgs)} tag results for '{query}'"),
                        ids,
                        f"⚠️ No index yet — showing tag results. Click 'Build Index' for semantic search.",
                    )
                except Exception:
                    return gr.Gallery(value=[], label="No results"), [], "⚠️ Build Index first."

            # Approach 3: embeddings top-100 → LLM re-rank → best 20
            results = sem.search_with_rerank(query, first_pass=100, final=20)
            imgs, ids = [], []
            for tid, score in results:
                t = lib_mod.get_template(tid)
                if t and t.get("local_face"):
                    path = lib_mod.image_path(tid, "face")
                    imgs.append((path, t.get("name") or tid))
                    ids.append(tid)

            status = f"✅ {len(imgs)} results for '{query}' (embeddings + AI re-rank)"
            return gr.Gallery(value=imgs, label=f"📸 {status}"), ids, status

        def _build_index():
            """Build or update the embedding index for all library images."""
            import semantic_search as sem
            before = sem.index_size()
            try:
                added = sem.ensure_current()
                total = sem.index_size()
                if added == 0:
                    return f"✅ Index already up to date ({total} templates indexed)"
                return f"✅ Indexed {added} new templates ({total} total)"
            except Exception as e:
                return f"❌ Error building index: {e}"

        def _add_category(new_name):
            slug = (new_name or "").strip()
            if not slug:
                return gr.Dropdown(), "", "⚠️ Enter a category name first."
            slug, cats = lib_mod.add_category(slug)
            if not slug:
                return gr.Dropdown(), "", "⚠️ Invalid category name."
            return gr.Dropdown(choices=cats, value=slug), "", f"✅ Category '{slug}' added."

        def _template_usage_str(t: dict) -> str:
            """Return a human-readable usage count for a template from the global db."""
            usage = lib_mod.load_global_url_usage()
            face_url = t.get("fal_face_url") or ""
            body_url = t.get("fal_body_url") or ""
            total = usage.get(face_url, 0) + usage.get(body_url, 0)
            if total == 0:
                return "0 — never used"
            return str(total)

        def _load_template(tmpl_choice):
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return "", "", "", None, None, "", "", "", ""
            t = lib_mod.get_template(tid)
            if not t:
                return tid, "", "", None, None, "", "", "", ""
            tags_str  = ", ".join(t.get("tags") or [])
            face      = lib_mod.get_face_image(tid)
            body      = lib_mod.get_body_image(tid)
            fal_face  = t.get("fal_face_url", "")
            fal_body  = t.get("fal_body_url", "")
            ai_markup = _ai_analysis_html(t.get("ai_analysis"))
            usage_str = _template_usage_str(t)
            return tid, t.get("name", tid), tags_str, face, body, fal_face, fal_body, ai_markup, usage_str

        def _add_template(category):
            if not category:
                return gr.Dropdown(), "⚠️ Select a category first."
            tid, _ = lib_mod.add_template(category)
            choices = _template_choices(category)
            sel = next((c for c in choices if c.startswith(tid)), choices[-1] if choices else None)
            return gr.Dropdown(choices=choices, value=sel), f"✅ Created {tid} in '{category}'."

        def _arm_delete(tmpl_choice):
            """First click: show the confirm button instead of deleting."""
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return gr.Button(visible=False), "⚠️ No template selected."
            return gr.Button(visible=True), f"⚠️ About to delete {tid} — click Confirm to proceed."

        def _delete_template(tmpl_choice, category):
            """Second click (confirm): actually delete."""
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return gr.Dropdown(), gr.Button(visible=False), "⚠️ No template selected."
            ok = lib_mod.delete_template(tid)
            choices = _template_choices(category or "female")
            return (
                gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                gr.Button(visible=False),
                f"✅ Deleted {tid}." if ok else f"⚠️ Not found: {tid}",
            )

        def _save_images(tmpl_choice, face_pil, body_pil, name_val, tags_val):
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return "⚠️ No template selected."
            tags = [t.strip() for t in (tags_val or "").split(",") if t.strip()]
            lib_mod.update_template(tid, name=name_val or tid, tags=tags)
            saved = []
            if face_pil is not None:
                lib_mod.save_image(tid, "face", face_pil)
                saved.append("face")
            if body_pil is not None:
                lib_mod.save_image(tid, "body", body_pil)
                saved.append("body")
            if saved:
                return f"✅ Saved {', '.join(saved)} for {tid}."
            return f"✅ Updated metadata for {tid} (no new images)."

        def _analyze_template(tmpl_choice):
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return "⚠️ No template selected.", "", ""
            result = lib_mod.analyze_and_update_template(tid)
            if "error" in result:
                return f"⚠️ Analysis failed: {result['error']}", "", _ai_analysis_html(None)
            # Reload template to get updated name/tags
            t = lib_mod.get_template(tid) or {}
            tags_str = ", ".join(t.get("tags") or [])
            ai_markup = _ai_analysis_html(result)
            return (
                f"✅ Analyzed {tid} — {result.get('auto_name', '')} / "
                f"{result.get('perceived_gender', '')} / {result.get('archetype', '')}",
                tags_str,
                ai_markup,
            )

        def _upload_to_fal(tmpl_choice):
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return "", "", "⚠️ No template selected."
            result = lib_mod.upload_template_to_fal(tid)
            status = result.get("status", "ok")
            msg = f"✅ {tid} uploaded." if status == "ok" else f"⚠️ {status}"
            return result.get("face_url", ""), result.get("body_url", ""), msg

        def _upload_all_to_fal():
            """Stream bulk FAL upload progress via thread+queue."""
            import queue as _queue

            lib = lib_mod.load_library()
            templates = lib.get("templates", {})
            need = sum(
                1 for t in templates.values()
                if not t.get("fal_face_url")
                and os.path.exists(t.get("local_face", ""))
            )
            if need == 0:
                yield "✅ All images already have FAL URLs — nothing to upload!"
                return

            yield f"⏳ Starting bulk FAL upload — {need} images to upload (6 parallel workers)…"

            q: _queue.Queue = _queue.Queue()
            _DONE = object()

            def _progress(done, total, tid, status):
                icon = "✅" if status == "ok" else ("⏭" if status == "skipped" else "⚠️")
                q.put(f"{icon} {done}/{total} — {tid} ({status})")

            def _run():
                try:
                    r = lib_mod.bulk_upload_all_to_fal(progress_cb=_progress)
                    q.put(
                        f"🎉 Done! {r['ok']} uploaded, {r['skipped']} already had URLs"
                        + (f", {r['errors']} errors" if r["errors"] else "")
                    )
                except Exception as exc:
                    q.put(f"❌ Bulk upload failed: {exc}")
                q.put(_DONE)

            threading.Thread(target=_run, daemon=True).start()

            while True:
                msg = q.get()
                if msg is _DONE:
                    break
                yield msg

        def _analyze_all():
            """Stream progress using 10 parallel workers via bulk_analyze_unanalyzed."""
            import threading, queue as _queue

            lib = lib_mod.load_library()
            templates = lib.get("templates", {})
            pending = [
                t for t in templates.values()
                if not t.get("ai_analysis")
            ]
            total = len(pending)
            if total == 0:
                yield "✅ All images are already analyzed!"
                return

            yield f"⏳ Starting… {total} images to analyze (10 parallel workers)"

            q: _queue.Queue = _queue.Queue()
            _DONE = object()

            def _progress(done, tot, tid):
                name = (templates.get(tid) or {}).get("name") or tid
                q.put(f"✅ {done}/{tot} — {name[:55]}")

            def _run():
                result = lib_mod.bulk_analyze_unanalyzed(
                    progress_cb=_progress, max_workers=10
                )
                done   = result.get("done", 0)
                errors = result.get("errors", [])
                summary = f"🎉 Done! {done}/{total} analyzed"
                if errors:
                    summary += f" ({len(errors)} errors)"
                q.put(summary)
                q.put(_DONE)

            threading.Thread(target=_run, daemon=True).start()

            while True:
                msg = q.get()
                if msg is _DONE:
                    break
                yield msg

        # ── Events ────────────────────────────────────────────────────────────

        add_cat_btn.click(
            _add_category,
            inputs=[new_cat_input],
            outputs=[category_dd, new_cat_input, lib_status],
        )
        # Category, tag, or sort change → refresh gallery and clear search
        def _refresh_and_clear(cat, selected_tags, sort):
            dd, gal = _refresh_gallery(cat, selected_tags, sort)
            return dd, gal, [], gr.Textbox(value=""), ""

        category_dd.change(
            _refresh_and_clear, inputs=[category_dd, tag_filter, sort_dd],
            outputs=[template_dd, gallery, lib_search_ids, lib_search_box, lib_search_status],
        )
        tag_filter.change(
            _refresh_and_clear, inputs=[category_dd, tag_filter, sort_dd],
            outputs=[template_dd, gallery, lib_search_ids, lib_search_box, lib_search_status],
        )
        sort_dd.change(
            _refresh_and_clear, inputs=[category_dd, tag_filter, sort_dd],
            outputs=[template_dd, gallery, lib_search_ids, lib_search_box, lib_search_status],
        )

        # Semantic search
        lib_search_btn.click(
            _do_lib_search, inputs=[lib_search_box],
            outputs=[gallery, lib_search_ids, lib_search_status],
        )
        lib_search_box.submit(
            _do_lib_search, inputs=[lib_search_box],
            outputs=[gallery, lib_search_ids, lib_search_status],
        )
        lib_build_btn.click(_build_index, outputs=[lib_search_status])

        gallery.select(
            _gallery_select,
            inputs=[category_dd, tag_filter, sort_dd, lib_search_ids],
            outputs=[template_dd, tmpl_id_box, tmpl_name, tmpl_tags,
                     face_img, body_img, fal_face_box, fal_body_box, ai_html, usage_count_box],
        )
        template_dd.change(
            _load_template, inputs=[template_dd],
            outputs=[tmpl_id_box, tmpl_name, tmpl_tags, face_img, body_img,
                     fal_face_box, fal_body_box, ai_html, usage_count_box],
        )
        add_tmpl_btn.click(
            _add_template, inputs=[category_dd], outputs=[template_dd, lib_status],
        )
        del_tmpl_btn.click(
            _arm_delete, inputs=[template_dd],
            outputs=[del_confirm, lib_status],
        )
        del_confirm.click(
            _delete_template, inputs=[template_dd, category_dd],
            outputs=[template_dd, del_confirm, lib_status],
        )
        save_imgs_btn.click(
            _save_images,
            inputs=[template_dd, face_img, body_img, tmpl_name, tmpl_tags],
            outputs=[lib_status],
        )
        reanalyze_btn.click(
            _analyze_template,
            inputs=[template_dd],
            outputs=[lib_status, tmpl_tags, ai_html],
        )
        upload_fal_btn.click(
            _upload_to_fal, inputs=[template_dd],
            outputs=[fal_face_box, fal_body_box, lib_status],
        )
        refresh_tmpl_btn.click(
            _load_template, inputs=[template_dd],
            outputs=[tmpl_id_box, tmpl_name, tmpl_tags, face_img, body_img,
                     fal_face_box, fal_body_box, ai_html, usage_count_box],
        )
        analyze_all_btn.click(
            _analyze_all, outputs=[analyze_all_out], show_progress=False,
        ).then(
            lambda: (_stats_text(), _activity_log_text()),
            outputs=[stats_box, activity_log_box],
        )
        def _auto_tag_all_free():
            r = lib_mod.enrich_all_tags()
            return (f"✅ Scanned {r['total']} templates — "
                    f"{r['enriched']} updated with new tags: "
                    f"{', '.join(r['new_tags'][:20]) or 'none'}")

        auto_tag_free_btn.click(
            _auto_tag_all_free, outputs=[analyze_all_out],
        ).then(
            _get_all_tags, outputs=[tag_filter],
        )
        upload_all_fal_btn.click(
            _upload_all_to_fal, outputs=[analyze_all_out], show_progress=False,
        )
        refresh_stats_btn.click(
            lambda: (_stats_text(), _activity_log_text()),
            outputs=[stats_box, activity_log_box],
        ).then(
            lambda: _refresh_gallery(_default_cat, []),
            outputs=[template_dd, gallery],
        )

        def _load_library_tab():
            """Called by demo.load() — populates stats, log, gallery, tags on first open."""
            stats = _stats_text()
            log   = _activity_log_text()
            dd_update, gal_update = _refresh_gallery(_default_cat, [])
            tags  = _get_all_tags()
            return stats, log, dd_update, gal_update, gr.Dropdown(choices=tags, value=[])

        return _load_library_tab, [stats_box, activity_log_box, template_dd, gallery, tag_filter]


# ── Story Cast sub-tab ────────────────────────────────────────────────────────

def _build_cast_subtab(state: gr.State, lib_status: gr.Textbox) -> None:
    with gr.Tab("🎭 Story Cast"):
        gr.Markdown(
            "Assign library templates to story characters. "
            "Load a project in Tab 1 first, then auto-cast or assign manually below."
        )

        with gr.Row():
            auto_cast_btn   = gr.Button("🎲 Auto-Cast All", variant="primary",   scale=1)
            save_cast_btn   = gr.Button("💾 Save Cast",     variant="secondary", scale=1)
            clear_cast_btn  = gr.Button("🧹 Clear Cast",    variant="stop",      scale=1)
            refresh_cast_btn = gr.Button("🔄 Refresh",      variant="secondary", scale=1, size="sm")

        cast_status = gr.Textbox(label="Cast Status", interactive=False, lines=1)

        cast_df = gr.Dataframe(
            headers=["Character", "Gender", "Template ID", "Template Name", "Face ✓", "Body ✓"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False, label="Current Story Cast", wrap=True,
        )

        gr.Markdown("### Manual Assignment")
        with gr.Row():
            cast_char_dd = gr.Dropdown(label="Character", choices=[], scale=2)
            cast_tmpl_dd = gr.Dropdown(
                label="Template", choices=[],
                value=None, scale=3,
                info="Click 🔄 Refresh to load template choices",
            )
            assign_btn   = gr.Button("✓ Assign",  variant="primary",   scale=1)
            unassign_btn = gr.Button("✗ Remove",  variant="secondary", scale=1)

        # ── Helpers ───────────────────────────────────────────────────────────
        def _cast_rows(st):
            if not st or not st.characters:
                return []
            cast = getattr(st, "character_cast", {}) or {}
            lib  = lib_mod.load_library()
            rows = []
            for cname, cdata in st.characters.items():
                fields  = cdata.get("fields") or {}
                gender  = fields.get("gender", "")
                tid     = cast.get(cname, "")
                tmpl    = lib.get("templates", {}).get(tid) if tid else None
                tname   = tmpl.get("name", tid) if tmpl else ""
                face_ok = "✓" if (tmpl and tmpl.get("fal_face_url")) else "✗"
                body_ok = "✓" if (tmpl and tmpl.get("fal_body_url")) else "✗"
                rows.append([cname, gender, tid, tname, face_ok, body_ok])
            return rows

        def _refresh_cast(st):
            tmpl_choices = _all_template_choices()
            if not st or not st.characters:
                return [], gr.Dropdown(choices=[]), gr.Dropdown(choices=tmpl_choices), "Load a project in Tab 1 first."
            names = list(st.characters.keys())
            return (
                _cast_rows(st),
                gr.Dropdown(choices=names, value=names[0] if names else None),
                gr.Dropdown(choices=tmpl_choices),
                "",
            )

        def _auto_cast(st):
            if not st or not st.characters:
                return st, [], "⚠️ Load a project first."
            existing = getattr(st, "character_cast", {}) or {}
            new_cast = lib_mod.auto_cast_characters(st.characters, st.project_id or "", existing)
            if not new_cast:
                return st, _cast_rows(st), "⚠️ No eligible templates. Add templates to the library first."
            st.character_cast = {**existing, **new_cast}
            return st, _cast_rows(st), f"✅ Auto-cast {len(new_cast)} character(s). Click 💾 Save Cast to persist."

        def _save_cast(st):
            if not st:
                return st, "⚠️ No project loaded."
            from build import _save_project_json
            _save_project_json(st)
            return st, "✅ Cast saved."

        def _clear_cast(st):
            if not st:
                return st, [], "⚠️ No project loaded."
            st.character_cast = {}
            return st, _cast_rows(st), "✅ Cast cleared. Click 💾 to persist."

        def _assign_one(st, char_name, tmpl_choice):
            if not st or not char_name:
                return st, [], "⚠️ Select a character."
            tid = _parse_tid(tmpl_choice)
            if not tid:
                return st, [], "⚠️ Select a template."
            if not hasattr(st, "character_cast") or st.character_cast is None:
                st.character_cast = {}
            st.character_cast[char_name] = tid
            return st, _cast_rows(st), f"✅ {char_name} → {tid}"

        def _unassign_one(st, char_name):
            if not st or not char_name:
                return st, [], "⚠️ Select a character."
            cast = getattr(st, "character_cast", {}) or {}
            cast.pop(char_name, None)
            st.character_cast = cast
            return st, _cast_rows(st), f"✅ Removed cast for {char_name}."

        # ── Events ────────────────────────────────────────────────────────────
        refresh_cast_btn.click(_refresh_cast, inputs=[state], outputs=[cast_df, cast_char_dd, cast_tmpl_dd, cast_status])
        auto_cast_btn.click(_auto_cast,       inputs=[state], outputs=[state, cast_df, cast_status])
        save_cast_btn.click(_save_cast,       inputs=[state], outputs=[state, cast_status])
        clear_cast_btn.click(_clear_cast,     inputs=[state], outputs=[state, cast_df, cast_status])
        assign_btn.click(_assign_one,    inputs=[state, cast_char_dd, cast_tmpl_dd], outputs=[state, cast_df, cast_status])
        unassign_btn.click(_unassign_one, inputs=[state, cast_char_dd],              outputs=[state, cast_df, cast_status])


# ── Clothing Classes sub-tab ──────────────────────────────────────────────────

def _build_clothing_classes_subtab(lib_status: gr.Textbox) -> None:
    """Manage clothing classes with up to 10 outfit variations each."""
    with gr.Tab("👗 Clothing Classes"):
        lib_mod.ensure_default_classes()

        gr.Markdown(
            "**How to use:**\n"
            "1. Pick a class from the dropdown.\n"
            "2. Optionally search the library and click an image to attach it as a reference.\n"
            "3. Write the clothing description (what this outfit looks like as a prompt).\n"
            "4. Click **➕ Add Variation** — repeat up to 10 times per class.\n\n"
            "Panel generation picks one variation at random, so characters of the same class "
            "each look different."
        )

        # ── States ────────────────────────────────────────────────────────────
        var_tid_state = gr.State("")
        var_gal_ids   = gr.State([])

        # ── Class selector ────────────────────────────────────────────────────
        def _cls_choices():
            return sorted(lib_mod.get_clothing_classes().keys())

        with gr.Row():
            cls_dd = gr.Dropdown(
                label="Class",
                choices=_cls_choices(),
                value=None,
                scale=5,
            )
            cls_refresh_btn = gr.Button("🔄 Refresh", scale=1)

        with gr.Row():
            new_cls_box = gr.Textbox(label="New class name", placeholder="e.g. samurai, cultist, highborn…", scale=4)
            new_cls_btn = gr.Button("➕ Create class", variant="secondary", scale=1)
            del_cls_btn = gr.Button("🗑️ Delete class", variant="stop", scale=1)

        cls_status = gr.Textbox(label="", interactive=False, lines=1)

        # ── Variations display ─────────────────────────────────────────────────
        def _vars_html(class_name: str) -> str:
            if not class_name:
                return "<p style='color:#555;font-size:12px'>↑ Select a class to see its variations.</p>"
            variations = lib_mod.get_clothing_class_variations(class_name)
            if not variations:
                return (
                    f"<p style='color:#f6ad55;font-size:13px;padding:8px 4px'>"
                    f"<b>👗 {class_name}</b> — no variations yet. Add one below.</p>"
                )
            cards = []
            for i, v in enumerate(variations):
                tid = v.get("tid") or ""
                dna = (v.get("clothing_dna") or "").strip()
                img_html = ""
                if tid:
                    p = lib_mod.image_path(tid, "face") or lib_mod.image_path(tid, "body")
                    if p and os.path.isfile(p):
                        try:
                            import base64 as _b64
                            with open(p, "rb") as _f:
                                b64 = _b64.b64encode(_f.read()).decode()
                            ext  = p.rsplit(".", 1)[-1].lower()
                            mime = "image/jpeg" if ext in ("jpg","jpeg") else f"image/{ext}"
                            img_html = (
                                f"<img src='data:{mime};base64,{b64}' "
                                f"style='width:72px;height:82px;object-fit:cover;"
                                f"border-radius:4px;display:block;margin-bottom:3px'>"
                            )
                        except Exception:
                            pass
                if not img_html:
                    img_html = (
                        "<div style='width:72px;height:82px;background:#2d3748;"
                        "border-radius:4px;display:flex;align-items:center;"
                        "justify-content:center;color:#4a5568;font-size:24px'>👗</div>"
                    )
                dna_preview = ((dna[:60] + "…") if len(dna) > 60 else dna) or \
                              "<em style='color:#718096'>no description</em>"
                cards.append(
                    f"<div style='display:inline-block;vertical-align:top;margin:4px;"
                    f"padding:6px;background:#1a202c;border:1px solid #4a5568;"
                    f"border-radius:6px;min-width:84px;max-width:104px;text-align:center'>"
                    f"{img_html}"
                    f"<div style='color:#68d391;font-size:10px;font-weight:700'>#{i+1}</div>"
                    f"<div style='color:#a0aec0;font-size:9px;text-align:left;"
                    f"margin-top:2px;word-break:break-word'>{dna_preview}</div>"
                    f"</div>"
                )
            hdr = (
                f"<div style='color:#e2e8f0;font-size:13px;font-weight:600;margin-bottom:6px'>"
                f"<span style='color:#f6ad55'>👗 {class_name}</span>"
                f"&nbsp;— {len(variations)}/10 variations</div>"
            )
            return hdr + "<div style='display:flex;flex-wrap:wrap;gap:2px'>" + "".join(cards) + "</div>"

        vars_html = gr.HTML("<p style='color:#555;font-size:12px'>↑ Select a class to see its variations.</p>")

        # ── Inline delete ─────────────────────────────────────────────────────
        def _del_choices(cn):
            out = []
            for i, v in enumerate(lib_mod.get_clothing_class_variations(cn or "")):
                dna = (v.get("clothing_dna") or "")[:50]
                suffix = "…" if len(v.get("clothing_dna") or "") > 50 else ""
                out.append(f"#{i+1}  {dna}{suffix}" if dna else f"#{i+1}  (image only)")
            return out

        with gr.Row():
            del_var_dd  = gr.Dropdown(
                label="× Remove a variation",
                choices=[], value=None, scale=4, interactive=True,
            )
            del_var_btn = gr.Button("× Remove selected", variant="stop", scale=1)
        del_var_status = gr.Textbox(label="", interactive=False, lines=1)

        # ── Add variation ─────────────────────────────────────────────────────
        gr.Markdown("---\n#### Add a Variation")
        gr.Markdown(
            "Search your library or click **✨ Suggestions** to auto-search by class name. "
            "Click any image — its clothing description auto-fills. Then hit **➕ Add Variation**."
        )

        with gr.Row():
            var_srch_box = gr.Textbox(
                label="Search library",
                placeholder="silk robes, armored, torn cloak, gold trim…",
                scale=5,
            )
            var_srch_btn = gr.Button("🔍 Search", scale=1)
            var_sugg_btn = gr.Button("✨ Suggestions", variant="secondary", scale=1)

        var_gallery = gr.Gallery(
            label="Click an image to select it as reference (optional)",
            columns=7, height=480, show_label=True, allow_preview=False,
        )

        var_sel_lbl = gr.Markdown("*No image selected — description alone is fine*")

        var_dna_box = gr.Textbox(
            label="Clothing description  (auto-fills from selected image — or click ✨ Generate)",
            placeholder=(
                "e.g.  ornate silk robes, gold trim, embroidered cuffs, rigid upright posture\n"
                "or:   worn linen shirt, earth-brown trousers, mud-caked leather boots"
            ),
            lines=3,
        )

        with gr.Row():
            var_add_btn = gr.Button("➕ Add Variation", variant="primary", scale=2)
            var_gen_btn = gr.Button("✨ Generate description", variant="secondary", scale=2)
            var_clr_btn = gr.Button("✖ Clear", scale=1)
            var_status  = gr.Textbox(label="", interactive=False, lines=1, scale=4)

        # ── Handlers ──────────────────────────────────────────────────────────

        # Class selected → refresh variations display + delete dropdown
        def _on_class_change(cn):
            return (
                _vars_html(cn),
                gr.Dropdown(choices=_del_choices(cn), value=None),
                "",
                "*No image selected — description alone is fine*",
                "",
            )

        cls_dd.change(
            _on_class_change,
            inputs=[cls_dd],
            outputs=[vars_html, del_var_dd, var_tid_state, var_sel_lbl, var_dna_box],
        )

        # Library search
        def _do_search(q):
            try:
                from storyboard_tab import _search_library
                imgs, ids = _search_library(q)
            except Exception:
                imgs, ids = [], []
            return gr.Gallery(value=imgs), ids

        var_srch_btn.click(_do_search, [var_srch_box], [var_gallery, var_gal_ids])
        var_srch_box.submit(_do_search, [var_srch_box], [var_gallery, var_gal_ids])

        # Suggestions button: auto-search library by class name
        def _do_suggestions(class_name):
            if not class_name:
                return gr.Gallery(value=[]), []
            try:
                from storyboard_tab import _search_library
                imgs, ids = _search_library(class_name)
            except Exception:
                imgs, ids = [], []
            return gr.Gallery(value=imgs), ids

        var_sugg_btn.click(_do_suggestions, [cls_dd], [var_gallery, var_gal_ids])

        # Image clicked → auto-fill description from ai_analysis
        def _on_gallery_select(evt: gr.SelectData, ids):
            if not ids or evt.index >= len(ids):
                return "", "*No image selected — description alone is fine*", ""
            tid = ids[evt.index]
            try:
                lib  = lib_mod.load_library()
                tmpl = lib.get("templates", {}).get(tid) or {}
                ai   = tmpl.get("ai_analysis") or {}
                desc = (ai.get("clothing_description") or "").strip()
            except Exception:
                desc = ""
            return tid, f"*✅ Image selected: `{tid}`*", desc

        var_gallery.select(_on_gallery_select, [var_gal_ids], [var_tid_state, var_sel_lbl, var_dna_box])

        # AI generate description
        def _generate_description(class_name, tid, current_dna):
            class_name = (class_name or "").strip()
            if not class_name:
                return current_dna, "⚠️ Select a class first."
            img_ctx = ""
            if tid:
                try:
                    lib  = lib_mod.load_library()
                    tmpl = lib.get("templates", {}).get(tid) or {}
                    ai   = tmpl.get("ai_analysis") or {}
                    img_ctx = (ai.get("clothing_description") or ai.get("summary") or "").strip()
                except Exception:
                    pass
            prompt = (
                f"Write a concise manhwa panel clothing prompt for the social class: **{class_name}**.\n"
                + (f"Reference image description: {img_ctx}\n" if img_ctx else "")
                + "\nOutput ONLY the clothing description as a single comma-separated phrase — "
                "specific fabrics, colors, notable garments, accessories, silhouette. "
                "No intro, no quotes, no extra text. ~20-35 words."
            )
            try:
                import anthropic as _ant
                client = _ant.Anthropic()
                resp   = client.messages.create(
                    model="claude-opus-4-5", max_tokens=120,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = resp.content[0].text.strip().strip('"').strip("'")
                return result, f"✅ Generated for '{class_name}'."
            except Exception as exc:
                return current_dna, f"⚠️ Generation failed: {exc}"

        var_gen_btn.click(
            _generate_description,
            inputs=[cls_dd, var_tid_state, var_dna_box],
            outputs=[var_dna_box, var_status],
        )

        # Clear form
        var_clr_btn.click(
            lambda: ("", "*No image selected — description alone is fine*", ""),
            inputs=[],
            outputs=[var_tid_state, var_sel_lbl, var_dna_box],
        )

        # Add variation → refresh cards + delete dropdown
        def _add_variation(class_name, tid, dna):
            class_name = (class_name or "").strip().lower()
            if not class_name:
                return _vars_html(""), gr.Dropdown(choices=[], value=None), "⚠️ Select a class first."
            err = lib_mod.add_clothing_class_variation(class_name, tid or "", dna or "")
            if err:
                return _vars_html(class_name), gr.Dropdown(choices=_del_choices(class_name), value=None), f"⚠️ {err}"
            n = len(lib_mod.get_clothing_class_variations(class_name))
            return (
                _vars_html(class_name),
                gr.Dropdown(choices=_del_choices(class_name), value=None),
                f"✅ Added variation #{n} to '{class_name}'.",
            )

        var_add_btn.click(
            _add_variation,
            inputs=[cls_dd, var_tid_state, var_dna_box],
            outputs=[vars_html, del_var_dd, var_status],
        )

        # Delete variation via dropdown
        def _delete_var(class_name, choice):
            class_name = (class_name or "").strip().lower()
            if not class_name:
                return _vars_html(""), gr.Dropdown(choices=[], value=None), "⚠️ Select a class first."
            if not choice:
                return _vars_html(class_name), gr.Dropdown(choices=_del_choices(class_name), value=None), \
                       "⚠️ Select a variation from the dropdown first."
            try:
                idx = int(choice.split("#")[1].split()[0]) - 1
            except Exception:
                return _vars_html(class_name), gr.Dropdown(choices=_del_choices(class_name), value=None), "⚠️ Invalid selection."
            msg = lib_mod.delete_clothing_class_variation(class_name, idx)
            return _vars_html(class_name), gr.Dropdown(choices=_del_choices(class_name), value=None), msg

        del_var_btn.click(
            _delete_var,
            inputs=[cls_dd, del_var_dd],
            outputs=[vars_html, del_var_dd, del_var_status],
        )

        # Create new class
        def _add_class(name):
            name = (name or "").strip().lower()
            if not name:
                return gr.Dropdown(choices=_cls_choices()), "⚠️ Enter a class name."
            lib_mod.set_clothing_class(name)
            return gr.Dropdown(choices=_cls_choices(), value=name), f"✅ Created class '{name}'."

        new_cls_btn.click(_add_class, [new_cls_box], [cls_dd, cls_status])

        # Delete entire class
        def _del_class(class_name):
            class_name = (class_name or "").strip().lower()
            if not class_name:
                return gr.Dropdown(choices=_cls_choices()), _vars_html(""), "⚠️ Select a class first."
            lib_mod.delete_clothing_class(class_name)
            return gr.Dropdown(choices=_cls_choices(), value=None), _vars_html(""), \
                   f"✅ Deleted class '{class_name}'."

        del_cls_btn.click(_del_class, [cls_dd], [cls_dd, vars_html, cls_status])

        # Refresh
        cls_refresh_btn.click(
            lambda: gr.Dropdown(choices=_cls_choices()),
            inputs=[], outputs=[cls_dd],
        )


# ── Gap Analysis sub-tab ──────────────────────────────────────────────────────

def _build_gaps_subtab(lib_status: gr.Textbox) -> None:
    with gr.Tab("🔍 Gaps"):
        gr.Markdown(
            "Scans all your saved story beat plans and checks whether the library has enough "
            "reference images for each scene type. High-severity gaps mean that scene type "
            "appears often in your stories but has very few matching templates."
        )

        run_gaps_btn = gr.Button("🔍 Run Gap Analysis", variant="primary")
        gaps_status  = gr.Textbox(label="Status", interactive=False, lines=1)

        gaps_df = gr.Dataframe(
            headers=["Scene Type", "Beats", "Templates", "Severity", "Tags Needed"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="Coverage Gaps",
            wrap=True,
        )

        gr.Markdown(
            "<span style='color:#6868a0;font-size:12px'>"
            "🔴 **High** = appears often, almost no coverage &nbsp; "
            "🟡 **Medium** = partial coverage &nbsp; "
            "🟢 **Low** = mostly covered &nbsp; "
            "✅ **OK** = well covered"
            "</span>"
        )

        def _run_gaps():
            try:
                gaps = lib_mod.detect_library_gaps()
                rows = _gap_rows(gaps)
                if not rows:
                    return "✅ No significant gaps detected.", []
                high  = sum(1 for g in gaps if g["severity"] == "high")
                med   = sum(1 for g in gaps if g["severity"] == "medium")
                return f"Found {len(gaps)} gaps — 🔴 {high} high, 🟡 {med} medium.", rows
            except Exception as e:
                return f"⚠️ Error: {e}", []

        run_gaps_btn.click(_run_gaps, outputs=[gaps_status, gaps_df])


# ── Pinterest status helper (module level) ────────────────────────────────────

def _pinterest_status_html() -> str:
    connected = bool(os.environ.get("PINTEREST_ACCESS_TOKEN", ""))
    if connected:
        return (
            "<div style='background:#052e16;border:1px solid #16a34a;border-radius:6px;"
            "padding:10px 14px;font-size:13px;color:#86efac'>"
            "✅ <b>Pinterest connected.</b> Click <b>🔄 Load My Boards</b> to list your boards, "
            "or hit <b>💾 Import All Saved Pins</b> to pull everything at once."
            "</div>"
        )
    return (
        "<div style='background:#1c0a00;border:1px solid #f97316;border-radius:6px;"
        "padding:12px 16px;font-size:13px;color:#fed7aa;line-height:1.8'>"
        "<b style='color:#f97316'>One-time setup required</b><br>"
        "1. In this Replit workspace: click <b>Tools → Secrets</b> (or the lock icon in the sidebar)<br>"
        "2. Search for <b>Pinterest</b> in the Connectors panel and click <b>Connect</b><br>"
        "3. Authorize with your Pinterest account<br>"
        "4. Come back here — the buttons below will activate automatically<br><br>"
        "<span style='color:#9a3412'>The server will fetch images directly from Pinterest — "
        "nothing downloads to your computer.</span>"
        "</div>"
    )


# ── Import sub-tab ────────────────────────────────────────────────────────────

def _build_import_subtab(lib_status: gr.Textbox) -> None:
    with gr.Tab("📥 Import"):

        # ── BIG STATUS BAR — always visible ──────────────────────────────────
        pinterest_status = gr.Textbox(
            label="⬇️  Status",
            value="Ready — upload your Pinterest ZIP below to start.",
            interactive=False,
            lines=5,
        )

        # ══════════════════════════════════════════════════════════════════════
        # PRIMARY: Pinterest export (ZIP or HTML files)
        # ══════════════════════════════════════════════════════════════════════
        gr.HTML(
            "<div style='background:#052e16;border:2px solid #16a34a;border-radius:8px;"
            "padding:14px 18px;font-size:14px;color:#86efac;line-height:1.9;margin:8px 0'>"
            "<b style='font-size:16px'>📌 Import Pinterest Export</b><br>"
            "Pinterest emailed you a download link. After downloading:<br>"
            "• <b>If it's still a .zip file</b> — drop the ZIP below<br>"
            "• <b>If macOS auto-unzipped it</b> (you see a <b>pinterest</b> folder) — "
            "open that folder → open <b>pins</b> → select all files (<b>⌘A</b>) → drop them below<br>"
            "The server reads the pin data and downloads every image automatically."
            "</div>"
        )
        pinterest_export_zip = gr.File(
            label="📦 Drop Pinterest ZIP  —  or  —  select all files from the pins/ folder",
            file_count="multiple",
            file_types=[".zip", ".html", ".htm", ".json", ".csv"],
            height=140,
        )
        with gr.Row():
            export_import_btn = gr.Button(
                "📦 Import Pinterest Export", variant="primary", size="lg", scale=3,
            )
            export_analyze = gr.Checkbox(
                label="🧠 Auto-analyze (do later in bulk)", value=False, scale=1,
            )

        gr.HTML("<hr style='border-color:#1e293b;margin:20px 0'>")

        # ══════════════════════════════════════════════════════════════════════
        # SECONDARY OPTIONS (collapsed by default)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🍪  Pinterest — Import via Browser Cookie (instant, no ZIP needed)", open=False):
            gr.HTML(
                "<div style='background:#1e1b4b;border:1px solid #4f46e5;border-radius:6px;"
                "padding:10px 14px;font-size:13px;color:#c7d2fe;line-height:1.8'>"
                "1. Open <b>pinterest.com</b> in Chrome → press <b>Cmd+Option+I</b><br>"
                "2. Click <b>Application</b> tab → <b>Cookies</b> → <b>https://www.pinterest.com</b><br>"
                "3. Find row <b>_pinterest_sess</b> → click its Value → <b>Cmd+A</b> → <b>Cmd+C</b><br>"
                "4. Paste below and click Import"
                "</div>"
            )
            cookie_input = gr.Textbox(
                label="_pinterest_sess cookie value",
                placeholder="TWc... (long string from DevTools)",
                type="password",
            )
            with gr.Row():
                cookie_import_btn = gr.Button("🚀 Import All Saved Pins", variant="primary", scale=2)
                cookie_analyze    = gr.Checkbox(label="🧠 Auto-analyze", value=False, scale=1)
                cookie_max        = gr.Slider(
                    label="Max", minimum=100, maximum=50000, value=50000, step=100, scale=2,
                )

        with gr.Accordion("🖼️  Import Image Files from your Computer", open=False):
            import_cat_dd = gr.Dropdown(
                label="Category",
                choices=["auto (AI decides)"] + lib_mod.get_categories(),
                value="auto (AI decides)",
            )
            import_analyze = gr.Checkbox(label="🧠 Auto-analyze", value=False)
            import_status  = gr.Textbox(label="Progress", interactive=False, lines=2)
            multi_upload   = gr.File(
                label="Images (jpg/png/webp — select thousands at once with Ctrl+A)",
                file_count="multiple",
                file_types=["image", ".jpg", ".jpeg", ".png", ".webp"],
                height=140,
            )
            file_import_btn = gr.Button("📥 Import Selected Files", variant="primary")
            zip_upload = gr.File(label="Or upload a ZIP of images", file_count="single", file_types=[".zip"])
            zip_import_btn = gr.Button("📦 Import from ZIP", variant="secondary")
            url_text = gr.Textbox(
                label="Or paste direct image URLs (one per line)", lines=4,
                placeholder="https://example.com/image.jpg",
            )
            url_import_btn = gr.Button("🌐 Import from URLs", variant="secondary")

        with gr.Accordion("🔑  Pinterest Direct API (waiting for Pinterest approval)", open=False):
            gr.HTML(_pinterest_status_html())
            import_saved_btn    = gr.Button("📌 Import ALL My Pins via API", variant="secondary")
            load_boards_btn     = gr.Button("🔄 Load My Boards", variant="secondary")
            pinterest_board_dd  = gr.Dropdown(label="Board", choices=[], value=None)
            pinterest_board_btn = gr.Button("📌 Import This Board", variant="secondary")
            pinterest_max       = gr.Slider(label="Max pins", minimum=10, maximum=10000, value=10000, step=100)
            pinterest_analyze   = gr.Checkbox(label="🧠 Auto-analyze", value=False, visible=False)

        # ── Helpers ───────────────────────────────────────────────────────────

        def _resolve_cat(cat_val: str) -> str:
            return "auto" if (not cat_val or cat_val.startswith("auto")) else cat_val

        def _do_file_import(files, cat_val, analyze):
            if not files:
                return "⚠️ No files selected."
            paths = [f.name if hasattr(f, "name") else str(f) for f in files]
            result = lib_mod.bulk_import_files(
                paths, category=_resolve_cat(cat_val),
                img_type="face", auto_analyze=bool(analyze),
            )
            n_ok, n_err = len(result["imported"]), len(result["errors"])
            msg = f"✅ Imported {n_ok} / {result['total']} file(s)."
            if not analyze:
                msg += " Run 🧠 Analyze All in the Templates tab to auto-tag & detect face/body."
            if n_err:
                msg += f"\n⚠️ {n_err} error(s): " + "; ".join(f"{p}: {e}" for p, e in result["errors"][:3])
            return msg

        def _do_zip_import(zip_file, cat_val, analyze):
            if zip_file is None:
                return "⚠️ Upload a ZIP file first."
            result = lib_mod.bulk_import_zip(
                zip_file.name if hasattr(zip_file, "name") else str(zip_file),
                category=_resolve_cat(cat_val),
                img_type="face", auto_analyze=bool(analyze),
            )
            n_ok, n_err = len(result["imported"]), len(result["errors"])
            msg = f"✅ Imported {n_ok} / {result['total']} image(s) from ZIP."
            if not analyze:
                msg += " Run 🧠 Analyze All to auto-tag & detect face/body."
            if n_err:
                msg += f"\n⚠️ {n_err} error(s): " + "; ".join(f"{p}: {e}" for p, e in result["errors"][:3])
            return msg

        def _do_url_import(urls_text, cat_val, analyze):
            urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
            if not urls:
                return "⚠️ No URLs provided."
            cat = _resolve_cat(cat_val)
            result = lib_mod.bulk_import_urls(
                urls, category="other" if cat == "auto" else cat,
                img_type="face", auto_analyze=bool(analyze),
            )
            n_ok, n_err = len(result["imported"]), len(result["errors"])
            msg = f"✅ Imported {n_ok} / {len(urls)} URL(s)."
            if n_err:
                msg += f"\n⚠️ {n_err} error(s): " + "; ".join(f"{u}: {e}" for u, e in result["errors"][:3])
            return msg

        def _do_cookie_import(cookie_val, analyze, max_count):
            sess = (cookie_val or "").strip()
            if not sess:
                yield "⚠️ Paste your _pinterest_sess cookie value first."
                return
            import threading, queue as _queue
            q: _queue.Queue = _queue.Queue()

            def _run():
                try:
                    import pinterest_cookie_import as pci
                    def _cb(cur, total, label):
                        q.put(("progress", cur, total, label))
                    result = pci.import_saved_pins_via_cookie(
                        pinterest_sess=sess,
                        max_count=int(max_count),
                        auto_analyze=bool(analyze),
                        progress_cb=_cb,
                    )
                    q.put(("done", result))
                except Exception as ex:
                    q.put(("error", str(ex)))

            yield "⏳ Connecting to Pinterest…"
            threading.Thread(target=_run, daemon=True).start()
            imported_so_far = 0
            while True:
                item = q.get()
                if item[0] == "progress":
                    _, cur, total, label = item
                    imported_so_far = cur
                    yield f"⏳ Downloading… {cur} / {total}  —  {label}"
                elif item[0] == "done":
                    result = item[1]
                    n     = len(result["imported"])
                    total = result["total"]
                    skip  = result.get("skipped", 0)
                    errs  = result.get("errors", [])
                    if total == 0 and errs:
                        first_err = errs[0][1] if isinstance(errs[0], (list, tuple)) else errs[0]
                        yield f"⚠️ {first_err}"
                        return
                    msg = f"✅ Done! Imported {n} / {total} pins."
                    if skip:
                        msg += f" ({skip} already in library)"
                    if not analyze:
                        msg += "\nGo to Templates tab → 🧠 Analyze All to auto-tag."
                    if errs:
                        sample = errs[0][1] if isinstance(errs[0], (list, tuple)) else errs[0]
                        msg += f"\n⚠️ {len(errs)} error(s): {str(sample)[:120]}"
                    yield msg
                    return
                elif item[0] == "error":
                    yield f"⚠️ Import failed: {item[1]}"
                    return

        def _do_export_import(files, analyze):
            if not files:
                yield "⚠️ Drop your Pinterest HTML files (or ZIP) above first."
                return
            # Normalise — Gradio may pass a single object or a list; each item
            # can be a NamedString, a dict, or a file-like with .name
            if not isinstance(files, list):
                files = [files]

            paths = []
            for f in files:
                if isinstance(f, dict):
                    p = f.get("name") or f.get("path") or ""
                elif hasattr(f, "name"):
                    p = f.name
                else:
                    p = str(f)
                if p:
                    paths.append(p)

            if not paths:
                yield "⚠️ Could not read file paths — try again."
                return

            import threading, queue as _queue
            q: _queue.Queue = _queue.Queue()

            def _run():
                import pinterest_import as pi

                def _cb(cur, total, label):
                    q.put(("progress", cur, total, label))

                try:
                    # Detect by extension; fall back to sniffing file contents
                    zip_paths, data_paths = [], []
                    for p in paths:
                        low = p.lower()
                        if low.endswith(".zip"):
                            zip_paths.append(p)
                        else:
                            # Peek at contents to detect ZIP or HTML
                            try:
                                with open(p, "rb") as fh:
                                    magic = fh.read(4)
                                if magic[:2] == b"PK":        # ZIP magic bytes
                                    zip_paths.append(p)
                                else:
                                    data_paths.append(p)
                            except Exception:
                                data_paths.append(p)

                    if zip_paths:
                        result = pi.import_from_export_zip(
                            zip_paths[0], auto_analyze=bool(analyze), progress_cb=_cb,
                        )
                    elif data_paths:
                        result = pi.import_from_export_files(
                            data_paths, auto_analyze=bool(analyze), progress_cb=_cb,
                        )
                    else:
                        q.put(("error", "No supported files found."))
                        return
                    q.put(("done", result))
                except Exception as ex:
                    import traceback
                    q.put(("error", f"{ex}\n{traceback.format_exc()[-400:]}"))

            yield f"⏳ Starting — {len(paths)} file(s) uploaded…"
            threading.Thread(target=_run, daemon=True).start()
            while True:
                item = q.get()
                if item[0] == "progress":
                    _, cur, total, label = item
                    yield f"⏳ Downloading… {cur} / {total}  —  {label}"
                elif item[0] == "done":
                    result = item[1]
                    n      = len(result["imported"])
                    e      = len(result["errors"])
                    total  = result["total"]
                    skipped = result.get("skipped", 0)
                    parsed  = result.get("parsed", total + skipped + n)

                    if total == 0 and n == 0:
                        already = skipped or parsed
                        if already:
                            yield (
                                f"✅ All {already} pins from this file are already in your library.\n"
                                f"If you have more HTML files in the pins/ folder, drop those in too to "
                                f"import any remaining pins. Then go to Templates → 🧠 Analyze All."
                            )
                        else:
                            yield (
                                "⚠️ No pin data found. Make sure you're uploading the HTML files "
                                "from inside the pins/ folder of your Pinterest export."
                            )
                        return

                    msg = f"✅ Done! Imported {n} / {total} pins."
                    if skipped:
                        msg += f" ({skipped} already in library)"
                    if not analyze:
                        msg += "\nGo to Templates tab → 🧠 Analyze All to auto-tag everything."
                    if e:
                        sample = "; ".join(f"{p}: {err}" for p, err in result["errors"][:2])
                        msg += f"\n⚠️ {e} error(s): {sample}"
                    yield msg
                    return
                elif item[0] == "error":
                    yield f"⚠️ Import failed: {item[1]}"
                    return

        def _load_boards():
            try:
                import pinterest_import as pi
                choices = pi.board_choices()
                if choices and choices[0].startswith("⚠️"):
                    return gr.Dropdown(choices=[], value=None), choices[0]
                return (
                    gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                    f"✅ Found {len(choices)} board(s). Pick one or use 'Import All Saved Pins'.",
                )
            except RuntimeError as e:
                return gr.Dropdown(choices=[], value=None), f"⚠️ {e}"
            except Exception as e:
                return gr.Dropdown(choices=[], value=None), f"⚠️ {e}"

        def _import_saved(max_count, analyze):
            try:
                import pinterest_import as pi
                result = pi.import_saved_pins(max_count=int(max_count), auto_analyze=bool(analyze))
                n, e = len(result["imported"]), len(result["errors"])
                msg = f"✅ Imported {n} pins from your Saved board."
                if not analyze:
                    msg += " Run 🧠 Analyze All to tag them."
                if e:
                    msg += f" ({e} errors)"
                return msg
            except RuntimeError as e:
                return f"⚠️ {e}"
            except Exception as e:
                return f"⚠️ Import failed: {e}"

        def _import_board(board_choice, max_count, analyze):
            if not board_choice:
                return "⚠️ Load boards first, then select one."
            try:
                import pinterest_import as pi
                result = pi.import_board_images(
                    board_choice=board_choice, max_count=int(max_count), auto_analyze=bool(analyze),
                )
                n, e = len(result["imported"]), len(result["errors"])
                msg = f"✅ Imported {n} / {result['total']} pins."
                if not analyze:
                    msg += " Run 🧠 Analyze All to tag them."
                if e:
                    msg += f" ({e} errors)"
                return msg
            except RuntimeError as e:
                return f"⚠️ {e}"
            except Exception as e:
                return f"⚠️ {e}"

        # ── Events ────────────────────────────────────────────────────────────
        file_import_btn.click(
            _do_file_import,
            inputs=[multi_upload, import_cat_dd, import_analyze],
            outputs=[import_status],
        )
        zip_import_btn.click(
            _do_zip_import,
            inputs=[zip_upload, import_cat_dd, import_analyze],
            outputs=[import_status],
        )
        url_import_btn.click(
            _do_url_import,
            inputs=[url_text, import_cat_dd, import_analyze],
            outputs=[import_status],
        )
        cookie_import_btn.click(
            _do_cookie_import,
            inputs=[cookie_input, cookie_analyze, cookie_max],
            outputs=[pinterest_status],
            show_progress=False,
        )
        export_import_btn.click(
            _do_export_import,
            inputs=[pinterest_export_zip, export_analyze],
            outputs=[pinterest_status],
            show_progress=False,
        )
        load_boards_btn.click(_load_boards, outputs=[pinterest_board_dd, pinterest_status])
        import_saved_btn.click(
            _import_saved, inputs=[pinterest_max, pinterest_analyze], outputs=[pinterest_status],
        )
        pinterest_board_btn.click(
            _import_board,
            inputs=[pinterest_board_dd, pinterest_max, pinterest_analyze],
            outputs=[pinterest_status],
        )
