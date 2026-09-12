"""
cloud_storage.py — Replit Object Storage (GCS) integration for Manhwa Generator.

Auth: Replit sidecar at http://127.0.0.1:1106 provides external-account credentials
identical to what the TypeScript @google-cloud/storage SDK uses.  In the dev
workspace the sidecar is not running so everything gracefully becomes a no-op.

All uploads are fire-and-forget daemon threads so generation is unaffected.
"""

import os
import threading
import logging
from typing import List

log = logging.getLogger(__name__)

BUCKET_ID   = os.environ.get("DEFAULT_OBJECT_STORAGE_BUCKET_ID", "")
_SIDECAR    = "http://127.0.0.1:1106"

_bucket     = None
_init_done  = False
_init_lock  = threading.Lock()


def _get_bucket():
    global _bucket, _init_done
    if _init_done:
        return _bucket
    if not BUCKET_ID:
        _init_done = True
        return None
    with _init_lock:
        if _init_done:
            return _bucket
        _init_done = True
        try:
            from google.auth.identity_pool import Credentials as _IDPoolCreds
            from google.cloud import storage as _gcs

            # Mirror the TypeScript objectStorageClient credentials exactly.
            # identity_pool.Credentials handles URL-based subject token retrieval.
            creds = _IDPoolCreds.from_info({
                "audience": "replit",
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "token_url": f"{_SIDECAR}/token",
                "credential_source": {
                    "url": f"{_SIDECAR}/credential",
                    "format": {
                        "type": "json",
                        "subject_token_field_name": "access_token",
                    },
                },
                "universe_domain": "googleapis.com",
            })

            client  = _gcs.Client(project="replit", credentials=creds)
            _bucket = client.bucket(BUCKET_ID)

            # Lightweight smoke-test: write a tiny probe object.
            _bucket.blob("__probe__").upload_from_string(b"ok")
            log.info(f"[cloud] Connected to bucket {BUCKET_ID} via sidecar")

        except Exception as exc:
            log.warning(f"[cloud] Cloud storage unavailable: {exc}")
            _bucket = None

    return _bucket


def is_available() -> bool:
    return bool(BUCKET_ID) and _get_bucket() is not None


# ─── upload counter (for UI status) ──────────────────────────────────────────

_upload_count  = 0
_upload_failed = 0
_last_upload_time: "float | None" = None
_counter_lock  = threading.Lock()


def get_status() -> str:
    """Return a one-line cloud backup status string for the UI."""
    import time as _time
    if not BUCKET_ID:
        return "☁️ No bucket configured"
    if not is_available():
        return "☁️ Backup: connecting…"
    with _counter_lock:
        n      = _upload_count
        failed = _upload_failed
        last   = _last_upload_time
    if n == 0:
        return "☁️ Backup: ready (0 uploaded)"
    ago = ""
    if last:
        secs = int(_time.time() - last)
        ago  = f" · last {secs}s ago" if secs < 120 else ""
    fail_str = f" · {failed} failed" if failed else ""
    return f"☁️ {n} backed up{ago}{fail_str}"


# ─── key helper ──────────────────────────────────────────────────────────────

def _local_to_key(local_path: str) -> str:
    return local_path.replace("\\", "/").lstrip("./")


# ─── upload helpers (fire-and-forget) ─────────────────────────────────────────

def _upload_with_retries(local_path: str, retries: int = 3) -> bool:
    """Blocking upload with exponential back-off. Returns True on success."""
    import time as _time
    global _upload_count, _upload_failed, _last_upload_time
    bucket = _get_bucket()
    if not bucket or not os.path.isfile(local_path):
        return False
    last_exc = None
    for attempt in range(retries):
        try:
            bucket.blob(_local_to_key(local_path)).upload_from_filename(local_path)
            with _counter_lock:
                _upload_count += 1
                _last_upload_time = _time.time()
            return True
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)  # 1 s, 2 s back-off
    with _counter_lock:
        _upload_failed += 1
    log.debug(f"[cloud] Upload failed after {retries} attempts {local_path}: {last_exc}")
    return False


def upload_file_blocking(local_path: str, retries: int = 3) -> bool:
    """Blocking upload — call this for critical files (images, manifest).
    Returns True on success, False if cloud is unavailable or upload failed."""
    return _upload_with_retries(local_path, retries)


def upload_file_bg(local_path: str, retries: int = 3) -> None:
    """Upload a single file to GCS in a background daemon thread, with retries.
    Use for non-critical files (thumbnails, project.json) where loss is acceptable."""
    threading.Thread(target=_upload_with_retries, args=(local_path, retries),
                     daemon=True).start()


def upload_project_meta_bg(project_dir: str) -> None:
    """Upload project.json and manifest.jsonl (background)."""
    for fname in ("project.json", "manifest.jsonl"):
        fpath = os.path.join(project_dir, fname)
        if os.path.isfile(fpath):
            upload_file_bg(fpath)


def sync_all_projects_to_cloud_bg(projects_root: str) -> None:
    """One-time background sync of ALL local project files on startup."""
    def _do():
        bucket = _get_bucket()
        if not bucket or not os.path.isdir(projects_root):
            return
        count = 0
        for root, _dirs, files in os.walk(projects_root):
            for fn in files:
                fpath = os.path.join(root, fn)
                try:
                    key  = _local_to_key(fpath)
                    blob = bucket.blob(key)
                    if not blob.exists():
                        blob.upload_from_filename(fpath)
                        count += 1
                except Exception as exc:
                    log.debug(f"[cloud] Sync upload failed {fpath}: {exc}")
        log.info(f"[cloud] Initial sync complete — {count} files uploaded")
    threading.Thread(target=_do, daemon=True).start()


# ─── restore helpers ──────────────────────────────────────────────────────────

def restore_project_meta(project_dir: str) -> bool:
    """Download project.json (+ manifest.jsonl) from cloud if missing locally."""
    bucket = _get_bucket()
    if not bucket:
        return False
    pid   = os.path.basename(project_dir)
    found = False
    for fname in ("project.json", "manifest.jsonl"):
        local = os.path.join(project_dir, fname)
        if os.path.isfile(local):
            if fname == "project.json":
                found = True
            continue
        key = f"projects/{pid}/{fname}"
        try:
            blob = bucket.blob(key)
            if blob.exists():
                os.makedirs(project_dir, exist_ok=True)
                blob.download_to_filename(local)
                if fname == "project.json":
                    found = True
        except Exception as exc:
            log.debug(f"[cloud] Restore meta failed {key}: {exc}")
    return found


def restore_missing_images_bg(project_dir: str, image_paths: List[str],
                               on_restored=None) -> None:
    """Background download of any images missing locally."""
    missing = [p for p in image_paths if not os.path.isfile(p)]
    if not missing:
        return

    def _do():
        bucket = _get_bucket()
        if not bucket:
            return
        for local_path in missing:
            key = _local_to_key(local_path)
            try:
                blob = bucket.blob(key)
                if blob.exists():
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
                    if on_restored:
                        try:
                            on_restored(local_path)
                        except Exception:
                            pass
            except Exception as exc:
                log.debug(f"[cloud] Restore image failed {key}: {exc}")

    threading.Thread(target=_do, daemon=True).start()


def delete_files_bg(local_paths: list) -> None:
    """Delete one or more files from GCS in a background thread (best-effort)."""
    def _do():
        bucket = _get_bucket()
        if not bucket:
            return
        for local_path in local_paths:
            key = _local_to_key(local_path)
            try:
                bucket.blob(key).delete()
            except Exception as exc:
                log.debug(f"[cloud] Delete failed {key}: {exc}")
    threading.Thread(target=_do, daemon=True).start()


def upload_zip_blocking(local_path: str, filename: str) -> bool:
    """Upload a zip to GCS at zips/{filename}. Blocking — returns True on success."""
    bucket = _get_bucket()
    if not bucket or not os.path.isfile(local_path):
        return False
    try:
        bucket.blob(f"zips/{filename}").upload_from_filename(local_path)
        log.info(f"[cloud] Zip uploaded: zips/{filename}")
        return True
    except Exception as exc:
        log.warning(f"[cloud] Zip upload failed {filename}: {exc}")
        return False


def download_zip_to_file(filename: str, dest_path: str) -> bool:
    """Download zips/{filename} from GCS to dest_path. Blocking — returns True on success."""
    bucket = _get_bucket()
    if not bucket:
        return False
    try:
        blob = bucket.blob(f"zips/{filename}")
        if not blob.exists():
            return False
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        return True
    except Exception as exc:
        log.warning(f"[cloud] Zip download failed {filename}: {exc}")
        return False


def list_cloud_project_ids() -> List[str]:
    """Return all project IDs that have a project.json in the bucket."""
    bucket = _get_bucket()
    if not bucket:
        return []
    try:
        ids = set()
        for blob in bucket.list_blobs(prefix="projects/"):
            parts = blob.name.split("/")
            if len(parts) >= 3 and parts[2] == "project.json":
                ids.add(parts[1])
        return sorted(ids, reverse=True)
    except Exception as exc:
        log.debug(f"[cloud] list_cloud_project_ids failed: {exc}")
        return []
