"""
┌─────────────────────────────────────┐
│          Resource Manager           │
├─────────────────────────────────────┤
│ model_type          [all ▾]         │
│                           Report  ○ │
└─────────────────────────────────────┘

Scans every model folder ComfyUI knows about (including extra_model_paths.yaml)
into a SQLite library keyed by sha256. Paths are only where a model currently
lives, so renames and moves keep their info and deleted models stay on record
as missing.

Previews live in <library>/.previews and mirror the model folder layout:
    loras/Anima/Anima_Style_Meteor.safetensors -> .previews/loras/Anima/Anima_Style_Meteor.webp
"""
import os
import json
import sqlite3
import hashlib
from contextlib import closing
from datetime import datetime

from aiohttp import web
from server import PromptServer
import folder_paths
import comfy.utils
import comfy.model_management
from .. import RESOURCE_LIBRARY_DIR
from ._fg_helperfunctions import log

MODEL_EXTENSIONS = folder_paths.supported_pt_extensions | {".gguf", ".onnx"}
PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
LIBRARY_DIR = os.path.normpath(RESOURCE_LIBRARY_DIR)
LIBRARY_DB_PATH = os.path.join(LIBRARY_DIR, ".fg_resources.db")
PREVIEWS_DIR = os.path.join(LIBRARY_DIR, ".previews")
HASH_CHUNK = 8 * 1024 * 1024
EDITABLE_TEXT = ("base_model", "source_url", "trigger_words", "notes")
INFO_COLUMNS = "m.sha256, m.name, m.kind, m.base_model, m.source_url, m.trigger_words, m.notes, m.settings, m.preview"

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    sha256        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    base_model    TEXT NOT NULL DEFAULT '',
    source_url    TEXT NOT NULL DEFAULT '',
    trigger_words TEXT NOT NULL DEFAULT '',
    notes         TEXT NOT NULL DEFAULT '',
    settings      TEXT NOT NULL DEFAULT '{}',
    preview       TEXT NOT NULL DEFAULT '',
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paths (
    path      TEXT PRIMARY KEY,
    sha256    TEXT NOT NULL REFERENCES models(sha256),
    kind      TEXT NOT NULL,
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    missing   INTEGER NOT NULL DEFAULT 0,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS paths_sha256 ON paths(sha256);
"""


def open_library() -> sqlite3.Connection:
    db = sqlite3.connect(LIBRARY_DB_PATH)
    db.executescript(SCHEMA)
    return db


def model_types() -> list[str]:
    # Folders registered from extra_model_paths.yaml often have no extension set, so
    # every registered type is scanned and files are filtered by MODEL_EXTENSIONS instead.
    # Registration order puts core types ahead of aliases custom nodes add for the same folders.
    return [k for k in folder_paths.folder_names_and_paths if k != "custom_nodes"]


def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            comfy.model_management.throw_exception_if_processing_interrupted()
            sha.update(chunk)
    return sha.hexdigest()


def preview_stem(path: str, kind: str) -> str:
    """Preview location relative to PREVIEWS_DIR, without extension, mirroring the model's folder layout.
    Models outside the library folder (other drives in extra_model_paths.yaml) go under <kind>/."""
    for root in [LIBRARY_DIR, *folder_paths.folder_names_and_paths.get(kind, ([],))[0]]:
        root = os.path.normpath(root)
        if os.path.normcase(path).startswith(os.path.normcase(root) + os.sep):
            rel = path[len(root) + 1:]
            return os.path.splitext(rel if root == LIBRARY_DIR else os.path.join(kind, rel))[0]
    return os.path.join(kind, os.path.splitext(os.path.basename(path))[0])


def model_info(row) -> dict:
    sha, name, kind, base_model, source_url, trigger_words, notes, settings, preview = row
    preview_url = None
    if preview and os.path.isfile(preview_path := os.path.join(PREVIEWS_DIR, preview)):
        preview_url = f"/fg/library/preview/{sha}?v={os.stat(preview_path).st_mtime_ns}"
    return {
        "sha256": sha,
        "name": name,
        "kind": kind,
        "base_model": base_model,
        "source_url": source_url,
        "trigger_words": trigger_words,
        "notes": notes,
        "settings": json.loads(settings),
        "preview_url": preview_url,
    }


class FG_ResourceManager:
    """Hashes and records every model file into the resource library."""

    def __init__(self):
        self.NODE_NAME = "Resource Manager"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["all"] + sorted(model_types()), {"tooltip": "📂 Model folder type to scan. Unchanged files are not re-hashed."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Report",)
    FUNCTION = "scan"
    CATEGORY = "Farrenzo's Garbage/Utils"
    DESCRIPTION = "Scans model folders into the resource library and reports new, moved and missing models."
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def scan(self, model_type):
        kinds = model_types() if model_type == "all" else [model_type]

        files = {}
        for kind in kinds:
            for root in folder_paths.get_folder_paths(kind):
                rel_paths, _ = folder_paths.recursive_search(root, excluded_dir_names=[".git"])
                for rel in rel_paths:
                    if os.path.splitext(rel)[1].lower() in MODEL_EXTENSIONS:
                        files.setdefault(os.path.normpath(os.path.join(root, rel)), kind)

        stamp = datetime.now().isoformat()
        hashed = unchanged = 0
        new_models, relocated = [], []
        pbar = comfy.utils.ProgressBar(len(files))
        with closing(open_library()) as db:
            for path, kind in files.items():
                st = os.stat(path)
                row = db.execute("SELECT sha256, size, mtime_ns FROM paths WHERE path = ?", (path,)).fetchone()
                if row and row[1] == st.st_size and row[2] == st.st_mtime_ns:
                    sha = row[0]
                    unchanged += 1
                else:
                    log(f"{self.NODE_NAME}: Hashing {path} ({st.st_size / 2**30:.2f} GB)")
                    sha = sha256_file(path)
                    hashed += 1

                name = os.path.splitext(os.path.basename(path))[0]
                if db.execute("SELECT 1 FROM models WHERE sha256 = ?", (sha,)).fetchone() is None:
                    new_models.append(name)
                elif row is None:
                    relocated.append((path, sha))

                db.execute(
                    "INSERT INTO models (sha256, name, kind, first_seen, last_seen) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(sha256) DO UPDATE SET name = excluded.name, kind = excluded.kind, last_seen = excluded.last_seen",
                    (sha, name, kind, stamp, stamp),
                )
                db.execute(
                    "INSERT INTO paths (path, sha256, kind, size, mtime_ns, missing, last_seen) VALUES (?, ?, ?, ?, ?, 0, ?) "
                    "ON CONFLICT(path) DO UPDATE SET sha256 = excluded.sha256, kind = excluded.kind, size = excluded.size, "
                    "mtime_ns = excluded.mtime_ns, missing = 0, last_seen = excluded.last_seen",
                    (path, sha, kind, st.st_size, st.st_mtime_ns, stamp),
                )
                db.commit()
                pbar.update(1)

            kind_marks = ", ".join("?" * len(kinds))
            gone_where = f"missing = 0 AND last_seen != ? AND kind IN ({kind_marks})"
            # A known model at a new path was moved/renamed if its old path vanished; otherwise it's a copy (see duplicates).
            moved = [path for path, sha in relocated if db.execute(
                f"SELECT 1 FROM paths WHERE sha256 = ? AND {gone_where}", (sha, stamp, *kinds)
            ).fetchone()]
            # A path that vanished only counts as missing when its model wasn't found anywhere else this scan.
            gone = [r[0] for r in db.execute(
                f"SELECT path FROM paths p WHERE {gone_where} "
                "AND NOT EXISTS (SELECT 1 FROM paths q WHERE q.sha256 = p.sha256 AND q.last_seen = ?) ORDER BY path",
                (stamp, *kinds, stamp),
            )]
            db.execute(f"UPDATE paths SET missing = 1 WHERE {gone_where}", (stamp, *kinds))

            # Keep previews named after their model when it was renamed or moved.
            for path in moved:
                sha, preview = db.execute(
                    "SELECT m.sha256, m.preview FROM paths p JOIN models m ON m.sha256 = p.sha256 WHERE p.path = ?", (path,)
                ).fetchone()
                if not preview:
                    continue
                new_preview = preview_stem(path, files[path]) + os.path.splitext(preview)[1]
                old_file = os.path.join(PREVIEWS_DIR, preview)
                if new_preview != preview and os.path.isfile(old_file):
                    new_file = os.path.join(PREVIEWS_DIR, new_preview)
                    os.makedirs(os.path.dirname(new_file), exist_ok=True)
                    os.replace(old_file, new_file)
                    db.execute("UPDATE models SET preview = ? WHERE sha256 = ?", (new_preview, sha))
            db.commit()

            duplicates = db.execute(
                "SELECT m.name, COUNT(*) FROM paths p JOIN models m ON m.sha256 = p.sha256 "
                "WHERE p.missing = 0 GROUP BY p.sha256 HAVING COUNT(*) > 1 ORDER BY m.name"
            ).fetchall()

        report = [
            f"Scanned {len(files)} files in {len(kinds)} folder type(s): {hashed} hashed, {unchanged} unchanged.",
            f"New models: {len(new_models)}",
            f"Moved or renamed: {len(moved)}",
            *(f"  {p}" for p in moved),
            f"Newly missing: {len(gone)}",
            *(f"  {p}" for p in gone),
            f"Duplicates on disk: {len(duplicates)}",
            *(f"  {name} x{count}" for name, count in duplicates),
            f"Library: {LIBRARY_DB_PATH}",
        ]
        log(f"{self.NODE_NAME}: {report[0]}", "finish")
        return ("\n".join(report),)


routes = PromptServer.instance.routes


@routes.get("/fg/library/kind/{kind}")
async def library_kind(request):
    """Returns { model name as the loader lists it: info } for every present model of a folder type."""
    kind = folder_paths.map_legacy(request.match_info["kind"])
    if kind not in folder_paths.folder_names_and_paths:
        return web.json_response({"error": f"Unknown model type {kind}"}, status=404)
    roots = [os.path.normcase(os.path.normpath(r)) + os.sep for r in folder_paths.get_folder_paths(kind)]
    out = {}
    with closing(open_library()) as db:
        rows = db.execute(f"SELECT p.path, {INFO_COLUMNS} FROM paths p JOIN models m ON m.sha256 = p.sha256 WHERE p.missing = 0")
        for path, *info in rows:
            for root in roots:
                if os.path.normcase(path).startswith(root):
                    out[path[len(root):]] = model_info(info)
                    break
    return web.json_response(out)


@routes.post("/fg/library/model/{sha256:[0-9a-f]{64}}")
async def library_update(request):
    sha = request.match_info["sha256"]
    body = await request.json()
    updates = {k: body[k] for k in EDITABLE_TEXT if isinstance(body.get(k), str)}
    if isinstance(body.get("settings"), dict):
        updates["settings"] = json.dumps(body["settings"])
    with closing(open_library()) as db:
        if updates:
            db.execute(f"UPDATE models SET {', '.join(f'{k} = ?' for k in updates)} WHERE sha256 = ?", (*updates.values(), sha))
            db.commit()
        row = db.execute(f"SELECT {INFO_COLUMNS} FROM models m WHERE m.sha256 = ?", (sha,)).fetchone()
    if row is None:
        return web.json_response({"error": "Unknown model"}, status=404)
    return web.json_response(model_info(row))


@routes.post("/fg/library/model/{sha256:[0-9a-f]{64}}/preview")
async def library_upload_preview(request):
    sha = request.match_info["sha256"]
    image = (await request.post()).get("image")
    ext = os.path.splitext(getattr(image, "filename", ""))[1].lower()
    if ext not in PREVIEW_EXTS:
        return web.json_response({"error": f"Preview must be one of {', '.join(PREVIEW_EXTS)}"}, status=400)
    with closing(open_library()) as db:
        model = db.execute("SELECT name, kind, preview FROM models WHERE sha256 = ?", (sha,)).fetchone()
        if model is None:
            return web.json_response({"error": "Unknown model"}, status=404)
        name, kind, old_preview = model
        location = db.execute("SELECT path, kind FROM paths WHERE sha256 = ? ORDER BY missing LIMIT 1", (sha,)).fetchone()
        preview = (preview_stem(*location) if location else os.path.join(kind, name)) + ext
        dest = os.path.join(PREVIEWS_DIR, preview)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(image.file.read())
        if old_preview and old_preview != preview and os.path.isfile(old_file := os.path.join(PREVIEWS_DIR, old_preview)):
            os.remove(old_file)
        db.execute("UPDATE models SET preview = ? WHERE sha256 = ?", (preview, sha))
        db.commit()
        row = db.execute(f"SELECT {INFO_COLUMNS} FROM models m WHERE m.sha256 = ?", (sha,)).fetchone()
    return web.json_response(model_info(row))


@routes.get("/fg/library/preview/{sha256:[0-9a-f]{64}}")
async def library_preview(request):
    with closing(open_library()) as db:
        row = db.execute("SELECT preview FROM models WHERE sha256 = ?", (request.match_info["sha256"],)).fetchone()
    if not row or not row[0] or not os.path.isfile(path := os.path.join(PREVIEWS_DIR, row[0])):
        return web.json_response({"error": "Not found"}, status=404)
    return web.FileResponse(path)
