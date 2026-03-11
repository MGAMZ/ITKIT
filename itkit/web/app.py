"""ITKIT Web Interface – Flask backend.

Provides a browser-based equivalent of the PyQt GUI:
  GET  /               → serves the single-page frontend
  GET  /api/browse     → list directory contents (file-tree / modal picker)
  POST /api/run/<tool> → start an itkit CLI command, returns {job_id}
  GET  /api/stream/<job_id> → SSE stream of stdout+stderr + progress events
  POST /api/kill/<job_id>   → terminate a running job
"""
from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)

# Root directory for the /api/browse endpoint. All browsed paths must remain
# within this directory to prevent directory traversal and arbitrary
# filesystem access.
_BROWSE_ROOT_PATH = Path.cwd().resolve()
_BROWSE_ROOT = str(_BROWSE_ROOT_PATH)

# ── Job registry ─────────────────────────────────────────────────────────────
# Maps job_id -> {"proc": subprocess.Popen | None, "queue": Queue[str | None]}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Whitelist of executable tools – prevents arbitrary command execution
_ALLOWED_TOOLS = frozenset(
    {
        "itk_check",
        "itk_resample",
        "itk_orient",
        "itk_patch",
        "itk_aug",
        "itk_extract",
        "itk_convert",
        "itk_evaluate",
        "itk_combine",
    }
)

_PERCENT_RE = re.compile(r"(\d{1,3})%")


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/browse")
def browse():
    """Return the contents of a directory as JSON.

    Query params
    ------------
    path : str
        Directory path to list.  ``~`` is expanded to the user's home.
    """
    raw = request.args.get("path", "~")
    requested = os.path.normpath(os.path.expanduser(raw))

    # Resolve the requested path under the configured browse root and ensure
    # that it does not escape this root directory.
    # Strip any leading path separator so that user input cannot override the root.
    candidate_path = (_BROWSE_ROOT_PATH / requested.lstrip(os.sep)).resolve(strict=False)
    candidate = str(candidate_path)
    try:
        # Ensure that the resolved candidate directory is inside the browse root.
        if not (candidate == _BROWSE_ROOT or candidate.startswith(_BROWSE_ROOT + os.sep)):
            return jsonify({"error": "Path not allowed"}), 400
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if not os.path.isdir(candidate):
        return jsonify({"error": f"Not a directory: {requested}"}), 400

    try:
        def _sort_key(name: str) -> tuple:
            full = os.path.join(candidate, name)
            try:
                return (not os.path.isdir(full), name.lower())
            except OSError:
                return (1, name.lower())

        entries = []
        for name in sorted(os.listdir(candidate), key=_sort_key):
            full = os.path.join(candidate, name)
            try:
                is_dir = os.path.isdir(full)
            except OSError:
                is_dir = False
            # Expose a path relative to the browse root instead of the absolute filesystem path.
            rel_path = os.path.relpath(full, _BROWSE_ROOT)
            entries.append({"name": name, "path": rel_path, "is_dir": is_dir})

        # Compute the parent directory relative to the browse root. Do not expose paths
        # above the browse root; represent the root itself with None.
        parent_path = Path(candidate)
        parent_path = parent_path.parent
        if str(parent_path) == _BROWSE_ROOT or str(parent_path) == candidate:
            parent = None
        else:
            parent = os.path.relpath(str(parent_path), _BROWSE_ROOT)

        # Expose the current directory path relative to the browse root.
        current_path = os.path.relpath(candidate, _BROWSE_ROOT)
        if current_path == ".":
            current_path = ""

        return jsonify({"path": current_path, "parent": parent, "entries": entries})
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403


@app.route("/api/run/<tool>", methods=["POST"])
def run_tool(tool: str):
    """Launch an itkit CLI tool as a background job.

    Request body (JSON)
    -------------------
    args : list[str]   CLI arguments to append after the tool name.

    Returns
    -------
    {"job_id": "<uuid>"}
    """
    if tool not in _ALLOWED_TOOLS:
        return jsonify({"error": f"Tool '{tool}' is not in the allow-list"}), 400

    data = request.get_json(force=True) or {}
    args = data.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return jsonify({"error": "'args' must be a flat list of strings"}), 400

    job_id = str(uuid.uuid4())
    q: queue.Queue[str | None] = queue.Queue()
    cmd = [tool, *args]
    cmd_display = shlex.join(cmd)

    def _run() -> None:
        try:
            q.put(f"$ {cmd_display}\n")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with _jobs_lock:
                _jobs[job_id]["proc"] = proc
            for line in proc.stdout:  # type: ignore[union-attr]
                q.put(line)
            proc.wait()
            q.put(f"\n[Process exited with code {proc.returncode}]\n")
        except FileNotFoundError:
            q.put(f"[Error] Command not found: '{tool}'\n")
        except Exception as exc:  # pylint: disable=broad-except
            q.put(f"[Error] {exc}\n")
        finally:
            q.put(None)  # sentinel → signals the SSE generator to close

    with _jobs_lock:
        _jobs[job_id] = {"proc": None, "queue": q}
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream_job(job_id: str):
    """Server-Sent Events stream for a running job.

    Events
    ------
    data          Standard log line (stdout/stderr)
    progress      Integer percentage parsed from tqdm-style output
    heartbeat     Keep-alive (sent every 30 s of silence)
    done          Emitted once when the process finishes
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    q: queue.Queue[str | None] = job["queue"]

    def _generate():
        while True:
            try:
                line = q.get(timeout=30)
            except queue.Empty:
                yield "event: heartbeat\ndata: \n\n"
                continue

            if line is None:
                yield "event: done\ndata: \n\n"
                break

            # SSE requires single-line data fields; strip trailing newline
            safe = line.rstrip("\n")
            yield f"data: {safe}\n\n"

            # Emit a separate progress event when tqdm-like percentages are found
            pcts = [int(p) for p in _PERCENT_RE.findall(line)]
            if pcts:
                yield f"event: progress\ndata: {min(100, max(pcts))}\n\n"

    return Response(
        stream_with_context(_generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/kill/<job_id>", methods=["POST"])
def kill_job(job_id: str):
    """Terminate a running job by sending SIGKILL to its process."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.kill()
        return jsonify({"status": "killed"})
    return jsonify({"status": "not_running"})


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ITKIT Web Interface – browser-based preprocessing GUI"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port number (default: 5050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug / auto-reload mode",
    )
    a = parser.parse_args()
    print(f"ITKIT Web  →  http://{a.host}:{a.port}/")
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)


if __name__ == "__main__":
    main()
