# Web Interface

ITKIT provides a browser-based graphical interface (`itkit-web`) as an alternative to the PyQt desktop GUI. It offers the same preprocessing capabilities through a single-page web application powered by Flask, accessible from any modern browser without installing a desktop application.

## Requirements

Install the `web` optional dependencies:

```bash
pip install "itkit[web]"
```

This installs `flask` and `flask-cors`.

## Starting the Server

### Command-Line Entry Point

```bash
itkit-web
```

By default the server listens on `http://127.0.0.1:5050`.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` to allow remote access. |
| `--port PORT` | `5050` | TCP port to listen on. |
| `--debug` | off | Enable Flask debug / auto-reload mode. |

**Examples**

```bash
# Local access only (default)
itkit-web

# Accessible from the local network
itkit-web --host 0.0.0.0 --port 5050

# Development mode with auto-reload
itkit-web --debug
```

### Python Module

```bash
python -m itkit.web.app --host 0.0.0.0 --port 5051
```

## Interface Overview

The interface is a single-page application divided into three regions:

```plaintext
┌─────────────────────────────────────────────────────────┐
│  Header: ITKIT Web  [tool tabs]                         │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  File        │  Tool Parameters Panel                   │
│  Browser     │  (changes per selected tab)              │
│  Sidebar     │                                          │
│              ├──────────────────────────────────────────┤
│              │  Log / Progress Output                   │
└──────────────┴──────────────────────────────────────────┘
```

### File Browser Sidebar

- Displays the server-side filesystem in a tree view.
- Click any directory to expand it.
- Click a file or folder to copy its absolute path to the currently focused path field in the parameters panel.
- The current directory path is shown at the top of the sidebar and can be edited directly.

### Tool Tabs

Each tab mirrors the corresponding CLI tool. The available tools are:

| Tab | CLI tool |
|-----|----------|
| Check | `itk_check` |
| Resample | `itk_resample` |
| Orient | `itk_orient` |
| Patch | `itk_patch` |
| Augment | `itk_aug` |
| Extract | `itk_extract` |
| Convert | `itk_convert` |
| Evaluate | `itk_evaluate` |
| Combine | `itk_combine` |

### Log Panel

- Displays real-time stdout / stderr from the running tool.
- Progress bars (tqdm-style percentages) are parsed and shown in a progress indicator.
- A **Kill** button terminates the running job immediately.
- Previous output can be cleared with the **Clear** button.

## REST API

The backend exposes a thin REST API used by the frontend. Advanced users can call it directly.

### `GET /api/browse`

List the contents of a directory on the server.

| Query param | Description |
|-------------|-------------|
| `path` | Directory path (`~` is expanded). Default: `~` |

**Response**

```json
{
  "path": "/home/user/data",
  "parent": "/home/user",
  "entries": [
    {"name": "images", "path": "/home/user/data/images", "is_dir": true},
    {"name": "labels", "path": "/home/user/data/labels", "is_dir": true}
  ]
}
```

### `POST /api/run/<tool>`

Start a tool as a background job.

**Request body**

```json
{ "args": ["<mode>", "--input", "/data/dataset", "--output", "/data/out"] }
```

**Response**

```json
{ "job_id": "3f2f1a..." }
```

Only tools in the allow-list (`itk_check`, `itk_resample`, `itk_orient`, `itk_patch`, `itk_aug`, `itk_extract`, `itk_convert`, `itk_evaluate`, `itk_combine`) are accepted.

### `GET /api/stream/<job_id>`

Server-Sent Events (SSE) stream for a running job.

| Event type | Payload | Description |
|------------|---------|-------------|
| `data` | log line | stdout / stderr line from the tool |
| `progress` | integer 0–100 | percentage parsed from tqdm output |
| `heartbeat` | *(empty)* | keep-alive, sent every 30 s of silence |
| `done` | *(empty)* | process has exited |

```javascript
const es = new EventSource(`/api/stream/${jobId}`);
es.addEventListener("progress", e => console.log(e.data + "%"));
es.addEventListener("done", () => es.close());
```

### `POST /api/kill/<job_id>`

Terminate a running job (SIGKILL).

**Response**

```json
{ "status": "killed" }   // or "not_running"
```

## Security Notes

- The tool allow-list prevents arbitrary command execution via the API.
- When binding to `0.0.0.0`, ensure the port is protected by a firewall or reverse proxy if deployed on a shared or public server.
- The file browser exposes the server-side filesystem; run the server in a restricted user account or container if untrusted users have access.

## Comparison: Web vs PyQt GUI

| Feature | Web Interface | PyQt GUI |
|---------|--------------|----------|
| Requires desktop environment | No | Yes |
| Remote access | Yes (browser) | No |
| Real-time log streaming | Yes (SSE) | Yes |
| File browser | Yes (server-side) | Yes (local) |
| Supported tools | All 9 processing tools | All 9 processing tools |
| Inference (`itk_infer`) | Not yet | Yes |
| Install dependency | `flask` | `PyQt6` |
