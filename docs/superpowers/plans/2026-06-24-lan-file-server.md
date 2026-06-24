# LAN File Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and start a zero-dependency LAN file-transfer server for `lan_share/`.

**Architecture:** `tools/lan_file_server.py` contains path-safety helpers, upload filename handling, a `SimpleHTTPRequestHandler` subclass, and a small CLI. Tests import helper functions and exercise one multipart upload through the handler against a temporary shared directory.

**Tech Stack:** Python standard library: `http.server`, `socketserver`, `cgi`, `urllib.parse`, `pathlib`, `unittest`, and `tempfile`.

---

## File Structure

- Create: `tools/lan_file_server.py`
  - Owns the server implementation and CLI entry point.
- Create: `tests/test_lan_file_server.py`
  - Owns focused unit and handler tests for path containment, filename cleaning, collision handling, and upload saving.
- Modify: `.gitignore`
  - Ignores `lan_share/` so transferred local files are not accidentally committed.
- Runtime directory: `lan_share/`
  - Created by the launcher and used as the only exposed directory.

### Task 1: Helper Behavior

**Files:**
- Create: `tests/test_lan_file_server.py`
- Create: `tools/lan_file_server.py`

- [ ] **Step 1: Write failing helper tests**

```python
import tempfile
import unittest
from pathlib import Path

from tools.lan_file_server import (
    resolve_within_root,
    safe_upload_name,
    unique_destination,
)


class LanFileServerHelpersTest(unittest.TestCase):
    def test_resolve_within_root_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                resolve_within_root(root, "../secret.txt")

    def test_safe_upload_name_keeps_basename_and_removes_unsafe_characters(self):
        self.assertEqual(safe_upload_name("../My File?.txt"), "My_File_.txt")

    def test_unique_destination_adds_suffix_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_bytes(b"old")
            self.assertEqual(unique_destination(root, "photo.jpg").name, "photo-1.jpg")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_lan_file_server.LanFileServerHelpersTest -v`

Expected: import failure because `tools.lan_file_server` does not exist.

- [ ] **Step 3: Write minimal helper implementation**

```python
from pathlib import Path


def resolve_within_root(root, request_path):
    root = Path(root).resolve()
    target = (root / request_path.lstrip("/")).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes shared root")
    return target


def safe_upload_name(raw_name):
    name = Path(raw_name or "upload.bin").name or "upload.bin"
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name)


def unique_destination(directory, filename):
    directory = Path(directory)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_lan_file_server.LanFileServerHelpersTest -v`

Expected: 3 tests pass.

### Task 2: HTTP Upload And Browse Server

**Files:**
- Modify: `tests/test_lan_file_server.py`
- Modify: `tools/lan_file_server.py`

- [ ] **Step 1: Write failing upload handler test**

```python
import http.client
import threading
from functools import partial
from http.server import ThreadingHTTPServer

from tools.lan_file_server import LanFileRequestHandler


class LanFileServerUploadTest(unittest.TestCase):
    def test_post_upload_saves_file_in_shared_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handler = partial(LanFileRequestHandler, shared_root=root)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                boundary = "BOUNDARY"
                body = (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="file"; filename="../note.txt"\r\n'
                    "Content-Type: text/plain\r\n\r\n"
                    "hello lan\r\n"
                    f"--{boundary}--\r\n"
                ).encode("utf-8")
                conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port)
                conn.request(
                    "POST",
                    "/upload",
                    body,
                    {"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                response = conn.getresponse()
                response.read()
                conn.close()
            finally:
                httpd.shutdown()
                thread.join(timeout=5)
                httpd.server_close()

        self.assertEqual(response.status, 303)
        self.assertEqual((root / "note.txt").read_bytes(), b"hello lan")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_lan_file_server.LanFileServerUploadTest -v`

Expected: failure because `LanFileRequestHandler` is missing.

- [ ] **Step 3: Implement request handler and CLI**

Implementation requirements:

- `LanFileRequestHandler.__init__(..., shared_root)` passes `directory=str(shared_root)` to `SimpleHTTPRequestHandler`.
- `list_directory()` renders the inherited file list plus an upload form.
- `do_POST()` accepts only `/upload`, parses multipart field `file`, writes to `unique_destination()`, and redirects to `/`.
- `run_server()` creates `lan_share/`, chooses a free port from the requested default, and prints local and LAN URLs.
- `main()` parses `--directory`, `--host`, and `--port`.

- [ ] **Step 4: Run upload test**

Run: `python3 -m unittest tests.test_lan_file_server.LanFileServerUploadTest -v`

Expected: 1 test passes.

### Task 3: Ignore Runtime Share And Verify

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add ignored runtime directory**

```gitignore
lan_share/
```

- [ ] **Step 2: Run full focused tests**

Run: `python3 -m unittest tests.test_lan_file_server -v`

Expected: all LAN file server tests pass.

- [ ] **Step 3: Start the server**

Run: `python3 tools/lan_file_server.py --directory lan_share --host 0.0.0.0 --port 8000`

Expected: terminal prints `Local:` and at least one `LAN:` URL, and the process stays running.

- [ ] **Step 4: Verify HTTP response**

Run from a second shell while the server is running: `curl -I http://127.0.0.1:<port>/`

Expected: HTTP `200 OK`.
