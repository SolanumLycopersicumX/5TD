#!/usr/bin/env python3
import argparse
import email.policy
import html
import io
import os
import posixpath
import socket
import sys
import urllib.parse
from email.parser import BytesParser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_SHARE_DIR = "lan_share"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def resolve_within_root(root, request_path):
    root = Path(root).resolve()
    target = (root / request_path.lstrip("/")).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes shared root")
    return target


def safe_upload_name(raw_name):
    name = Path(raw_name or "upload.bin").name or "upload.bin"
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name)
    return safe or "upload.bin"


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


def lan_ipv4_addresses():
    addresses = set()
    try:
        hostname = socket.gethostname()
        for address in socket.gethostbyname_ex(hostname)[2]:
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass

    return sorted(addresses)


class LanFileRequestHandler(SimpleHTTPRequestHandler):
    server_version = "LanFileServer/1.0"

    def __init__(self, *args, shared_root=None, **kwargs):
        self.shared_root = Path(shared_root or DEFAULT_SHARE_DIR).resolve()
        super().__init__(*args, directory=str(self.shared_root), **kwargs)

    def translate_path(self, path):
        raw_path = urllib.parse.urlsplit(path).path
        raw_path = urllib.parse.unquote(raw_path, errors="surrogatepass")
        raw_path = posixpath.normpath(raw_path)
        parts = [part for part in raw_path.split("/") if part]

        target = self.shared_root
        for part in parts:
            _, part = os.path.splitdrive(part)
            _, part = os.path.split(part)
            if part in (os.curdir, os.pardir, ""):
                continue
            target = target / part
        return str(resolve_within_root(self.shared_root, str(target.relative_to(self.shared_root))))

    def list_directory(self, path):
        try:
            entries = sorted(Path(path).iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "No permission to list directory")
            return None

        display_path = html.escape(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path))
        rows = []
        if urllib.parse.urlsplit(self.path).path.rstrip("/"):
            rows.append('<li><a href="../">../</a></li>')
        for entry in entries:
            name = entry.name + ("/" if entry.is_dir() else "")
            href = urllib.parse.quote(name)
            rows.append(f'<li><a href="{href}">{html.escape(name)}</a></li>')

        body = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LAN File Server</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 32px auto; padding: 0 16px; color: #202124; }}
header {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; border-bottom: 1px solid #ddd; padding-bottom: 16px; }}
h1 {{ font-size: 24px; margin: 0; }}
form {{ margin: 24px 0; padding: 16px; border: 1px solid #ddd; border-radius: 8px; display: flex; gap: 12px; flex-wrap: wrap; }}
button {{ padding: 8px 14px; cursor: pointer; }}
ul {{ line-height: 1.9; padding-left: 24px; }}
a {{ color: #0756a6; }}
.path {{ color: #5f6368; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<header>
<h1>LAN File Server</h1>
<div class="path">{display_path}</div>
</header>
<form method="post" action="/upload" enctype="multipart/form-data">
<input type="file" name="file" required>
<button type="submit">上传</button>
</form>
<ul>
{''.join(rows) if rows else '<li>这个目录现在是空的。</li>'}
</ul>
</body>
</html>
'''.encode("utf-8")
        encoded = io.BytesIO(body)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return encoded

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/upload":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown upload endpoint")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "Empty upload")
            return

        body = self.rfile.read(length)
        message = BytesParser(policy=email.policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
            b"MIME-Version: 1.0\r\n\r\n" + body
        )
        if not message.is_multipart():
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid multipart upload")
            return

        upload_part = None
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "file":
                upload_part = part
                break
        if upload_part is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing file field")
            return

        filename = safe_upload_name(upload_part.get_filename() or "upload.bin")
        data = upload_part.get_payload(decode=True) or b""
        destination = unique_destination(self.shared_root, filename)
        destination.write_bytes(data)

        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()


def make_server(host, port, shared_root):
    handler = partial(LanFileRequestHandler, shared_root=shared_root)
    last_error = None
    for candidate_port in range(port, port + 100):
        try:
            return ThreadingHTTPServer((host, candidate_port), handler)
        except OSError as exc:
            last_error = exc
    raise OSError(f"no free port found from {port} to {port + 99}") from last_error


def run_server(directory=DEFAULT_SHARE_DIR, host=DEFAULT_HOST, port=DEFAULT_PORT):
    shared_root = Path(directory).resolve()
    shared_root.mkdir(parents=True, exist_ok=True)
    httpd = make_server(host, port, shared_root)
    actual_port = httpd.server_port

    print(f"Serving: {shared_root}", flush=True)
    print(f"Local:   http://127.0.0.1:{actual_port}/", flush=True)
    addresses = lan_ipv4_addresses()
    if addresses:
        for address in addresses:
            print(f"LAN:     http://{address}:{actual_port}/", flush=True)
    else:
        print(f"LAN:     http://<your-lan-ip>:{actual_port}/", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        httpd.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Share a local directory with upload/download over LAN.")
    parser.add_argument("--directory", default=DEFAULT_SHARE_DIR, help="Directory to share. Default: lan_share")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Starting port. Default: 8000")
    args = parser.parse_args(argv)
    run_server(args.directory, args.host, args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
