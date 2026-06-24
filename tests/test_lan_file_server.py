import http.client
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.lan_file_server import (
    LanFileRequestHandler,
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


if __name__ == "__main__":
    unittest.main()
