import http.client
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.lan_file_server import (
    LanFileRequestHandler,
    default_route_source,
    is_lan_ipv4_address,
    prioritize_lan_addresses,
    resolve_within_root,
    safe_upload_name,
    unique_destination,
)


class LanFileServerHelpersTest(unittest.TestCase):
    def test_lan_ipv4_filter_accepts_only_real_private_lan_ranges(self):
        self.assertTrue(is_lan_ipv4_address("192.168.110.16"))
        self.assertTrue(is_lan_ipv4_address("10.0.0.5"))
        self.assertTrue(is_lan_ipv4_address("172.16.1.2"))
        self.assertFalse(is_lan_ipv4_address("192.18.0.1"))
        self.assertFalse(is_lan_ipv4_address("198.18.0.1"))
        self.assertFalse(is_lan_ipv4_address("127.0.0.1"))

    def test_prioritize_lan_addresses_puts_primary_route_first(self):
        addresses = ["192.168.122.1", "192.168.110.16", "198.18.0.1", "192.18.0.1"]

        self.assertEqual(prioritize_lan_addresses(addresses, primary="192.168.110.16"), ["192.168.110.16", "192.168.122.1"])

    def test_default_route_source_uses_real_lan_src_address(self):
        route_output = "default via 192.168.110.1 dev wlp60s0 proto dhcp src 192.168.110.16 metric 20600\n"

        self.assertEqual(default_route_source(route_output), "192.168.110.16")
        self.assertIsNone(default_route_source("default via 198.18.0.2 dev Meta src 198.18.0.1\n"))
        self.assertIsNone(default_route_source("default via 192.168.110.1 dev wlp60s0\n"))

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
