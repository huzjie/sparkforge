"""HTTP serving smoke test (in-process)."""
import json
import threading
import unittest
import urllib.request

from sparkforge.config import Config
from sparkforge.serving import make_handler, run
from http.server import HTTPServer


class TestServing(unittest.TestCase):
    def test_handler_health(self):
        from sparkforge.engine import SparkEngine
        cfg = Config()
        eng = SparkEngine(cfg)
        handler = make_handler(eng)
        srv = HTTPServer(("127.0.0.1", 0), handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.handle_request, daemon=True)
        t.start()
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=10) as r:
            data = json.loads(r.read())
        self.assertEqual(data["status"], "ok")
        srv.server_close()


if __name__ == "__main__":
    unittest.main()
