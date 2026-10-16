"""Zero-dependency stdlib HTTP server (OpenAI-style /v1 endpoints)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .config import load_config
from .engine import SparkEngine


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json({"status": "ok", "model": engine.config.model.name})
            if self.path == "/v1/models":
                return self._json({"object": "list", "data": [{"id": "sparkforge-mock"}]})
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                ln = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(ln) or b"{}")
                msgs = data.get("messages", [])
                prompt = " ".join(m.get("content", "") for m in msgs)
                reply = engine.complete(prompt, max_tokens=data.get("max_tokens", 16))
                return self._json({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                                 "finish_reason": "stop"}],
                })
            if self.path == "/v1/code":
                ln = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(ln) or b"{}")
                result = engine.code(data.get("task", ""), data.get("repo", "."),
                                     data.get("symbols", []))
                return self._json(result)
            return self._json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):
            pass

    return Handler


def run(cfg=None):
    cfg = cfg or load_config("configs/default.yaml")
    engine = SparkEngine(cfg)
    srv = HTTPServer((cfg.serving.host, cfg.serving.port), make_handler(engine))
    print("sparkforge serving on http://%s:%s" % (cfg.serving.host, cfg.serving.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
