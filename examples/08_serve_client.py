"""HTTP client example (run after `python -m sparkforge.cli serve`)."""
import json
import urllib.request


def post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


print(post("http://127.0.0.1:8000/v1/chat/completions",
           {"messages": [{"role": "user", "content": "hello"}]}))
