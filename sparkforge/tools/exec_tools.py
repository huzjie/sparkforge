"""Run-tool: execute a shell command and capture output."""
import subprocess


class RunTool:
    name = "run"

    def __init__(self, root=".", timeout=120):
        self.root = root
        self.timeout = timeout

    def __call__(self, command):
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, timeout=self.timeout,
                               capture_output=True, text=True)
            return {"code": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"code": -1, "stdout": "", "stderr": "timeout"}
