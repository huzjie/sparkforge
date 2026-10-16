"""Tiny logger (zero-dependency) writing to stderr + optional file."""
import sys
import time


class Logger:
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self, name="sparkforge", level="INFO"):
        self.name = name
        self.level = self.LEVELS.get(level, 20)

    def _log(self, lvl, msg):
        if self.LEVELS[lvl] < self.level:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print("[%s] %s %s: %s" % (ts, lvl, self.name, msg), file=sys.stderr, flush=True)

    def debug(self, msg):
        self._log("DEBUG", msg)

    def info(self, msg):
        self._log("INFO", msg)

    def warn(self, msg):
        self._log("WARN", msg)

    def error(self, msg):
        self._log("ERROR", msg)


def get_logger(name="sparkforge", level="INFO"):
    return Logger(name, level)
