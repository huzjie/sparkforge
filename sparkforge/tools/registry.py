"""A small tool registry with JSON-schema-ish signatures."""
from dataclasses import dataclass, field


def tool(name, description, parameters):
    """Decorator/factory: register a callable as a tool."""
    def deco(fn):
        fn.__tool_name__ = name
        fn.__tool_desc__ = description
        fn.__tool_params__ = parameters
        return fn
    return deco


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, fn, description="", parameters=None):
        self._tools[name] = {"fn": fn, "description": description, "parameters": parameters or {}}

    def call(self, name, **kwargs):
        if name not in self._tools:
            raise KeyError("unknown tool: %s" % name)
        return self._tools[name]["fn"](**kwargs)

    def schema(self):
        return [{"name": n, "description": t["description"], "parameters": t["parameters"]}
                for n, t in self._tools.items()]

    def names(self):
        return list(self._tools.keys())
