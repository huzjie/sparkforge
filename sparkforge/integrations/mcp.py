"""MCP (Model Context Protocol) tool adapter.

Turns sparkforge's tool registry into MCP-style ``name -> (schema, handler)`` so
an MCP host can call the coding-agent tools (read_file / write_file / grep / run).
"""


class MCPToolAdapter:
    def __init__(self, registry):
        self.registry = registry

    def list_tools(self):
        return [{"name": name, "description": t["description"],
                 "inputSchema": {"type": "object", "properties": t["parameters"]}}
                for name, t in self.registry._tools.items()]

    def call(self, name, arguments):
        return self.registry.call(name, **arguments)
