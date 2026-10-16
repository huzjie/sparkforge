"""Optional integrations (LangChain / MCP) — gracefully degrade offline."""
from .langchain import LangChainAdapter
from .mcp import MCPToolAdapter

__all__ = ["LangChainAdapter", "MCPToolAdapter"]
