"""
Run MCP server as a module.

Usage:
    python -m ragtime.mcp
    python -m ragtime.mcp --transport stdio
"""

from ragtime.mcp.server import main

if __name__ == "__main__":
    main()
