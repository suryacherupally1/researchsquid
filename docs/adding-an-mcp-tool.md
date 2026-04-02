# Adding an MCP Tool

The MCP server in `squid/mcp/server.py` exposes 5 tools via FastMCP.

## To add a new tool:

1. Add a function to `squid/mcp/server.py`
2. Decorate with `@mcp.tool()`
3. The tool should call the Coordinator API or DAG directly
4. Update `TOOLS` list with the tool definition

## Tools must:
- Be thin wrappers over canonical interfaces
- Return JSON-serializable dicts
- Handle errors gracefully (never raise to MCP client)
