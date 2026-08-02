# MCP Guide

AgentNet's MCP integration is descriptor-first. `MCPServer` describes an MCP
server and its tools, `MCPRegistry` stores server descriptors and allowlists,
and `MCPToolAdapter` exposes allowed MCP tools through `ToolRegistry`.

## Register Descriptors

MCP descriptors do not launch server commands. Commands and environment values
are configuration records only, and environment values are omitted from
serialization.

```python
import agentnet as an

server = an.MCPServer(
    name="github",
    command=["npx", "server-github"],
    tools=[
        an.MCPToolDescriptor("search_repos", description="Search repositories"),
        an.MCPToolDescriptor("read_issue", description="Read one issue"),
    ],
    env={"GITHUB_TOKEN": "kept outside serialized descriptors"},
)

registry = an.MCPRegistry()
registry.register(server, allow_tools=["search_repos"])

print(registry.qualified_tool_names)
```

The registry enforces allowlists before a tool can be resolved. Use qualified
tool names such as `github.search_repos` in agents.

## Adapt Tools

`MCPToolAdapter` requires an injected client. The client supplies `call_tool` or
`acall_tool`; AgentNet does not start MCP processes from descriptors.

```python
import agentnet as an


class MCPClient:
    def call_tool(self, server_name, tool_name, arguments):
        return {"server": server_name, "tool": tool_name, "arguments": arguments}


tools = an.ToolRegistry()
adapter = an.MCPToolAdapter(registry, MCPClient())
adapter.register_tool(tools, "github.search_repos")

agent = an.ReActAgent(
    "researcher",
    llms=[an.FakeLLM(["ok"], name="strong")],
    tools=["github.search_repos"],
)

result = tools.execute(
    "github.search_repos",
    {"query": "agentnet"},
    agent=agent,
    context=an.RunContext(run_id="mcp-run"),
)
```

## Approvals

Pin descriptor approval to the descriptor hash when tool definitions require
review:

```python
import agentnet as an

approvals = an.MCPApprovalStore()
descriptor = registry.get_tool("github.search_repos")
approvals.approve("github.search_repos", descriptor, approved_by="platform")

adapter = an.MCPToolAdapter(
    registry,
    MCPClient(),
    approvals=approvals,
    require_approval=True,
)
```

Approvals fail if the descriptor changes after review. This is useful for tools
with broad access or side effects.

## Serialization

Save MCP descriptors with artifacts to preserve tool requirements:

```python
import agentnet as an

an.save(
    agent,
    "researcher.agentnet",
    name="researcher",
    tools=tools,
    mcp_registry=registry,
)
```

Loading still requires live MCP dependencies to be injected by the application.
