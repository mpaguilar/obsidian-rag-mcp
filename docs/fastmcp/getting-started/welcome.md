> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Welcome to FastMCP

> The fast, Pythonic way to build MCP servers, clients, and applications.

<video autoPlay muted loop playsInline className="rounded-2xl block dark:hidden" src="https://mintcdn.com/fastmcp/-fU9AuXWlaP61Fuq/assets/brand/f-watercolor-waves-4-animated.mp4?fit=max&auto=format&n=-fU9AuXWlaP61Fuq&q=85&s=5eb68c6916a4c338185cae8b742f144d" data-path="assets/brand/f-watercolor-waves-4-animated.mp4" />

<video autoPlay muted loop playsInline className="rounded-2xl hidden dark:block" src="https://mintcdn.com/fastmcp/-fU9AuXWlaP61Fuq/assets/brand/f-watercolor-waves-4-dark-animated.mp4?fit=max&auto=format&n=-fU9AuXWlaP61Fuq&q=85&s=aa3158596f22114e69a601d9c68aa8e4" data-path="assets/brand/f-watercolor-waves-4-dark-animated.mp4" />

**FastMCP is the standard framework for building MCP applications.** The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) connects LLMs to tools and data. FastMCP gives you everything you need to go from prototype to production — build servers that expose capabilities, connect clients to any MCP service, and give your tools interactive UIs:

```python {1} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

## Move Fast and Make Things

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) lets you give agents access to your tools and data. But building an effective MCP application is harder than it looks.

FastMCP handles all of it. Declare a tool with a Python function, and the schema, validation, and documentation are generated automatically. Connect to a server with a URL, and transport negotiation, authentication, and protocol lifecycle are managed for you. You focus on your logic, and the MCP part just works: **with FastMCP, best practices are built in.**

**That's why FastMCP is the standard framework for working with MCP.** FastMCP 1.0 was incorporated into the official MCP Python SDK in 2024. Today, the actively maintained standalone project is downloaded a million times a day, and some version of FastMCP powers 70% of MCP servers across all languages.

FastMCP has three pillars:

<CardGroup cols={3}>
  <Card title="Servers" img="https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=2cddc3be3355623b1b81024811a9f443" href="/servers/server" data-og-width="1194" width="1194" data-og-height="895" height="895" data-path="assets/images/servers-card.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?w=280&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=280b7fa422f26d41c411c654aeaa4e8b 280w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?w=560&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=e7c159db0a2e52410a0f9fbe84aa8abf 560w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?w=840&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=0a5c59dd9cb903bbc2590dd8cc15b8dc 840w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?w=1100&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=6a4b5feeb2fea89e480a39992b27f656 1100w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?w=1650&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=4bc51a3621c07cf1059a1beeeead4d49 1650w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/servers-card.png?w=2500&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=3a9a170ca5b36975df4498566c1de697 2500w">
    Expose tools, resources, and prompts to LLMs.
  </Card>

  <Card title="Apps" img="https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=865d32af9c41cf6266a09a8a4fc03fe1" href="/apps/overview" data-og-width="1194" width="1194" data-og-height="895" height="895" data-path="assets/images/apps-card.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?w=280&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=6e4e8567a17bc86bd75c9de2726d23ae 280w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?w=560&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=0f9942a756a486a5fac75b4a12999b7f 560w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?w=840&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=cba37f52a3d9430db85b5d60be885b7c 840w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?w=1100&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=6627b3a30b481f0a36fb3c0604341b6e 1100w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?w=1650&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=338ca4f1f30d1cee51253d5baf0ee2f5 1650w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/apps-card.png?w=2500&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=76e86b9405507b007677ac640eb6627a 2500w">
    Give your tools interactive UIs rendered directly in the conversation.
  </Card>

  <Card title="Clients" img="https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=fbb306d0b3e0858afd1eef7aeacc02cf" href="/clients/client" data-og-width="1194" width="1194" data-og-height="895" height="895" data-path="assets/images/clients-card.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?w=280&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=44b8112199e0366bb7ff4e608e5ac35b 280w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?w=560&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=4f5852e899f2374e7d3f37b76a6fcedf 560w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?w=840&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=2c7796159d1b2db8b78884111a2538db 840w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?w=1100&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=b0d5978870078ba8efe351f247ac554c 1100w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?w=1650&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=8097b4fe75323cde23f152d881498a2b 1650w, https://mintcdn.com/fastmcp/uaPe2cZCul164Sax/assets/images/clients-card.png?w=2500&fit=max&auto=format&n=uaPe2cZCul164Sax&q=85&s=039a3bfa03648e310a63e875f05ef01a 2500w">
    Connect to any MCP server — local or remote, programmatic or CLI.
  </Card>
</CardGroup>

**[Servers](/servers/server)** wrap your Python functions into MCP-compliant tools, resources, and prompts. **[Clients](/clients/client)** connect to any server with full protocol support. And **[Apps](/apps/overview)** give your tools interactive UIs rendered directly in the conversation.

Ready to build? Start with the [installation guide](/getting-started/installation) or jump straight to the [quickstart](/getting-started/quickstart). When you're ready to deploy, [Prefect Horizon](https://www.prefect.io/horizon) offers free hosting for FastMCP users.

FastMCP is made with 💙 by [Prefect](https://www.prefect.io/).

<Tip>
  **This documentation reflects FastMCP's `main` branch**, meaning it always reflects the latest development version. Features are generally marked with version badges (e.g. `New in version: 3.0.0`) to indicate when they were introduced. Note that this may include features that are not yet released.
</Tip>

## LLM-Friendly Docs

The FastMCP documentation is available in multiple LLM-friendly formats:

### MCP Server

The FastMCP docs are accessible via MCP! The server URL is `https://gofastmcp.com/mcp`.

In fact, you can use FastMCP to search the FastMCP docs:

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import asyncio
from fastmcp import Client

async def main():
    async with Client("https://gofastmcp.com/mcp") as client:
        result = await client.call_tool(
            name="SearchFastMcp",
            arguments={"query": "deploy a FastMCP server"}
        )
    print(result)

asyncio.run(main())
```

### Text Formats

The docs are also available in [llms.txt format](https://llmstxt.org/):

* [llms.txt](https://gofastmcp.com/llms.txt) - A sitemap listing all documentation pages
* [llms-full.txt](https://gofastmcp.com/llms-full.txt) - The entire documentation in one file (may exceed context windows)

Any page can be accessed as markdown by appending `.md` to the URL. For example, this page becomes `https://gofastmcp.com/getting-started/welcome.md`.

You can also copy any page as markdown by pressing "Cmd+C" (or "Ctrl+C" on Windows) on your keyboard.
