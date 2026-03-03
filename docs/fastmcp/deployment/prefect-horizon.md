> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Prefect Horizon

> The MCP platform from the FastMCP team

[Prefect Horizon](https://www.prefect.io/horizon) is a platform for deploying and managing MCP servers. Built by the FastMCP team at [Prefect](https://www.prefect.io), Horizon provides managed hosting, authentication, access control, and a registry of MCP capabilities.

Horizon includes a **free personal tier for FastMCP users**, making it the fastest way to get a secure, production-ready server URL with built-in OAuth authentication.

<Info>
  Horizon is free for personal projects. Enterprise governance features are available for teams deploying to thousands of users.
</Info>

## The Platform

Horizon is organized into four integrated pillars:

* **Deploy**: Managed hosting with CI/CD, scaling, monitoring, and rollbacks. Push code and get a live, governed endpoint in 60 seconds.
* **Registry**: A central catalog of MCP servers across your organization—first-party, third-party, and curated remix servers composed from multiple sources.
* **Gateway**: Role-based access control, authentication, and audit logs. Define what agents can see and do at the tool level.
* **Agents**: A permissioned chat interface for interacting with any MCP server or curated combination of servers.

This guide focuses on **Horizon Deploy**, the managed hosting layer that gives you the fastest path from a FastMCP server to a production URL.

## Prerequisites

To use Horizon, you'll need a [GitHub](https://github.com) account and a GitHub repo containing a FastMCP server. If you don't have one yet, Horizon can create a starter repo for you during onboarding.

Your repo can be public or private, but must include at least a Python file containing a FastMCP server instance.

<Tip>
  To verify your file is compatible with Horizon, run `fastmcp inspect <file.py:server_object>` to see what Horizon will see when it runs your server.
</Tip>

If you have a `requirements.txt` or `pyproject.toml` in the repo, Horizon will automatically detect your server's dependencies and install them. Your file *can* have an `if __name__ == "__main__"` block, but it will be ignored by Horizon.

For example, a minimal server file might look like:

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"
```

## Getting Started

There are just three steps to deploying a server to Horizon:

### Step 1: Select a Repository

Visit [horizon.prefect.io](https://horizon.prefect.io) and sign in with your GitHub account. Connect your GitHub account to grant Horizon access to your repositories, then select the repo you want to deploy.

<img src="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=0261c7638aad85d04a4fee5598e8a662" alt="Horizon repository selection" data-og-width="2758" width="2758" data-og-height="1930" height="1930" data-path="assets/images/horizon/select-repo.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?w=280&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=0fcb506158c4eb47f1f6457fd1e8ac3c 280w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?w=560&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=3f5d97028a97a6d830fdd670880e07c3 560w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?w=840&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=dfb39f7ec4b32f592fab3330149aec53 840w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?w=1100&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=71f00c5e76ae9fcfd7896235c855c4f8 1100w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?w=1650&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=cfbc4d5299d276281b59e5c19e612193 1650w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/select-repo.png?w=2500&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=93e20eb6f8d21a51950918fbf5c32ea6 2500w" />

### Step 2: Configure Your Server

Next, you'll configure how Horizon should build and deploy your server.

<img src="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=65f08feefd47d66fc73a296246bbf1b1" alt="Horizon server configuration" data-og-width="2758" width="2758" data-og-height="1930" height="1930" data-path="assets/images/horizon/configure-server.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?w=280&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=bcd68d209c43aff1e54caf2e70791279 280w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?w=560&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=0f608d9ba5498d223cd695efebd39521 560w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?w=840&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=1d040da9300d5ec48524cab93c085539 840w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?w=1100&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=4657affea97a6205dae6d7f2f9fc1828 1100w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?w=1650&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=4a075cc2774dac4c484aed9bd0900537 1650w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/configure-server.png?w=2500&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=44f0a736b7fa3b16330c2dd95d2a3e5a 2500w" />

The configuration screen lets you specify:

* **Server name**: A unique name for your server. This determines your server's URL.
* **Description**: A brief description of what your server does.
* **Entrypoint**: The Python file containing your FastMCP server (e.g., `main.py`). This field has the same syntax as the `fastmcp run` command—use `main.py:mcp` to specify a specific object in the file.
* **Authentication**: When enabled, only authenticated users in your organization can connect. Horizon handles all the OAuth complexity for you.

Horizon will automatically detect your server's Python dependencies from either a `requirements.txt` or `pyproject.toml` file.

### Step 3: Deploy and Connect

Click **Deploy Server** and Horizon will clone your repository, build your server, and deploy it to a unique URL—typically in under 60 seconds.

<img src="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=9df5871545f94c370ae1758f4de3b185" alt="Horizon deployment view showing live server" data-og-width="2758" width="2758" data-og-height="1930" height="1930" data-path="assets/images/horizon/deployment-live.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?w=280&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=88447209212970644ce4fde7a87c1485 280w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?w=560&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=3ad035307e818e51b179ef2691006f8d 560w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?w=840&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=f2c1d00412ac38d44a5bac972b41c8b7 840w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?w=1100&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=11894a4f1be90cb0d3e8bd5bf2112bc1 1100w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?w=1650&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=6327988e3a2371c829d6c5dbbf46ac35 1650w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/deployment-live.png?w=2500&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=3f4dfb539811dbddec4340a05ba038b5 2500w" />

Once deployed, your server is accessible at a URL like:

```
https://your-server-name.fastmcp.app/mcp
```

Horizon monitors your repo and redeploys automatically whenever you push to `main`. It also builds preview deployments for every PR, so you can test changes before they go live.

## Testing Your Server

Horizon provides two ways to verify your server is working before connecting external clients.

### Inspector

The Inspector gives you a structured view of everything your server exposes—tools, resources, and prompts. You can click any tool, fill in the inputs, execute it, and see the output. This is useful for systematically validating each capability and debugging specific behaviors.

### ChatMCP

For quick end-to-end testing, ChatMCP lets you interact with your server conversationally. It uses a fast model optimized for rapid iteration—you can verify the server works, test tool calls in context, and confirm the overall behavior before sharing it with others.

<img src="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=819c704307a796a29c9b08b5eca31881" alt="Horizon ChatMCP interface" data-og-width="2758" width="2758" data-og-height="1930" height="1930" data-path="assets/images/horizon/chat.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?w=280&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=d18c8dcb4ce5e6c3fb59955c81827dbd 280w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?w=560&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=227b7d736d3eb28d14c96e3cc3a1ac25 560w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?w=840&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=e5b543184fd0415dbcf20e293397d4a8 840w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?w=1100&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=4ec4cc1cd39a4706f367febae1b61a66 1100w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?w=1650&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=c2a2a097081a89a8f5229d5f59f4dc9a 1650w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/chat.png?w=2500&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=06c32801c8f770819f6021bddfbf9ba7 2500w" />

ChatMCP is designed for testing, not as a daily work environment. Once you've confirmed your server works, you can copy connection snippets for Claude Desktop, Cursor, Claude Code, and other MCP clients—or use the FastMCP client library to connect programmatically.

## Horizon Agents

Beyond testing individual servers, Horizon lets you create **Agents**—chat interfaces backed by one or more MCP servers. While ChatMCP tests a single server, Agents let you compose capabilities from multiple servers into a unified experience.

<img src="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=bb4fe7f77cd4a560d02f9f7865c8840f" alt="Horizon Agent configuration" data-og-width="2758" width="2758" data-og-height="1930" height="1930" data-path="assets/images/horizon/agent-detail.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?w=280&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=c206087b563ac17fef950cb28b44d67f 280w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?w=560&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=cd957f62db7267ef36236bb3f92f2264 560w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?w=840&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=a8b1a430eccafb8e783cadf029a3d3db 840w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?w=1100&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=4b7eaea00c57158ccf112d15617d6857 1100w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?w=1650&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=ad8cc4a2a8f26497e9db4437570e82d6 1650w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-detail.png?w=2500&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=c72378a6a0721d415c0cab806eecf5d1 2500w" />

To create an agent:

1. Navigate to **Agents** in the sidebar
2. Click **Create Agent** and give it a name and description
3. Add MCP servers to the agent—these can be servers you've deployed to Horizon or external servers in the registry

Once configured, you can chat with your agent directly in Horizon:

<img src="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=63371310b14c6224de18f0cc0a0af123" alt="Chatting with a Horizon Agent" data-og-width="2758" width="2758" data-og-height="1930" height="1930" data-path="assets/images/horizon/agent-chat.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?w=280&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=faa6b2986ca19bd9a306224d4e21712a 280w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?w=560&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=4241564b710f61f306b3c9971ac1a00a 560w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?w=840&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=a9d6f44e4ea7a8be4814659fcd46a8f0 840w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?w=1100&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=38c7f58b1b08410cceefd7bd71f3f1f2 1100w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?w=1650&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=87f15ab95d0ba2523c3f5965b7d10381 1650w, https://mintcdn.com/fastmcp/kmZMfDguWt_dAKty/assets/images/horizon/agent-chat.png?w=2500&fit=max&auto=format&n=kmZMfDguWt_dAKty&q=85&s=55a8574c8b73bad340a7db2e801aa2e2 2500w" />

Agents are useful for creating purpose-built interfaces that combine tools from different servers. For example, you might create an agent that has access to both your company's internal data server and a general-purpose utilities server.
