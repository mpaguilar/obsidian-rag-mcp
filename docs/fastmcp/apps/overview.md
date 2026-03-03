> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Apps

> Give your tools interactive UIs rendered directly in the conversation.

export const VersionBadge = ({version}) => {
  return <Badge stroke size="lg" icon="gift" iconType="regular" className="version-badge">
            New in version <code>{version}</code>
        </Badge>;
};

<VersionBadge version="3.0.0" />

MCP Apps let your tools return interactive UIs — rendered in a sandboxed iframe right inside the host client's conversation. Instead of returning plain text, a tool can show a chart, a sortable table, a form, or anything you can build with HTML.

FastMCP implements the [MCP Apps extension](https://modelcontextprotocol.io/docs/extensions/apps) and provides two approaches:

## Prefab Apps (Recommended)

<VersionBadge version="3.1.0" />

<Tip>
  [Prefab](https://prefab.prefect.io) is in extremely early, active development — its API changes frequently and breaking changes can occur with any release. The FastMCP integration is equally new and under rapid development. These docs are included for users who want to work on the cutting edge; production use is not recommended. Always [pin `prefab-ui` to a specific version](/apps/prefab#getting-started) in your dependencies.
</Tip>

[Prefab UI](https://prefab.prefect.io) is a declarative UI framework for Python. You describe layouts, charts, tables, forms, and interactive behaviors using a Python DSL — and the framework compiles them to a JSON protocol that a shared renderer interprets. It started as a component library inside FastMCP and grew into its own framework with [comprehensive documentation](https://prefab.prefect.io).

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, BarChart, ChartSeries
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Dashboard")

@mcp.tool(app=True)
def sales_chart(year: int) -> PrefabApp:
    """Show sales data as an interactive chart."""
    data = get_sales_data(year)

    with Column(gap=4, css_class="p-6") as view:
        Heading(f"{year} Sales")
        BarChart(
            data=data,
            series=[ChartSeries(data_key="revenue", label="Revenue")],
            x_axis="month",
        )

    return PrefabApp(view=view)
```

Install with `pip install "fastmcp[apps]"` and see [Prefab Apps](/apps/prefab) for the integration guide.

## Custom HTML Apps

The [MCP Apps extension](https://modelcontextprotocol.io/docs/extensions/apps) is an open protocol, and you can use it directly when you need full control. You write your own HTML/CSS/JavaScript and communicate with the host via the [`@modelcontextprotocol/ext-apps`](https://github.com/modelcontextprotocol/ext-apps) SDK.

This is the right choice for custom rendering (maps, 3D, video), specific JavaScript frameworks, or capabilities beyond what the component library offers.

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP

mcp = FastMCP("Custom App")

@mcp.tool(app=AppConfig(resource_uri="ui://my-app/view.html"))
def my_tool() -> str:
    return '{"values": [1, 2, 3]}'

@mcp.resource(
    "ui://my-app/view.html",
    app=AppConfig(csp=ResourceCSP(resource_domains=["https://unpkg.com"])),
)
def view() -> str:
    return "<html>...</html>"
```

See [Custom HTML Apps](/apps/low-level) for the full reference.
