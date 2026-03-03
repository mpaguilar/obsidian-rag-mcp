> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# AuthKit 🤝 FastMCP

> Secure your FastMCP server with AuthKit by WorkOS

export const VersionBadge = ({version}) => {
  return <Badge stroke size="lg" icon="gift" iconType="regular" className="version-badge">
            New in version <code>{version}</code>
        </Badge>;
};

<VersionBadge version="2.11.0" />

This guide shows you how to secure your FastMCP server using WorkOS's **AuthKit**, a complete authentication and user management solution. This integration uses the [**Remote OAuth**](/servers/auth/remote-oauth) pattern, where AuthKit handles user login and your FastMCP server validates the tokens.

<Warning>
  AuthKit does not currently support [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707.html) resource indicators, so FastMCP cannot validate that tokens were issued for the specific resource server. If you need resource-specific audience validation, consider using [WorkOSProvider](/integrations/workos) (OAuth proxy pattern) instead.
</Warning>

## Configuration

### Prerequisites

Before you begin, you will need:

1. A **[WorkOS Account](https://workos.com/)** and a new **Project**.
2. An **[AuthKit](https://www.authkit.com/)** instance configured within your WorkOS project.
3. Your FastMCP server's URL (can be localhost for development, e.g., `http://localhost:8000`).

### Step 1: AuthKit Configuration

In your WorkOS Dashboard, enable AuthKit and configure the following settings:

<Steps>
  <Step title="Enable Dynamic Client Registration">
    Go to **Applications → Configuration** and enable **Dynamic Client Registration**. This allows MCP clients register with your application automatically.

        <img src="https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=5849352618ef89da08cf452c5dcf38a8" alt="Enable Dynamic Client Registration" data-og-width="2644" width="2644" data-og-height="1588" height="1588" data-path="integrations/images/authkit/enable_dcr.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?w=280&fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=afe3af1599ffac48411edaa7eb8ee35e 280w, https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?w=560&fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=fa6e286fd2e98a984dbed9dd30375e07 560w, https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?w=840&fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=26ba396eef845948a2bbe7405716a419 840w, https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?w=1100&fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=5270b0b09891741087494f2819be6f19 1100w, https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?w=1650&fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=604955d22d5556d3464f5b4919664c41 1650w, https://mintcdn.com/fastmcp/hUosZw7ujHZFemrG/integrations/images/authkit/enable_dcr.png?w=2500&fit=max&auto=format&n=hUosZw7ujHZFemrG&q=85&s=a4e1627dd14ee7c1c2ad6b6628dc8520 2500w" />
  </Step>

  <Step title="Note Your AuthKit Domain">
    Find your **AuthKit Domain** on the configuration page. It will look like `https://your-project-12345.authkit.app`. You'll need this for your FastMCP server configuration.
  </Step>
</Steps>

### Step 2: FastMCP Configuration

Create your FastMCP server file and use the `AuthKitProvider` to handle all the OAuth integration automatically:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.workos import AuthKitProvider

# The AuthKitProvider automatically discovers WorkOS endpoints
# and configures JWT token validation
auth_provider = AuthKitProvider(
    authkit_domain="https://your-project-12345.authkit.app",
    base_url="http://localhost:8000"  # Use your actual server URL
)

mcp = FastMCP(name="AuthKit Secured App", auth=auth_provider)
```

## Testing

To test your server, you can use the `fastmcp` CLI to run it locally. Assuming you've saved the above code to `server.py` (after replacing the `authkit_domain` and `base_url` with your actual values!), you can run the following command:

```bash  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --transport http --port 8000
```

AuthKit defaults DCR clients to `client_secret_basic` for token exchange, which conflicts with how some MCP clients send credentials. To avoid token exchange errors, register as a public client by setting `token_endpoint_auth_method` to `"none"`:

```python client.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth
import asyncio

auth = OAuth(additional_client_metadata={"token_endpoint_auth_method": "none"})

async def main():
    async with Client("http://localhost:8000/mcp", auth=auth) as client:
        assert await client.ping()

if __name__ == "__main__":
    asyncio.run(main())
```

## Production Configuration

For production deployments, load sensitive configuration from environment variables:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp import FastMCP
from fastmcp.server.auth.providers.workos import AuthKitProvider

# Load configuration from environment variables
auth = AuthKitProvider(
    authkit_domain=os.environ.get("AUTHKIT_DOMAIN"),
    base_url=os.environ.get("BASE_URL", "https://your-server.com")
)

mcp = FastMCP(name="AuthKit Secured App", auth=auth)
```
