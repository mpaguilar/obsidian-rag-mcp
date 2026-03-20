#!/usr/bin/env python3
"""Health check script for Docker container."""

import json
import sys
import urllib.request


def main() -> int:
    """Check MCP server health endpoint.

    Returns:
        0 if healthy, 1 otherwise.

    """
    try:
        response = urllib.request.urlopen(
            "http://localhost:8000/health",
            timeout=5,
        )
        data = json.loads(response.read())
        if data.get("status") == "healthy":
            return 0
        else:
            print(f"Health check failed: {data}")  # noqa: T201
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"Health check error: {e}")  # noqa: T201
        return 1


if __name__ == "__main__":
    sys.exit(main())
