> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# dependencies

# `fastmcp.dependencies`

Dependency injection exports for FastMCP.

This module re-exports dependency injection symbols to provide a clean,
centralized import location for all dependency-related functionality.

DI features (Depends, CurrentContext, CurrentFastMCP) work without pydocket
using the uncalled-for DI engine. Only task-related dependencies (CurrentDocket,
CurrentWorker) and background task execution require fastmcp\[tasks].
