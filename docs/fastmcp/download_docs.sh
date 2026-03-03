#!/bin/bash

set -euo pipefail

DOMAIN="gofastmcp.com"
LIST_URL="https://$DOMAIN/llms.txt"

echo "Fetching file list from $LIST_URL..."

# Download the index and extract markdown URLs ending in .md
curl -fsSL "$LIST_URL" 2>/dev/null | \
  grep -oE '\[.*?\]\(https?://[^)]+\.md\)' | \
  sed 's/^.*](//;s/)$//' | \
  while read -r url; do
    # Security check: only process URLs from the target domain
    if [[ "$url" != *"$DOMAIN"* ]]; then
      echo "Skipping external domain: $url"
      continue
    fi
    
    # Convert URL to local file path (remove domain)
    local_path="${url#https://$DOMAIN/}"
    local_path="${local_path#http://$DOMAIN/}"
    
    # Skip if path is empty or just a slash
    [[ -z "$local_path" || "$local_path" == "/" ]] && continue
    
    # Create the directory structure
    dir=$(dirname "$local_path")
    if [[ "$dir" != "." ]]; then
      mkdir -p "$dir"
    fi
    
    # Download the file if it doesn't exist (remove -f flag to always overwrite)
    if [[ ! -f "$local_path" ]]; then
      echo "Downloading: $url"
      if curl -fsSL "$url" -o "$local_path" 2>/dev/null; then
        echo "  ✓ Saved to $local_path"
      else
        echo "  ✗ Failed to download $url" >&2
      fi
    else
      echo "Skipping (exists): $local_path"
    fi
  done

echo "Done!"

