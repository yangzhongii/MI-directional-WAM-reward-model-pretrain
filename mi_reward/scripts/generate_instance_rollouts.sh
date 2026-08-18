#!/usr/bin/env bash
# Backward-compatible alias for the Stage 1 data-preparation entry point.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/prepare_instance_data.sh" "$@"
