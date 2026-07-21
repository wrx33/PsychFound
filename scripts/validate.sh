#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
pytest -q
psychfound pipeline --dry-run

