#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/workspace"

echo "============================================================"
echo "CropGuard Knowledge Index Refresh"
echo "============================================================"
echo
echo "Project root : ${PROJECT_ROOT}"
echo "Started      : $(date)"
echo

python "${PROJECT_ROOT}/index/refresh_index.py" \
  --batch-size 16

echo
echo "============================================================"
echo "CropGuard Index Refresh Finished"
echo "Finished     : $(date)"
echo "============================================================"
