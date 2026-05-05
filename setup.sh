#!/usr/bin/env bash
set -euo pipefail

python --version
poetry --version
poetry install --with dev
