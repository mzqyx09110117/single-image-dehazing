#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python scripts/make_hazy.py
python scripts/evaluate.py
python scripts/run_demo.py
python scripts/make_figures.py
python -m unittest discover -s tests -v
