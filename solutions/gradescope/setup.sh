#!/usr/bin/env bash
# Gradescope runs this once when the image is built.
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends python3-pip
pip3 install --no-cache-dir -r /autograder/source/requirements.txt
