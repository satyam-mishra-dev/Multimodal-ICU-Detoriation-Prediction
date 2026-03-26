#!/bin/bash
# Start the MDS-ICU FastAPI backend
cd "$(dirname "$0")"
pip3 install -r api/requirements.txt -q --break-system-packages
PYTHONPATH="" uvicorn api.main:app --reload --reload-dir api --host 0.0.0.0 --port 8000
