#!/bin/bash
gunicorn --bind=0.0.0.0 --timeout 600 -k uvicorn.workers.UvicornWorker app.main:app
