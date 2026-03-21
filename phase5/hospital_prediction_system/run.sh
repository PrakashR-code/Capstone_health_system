#!/bin/bash

cd ./app
echo "Starting Hospital AI API..."

uvicorn main:app --host 0.0.0.0 --port 8000