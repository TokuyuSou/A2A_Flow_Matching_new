#!/bin/bash

# Install additional dependencies
echo "Install additional dependencies..."
# Fix .zarr issue
uv pip install zarr==2.16.1
uv pip install numcodecs==0.11.0

# Fix hydra issue
uv pip install --upgrade hydra-core
