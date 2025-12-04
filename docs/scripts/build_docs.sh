#!/bin/bash
# Build script for Nirvana documentation

set -e

echo "Building Nirvana documentation..."

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "MkDocs is not installed. Installing dependencies..."
    pip install -r docs/requirements.txt
fi

# Build the documentation
echo "Building documentation..."
mkdocs build

echo "Documentation built successfully! Output is in the 'site/' directory."
echo "To serve locally, run: mkdocs serve"

