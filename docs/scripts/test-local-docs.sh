#!/bin/bash
set -e

echo "Testing Vectors Documentation with Javadoc Integration"
echo "======================================================="

cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

echo ""
echo "Cleaning previous builds..."
./gradlew clean
rm -rf docs/build/site
rm -rf docs/content/modules/ROOT/attachments/javadoc

echo ""
echo "Building documentation with Javadoc integration..."
./gradlew :docs:build

echo ""
echo "Validating build output..."

if [ ! -d "docs/build/site" ]; then
    echo "Build failed: docs/build/site directory not found"
    exit 1
fi

if [ ! -f "docs/build/site/index.html" ]; then
    echo "Build failed: No index.html found in build output"
    exit 1
fi

echo "Documentation site built successfully"
echo ""
echo "To serve locally:"
echo "  cd docs"
echo "  docker-compose up -d"
echo "  open http://localhost:8000"
