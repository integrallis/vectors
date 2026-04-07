#!/bin/bash
set -e

echo "Rebuilding Vectors Documentation with Javadoc Integration"
echo "=========================================================="

cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

echo ""
echo "Cleaning previous builds..."
./gradlew clean
rm -rf docs/build/site
rm -rf docs/content/modules/ROOT/attachments/javadoc
rm -rf docs/node_modules/.cache

echo ""
echo "Building documentation with Javadoc integration..."
./gradlew :docs:build

echo ""
echo "Build completed!"

if [ -d "docs/build/site/vectors/current/_attachments/javadoc" ]; then
    echo "Javadoc attachments found in built site"
else
    echo "Warning: Javadoc attachments not found in built site"
fi

echo ""
echo "To test locally:"
echo "  cd docs"
echo "  docker-compose up -d"
echo "  open http://localhost:8000"
