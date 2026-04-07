# Vectors Documentation Makefile

.PHONY: help build build-fast serve clean stop

help:
	@echo "Vectors Documentation Commands"
	@echo "=============================="
	@echo ""
	@echo "Development Commands:"
	@echo "  make build      - Build complete documentation with Javadoc"
	@echo "  make build-fast - Build documentation without Javadoc (faster)"
	@echo "  make serve      - Build and serve documentation locally at http://localhost:8000"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean      - Clean all build artifacts"
	@echo "  make stop       - Stop Docker container"
	@echo ""

build:
	@echo "Building complete documentation with Javadoc integration..."
	cd .. && ./gradlew :docs:build
	@echo "Build complete! Documentation is in build/site/"

build-fast:
	@echo "Building documentation without Javadoc (faster)..."
	npx antora antora-playbook.yml --to-dir=build/site
	@echo "Fast build complete! Documentation is in build/site/"

serve: build
	@echo "Starting local documentation server..."
	@echo "Documentation will be available at http://localhost:8000"
	docker-compose up

clean:
	@echo "Cleaning build artifacts..."
	cd .. && ./gradlew clean
	rm -rf build/site
	rm -rf content/modules/ROOT/attachments/javadoc
	rm -rf node_modules/.cache
	@echo "Clean complete!"

stop:
	@echo "Stopping documentation server..."
	docker-compose down
	@echo "Server stopped!"
