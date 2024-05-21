# Global variables
PYTHON_INTERPRETER = python3

export PATH := $(shell pwd)/:$(PATH)

.PHONY: clean-build
clean-build: ## Remove build artifacts
	@echo "+ $@"
	@rm -fr build/
	@rm -fr dist/
	@rm -fr *.egg-info

.PHONY: clean-pyc
clean-pyc: ## Remove Python file artifacts
	@echo "+ $@"
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +

test_config:			## create a ~/.metaflow/config_test.json for using METAFLOW_PROFILE=test
	mkdir -p ~/.metaflowconfig
	echo '{"METAFLOW_DEFAULT_METADATA": "local", "METAFLOW_DATASTORE_SYSROOT_LOCAL": "/tmp/metaflow", "METAFLOW_DEFAULT_DATASTORE": "local"}' > ~/.metaflowconfig/config_test.json

.PHONY: help
help: ## Display make help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

unittest:  ## run unit tests
	@echo "+ $@"
	@$(PYTHON_INTERPRETER) -m pytest --numprocesses=auto --failed-first --dist=loadfile -vv --durations=10

pypi:  ## clean the build and upload to pypi
	@echo "+ $@"
	@make clean-build
	@$(PYTHON_INTERPRETER) -m build
	@twine upload dist/* --verbose
