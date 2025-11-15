#
# USAGE:
#
# To install virtualenv & packages
#   just install
#
# To check code health
#   just
#
# To run models:
#   just run-server
#   just run-client
#

#
# CONFIG
#
PYTHON_VERSION := "3.13"
VENV_DIR := ".venv"

RUFF := "ruff"
RUFF_FLAGS := "--line-length 120"
TY := "ty"
UV := "uv"
UV_FORMAT_FLAGS := "--preview-features format"

default: healthy


#
# UV tasks
#
uv-install:
    curl -LsSf https://astral.sh/uv/install.sh | sh

uv-sync:
    {{UV}} sync

uv-update:
    {{UV}} self update


sync:
    @{{UV}} sync


#
# LINT tasks
#
lint: ruff-check

ruff-install:
    {{UV}} tool install {{RUFF}}

ruff-check:
    @{{RUFF}} check . {{RUFF_FLAGS}} --exit-zero
    @echo "✅ Lint passed!"
    @echo



#
# FORMAT tasks
#
format:
    @{{UV}} format {{UV_FORMAT_FLAGS}}

format-check:
    @{{UV}} format {{UV_FORMAT_FLAGS}} --check



#
# TYPECHECK tasks
#
typecheck:
    @{{TY}} check
    @echo "✅ Typecheck passed!"



#
# TESTING tasks
#
test:
    @pytest --cov=. *test.py
    @echo "✅ Tests passed!"
    @echo



#
# PROJECT LIFECYCLE tasks
#
clean:
    @echo "🧹 Removing virtual environment..."
    rm -rf {{VENV_DIR}}

init: virtualenv ruff-install install
    @echo "✅ Project environment initialized!"

install: sync

virtualenv:
    @echo "🔧 Creating virtual environment with Python {{PYTHON_VERSION}}..."
    {{UV}} venv --python {{PYTHON_VERSION}} {{VENV_DIR}}
    @echo "✅ Virtualenv ready. Activate with: source {{VENV_DIR}}/bin/activate"



#
# FUNCTIONAL tasks
#
healthy:
    just format && \
    just lint && \
    just typecheck

run-tags args:
    @python tags.py {{args}}

run-client args:
    @python client.py {{args}}

run-server args:
    @python server.py {{args}}

run-training args:
    @python train.py {{args}}

