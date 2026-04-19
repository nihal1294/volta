set shell := ["bash", "-cu"]

default:
    @just --list

setup:
    ./scripts/bootstrap_api.sh
    ./scripts/bootstrap_web.sh

doctor:
    python3 ./scripts/doctor.py

api:
    ./scripts/dev_backend.sh

web:
    ./scripts/dev_web.sh

dev:
    ./scripts/quickstart.sh

quickstart:
    ./scripts/quickstart.sh

test:
    uv run --project apps/api --python 3.14 --extra dev pytest apps/api/tests -q
    npm --prefix apps/web run build

lint:
    uv run --project apps/api --python 3.14 --extra dev ruff check apps/api scripts

fmt:
    uv run --project apps/api --python 3.14 --extra dev ruff format apps/api scripts

check:
    just lint
    just fmt
    just test

clean:
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
    find . -name .pytest_cache -type d -prune -exec rm -rf {} +
    find . -name .ruff_cache -type d -prune -exec rm -rf {} +
    find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
    rm -rf .runtime .venv apps/web/dist
