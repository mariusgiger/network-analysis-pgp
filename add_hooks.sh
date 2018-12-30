#!/usr/bin/env bash
if [ ! -x "$(command -v pre-commit)" ]; then
    echo "ERROR: Please install pre-commit, see https://pre-commit.com/#install" >&2
    exit 1
fi

CONFIG=""
if [ ! -f .pre-commit-config.yaml ]; then
    CONFIG="--config repo/.pre-commit-config.yaml"
fi

pre-commit autoupdate $CONFIG
pre-commit install $CONFIG
