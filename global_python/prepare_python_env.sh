#!/usr/bin/env bash

# Create an isolated Python environment without requiring Conda or sudo.
# This file is sourced by the global-Python entry points.

prepare_python_env() {
  local base_python="$1"
  local env_dir="$2"
  local partial_dir bootstrap_dir stale_dir

  if ! command -v "$base_python" >/dev/null 2>&1; then
    echo "ERROR: Python is not available: $base_python" >&2
    return 1
  fi
  base_python="$(command -v "$base_python")"

  if [[ -x "$env_dir/bin/python" ]] \
      && "$env_dir/bin/python" -m pip --version >/dev/null 2>&1; then
    DROID_DEPTH_PYTHON="$env_dir/bin/python"
    export DROID_DEPTH_PYTHON
    echo "Isolated Python environment ready: $env_dir"
    return 0
  fi

  mkdir -p "$(dirname -- "$env_dir")"
  partial_dir="${env_dir}.partial.$$"
  rm -rf -- "$partial_dir"

  echo "Creating isolated Python environment: $env_dir"
  if ! "$base_python" -m venv "$partial_dir" >/dev/null 2>&1; then
    echo "python -m venv is unavailable; bootstrapping virtualenv locally."
    if ! "$base_python" -m pip --version >/dev/null 2>&1; then
      echo "ERROR: neither venv nor pip is available for $base_python" >&2
      return 1
    fi
    bootstrap_dir="${env_dir}.virtualenv-bootstrap"
    mkdir -p "$bootstrap_dir"
    PIP_ROOT_USER_ACTION=ignore "$base_python" -m pip install \
      --disable-pip-version-check --upgrade --target "$bootstrap_dir" \
      virtualenv==20.35.4
    PYTHONPATH="$bootstrap_dir" "$base_python" -m virtualenv \
      --no-download "$partial_dir"
  fi

  if [[ ! -x "$partial_dir/bin/python" ]] \
      || ! "$partial_dir/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "ERROR: failed to create isolated Python environment: $partial_dir" >&2
    return 1
  fi

  if [[ -e "$env_dir" ]]; then
    stale_dir="${env_dir}.incomplete.$(date +%Y%m%d%H%M%S)"
    mv "$env_dir" "$stale_dir"
    echo "Moved the previous incomplete Python environment to: $stale_dir"
  fi
  mv "$partial_dir" "$env_dir"
  DROID_DEPTH_PYTHON="$env_dir/bin/python"
  export DROID_DEPTH_PYTHON
  echo "Isolated Python environment created: $env_dir"
}
