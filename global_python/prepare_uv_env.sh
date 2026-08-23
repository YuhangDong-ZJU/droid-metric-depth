#!/usr/bin/env bash

# Create an isolated Python environment with a pinned, runtime-local uv.
# This file is sourced by the no-Conda entry points.

UV_VERSION="0.12.5"

prepare_uv_env() {
  local base_python="$1"
  local env_dir="$2"
  local env_parent partial_dir stale_dir
  local uv_bin uv_install_dir uv_partial_dir uv_stale_dir uv_version
  local installer_url="https://astral.sh/uv/${UV_VERSION}/install.sh"

  if [[ "${DROID_DEPTH_UV_UNTESTED_WARNING_SHOWN:-0}" != "1" ]]; then
    echo "WARNING: the global_python uv environment path is currently marked untested."
    DROID_DEPTH_UV_UNTESTED_WARNING_SHOWN=1
    export DROID_DEPTH_UV_UNTESTED_WARNING_SHOWN
  fi

  if ! command -v "$base_python" >/dev/null 2>&1; then
    echo "ERROR: Python is not available: $base_python" >&2
    return 1
  fi
  base_python="$(command -v "$base_python")"
  env_parent="$(dirname -- "$env_dir")"

  if [[ -n "${DROID_DEPTH_UV_BIN:-}" ]]; then
    uv_bin="$DROID_DEPTH_UV_BIN"
  elif command -v uv >/dev/null 2>&1 \
      && [[ "$(uv --version 2>/dev/null || true)" == "uv $UV_VERSION" ]]; then
    uv_bin="$(command -v uv)"
  else
    uv_install_dir="${DROID_DEPTH_UV_INSTALL_DIR:-$env_parent/uv-$UV_VERSION}"
    uv_bin="$uv_install_dir/uv"
    if [[ ! -x "$uv_bin" ]] \
        || [[ "$($uv_bin --version 2>/dev/null || true)" != "uv $UV_VERSION" ]]; then
      mkdir -p "$env_parent"
      uv_partial_dir="${uv_install_dir}.partial.$$"
      rm -rf -- "$uv_partial_dir"
      echo "Installing uv $UV_VERSION into the runtime directory: $uv_install_dir"
      if command -v wget >/dev/null 2>&1; then
        wget -qO- "$installer_url" \
          | env UV_UNMANAGED_INSTALL="$uv_partial_dir" UV_NO_MODIFY_PATH=1 sh
      elif command -v curl >/dev/null 2>&1; then
        curl -LsSf "$installer_url" \
          | env UV_UNMANAGED_INSTALL="$uv_partial_dir" UV_NO_MODIFY_PATH=1 sh
      else
        echo "ERROR: wget or curl is required to install uv." >&2
        return 1
      fi
      if [[ ! -x "$uv_partial_dir/uv" ]] \
          || [[ "$($uv_partial_dir/uv --version 2>/dev/null || true)" != "uv $UV_VERSION" ]]; then
        echo "ERROR: failed to install uv $UV_VERSION into $uv_partial_dir" >&2
        return 1
      fi
      if [[ -e "$uv_install_dir" ]]; then
        uv_stale_dir="${uv_install_dir}.incomplete.$(date +%Y%m%d%H%M%S)"
        mv "$uv_install_dir" "$uv_stale_dir"
        echo "Moved the previous incomplete uv installation to: $uv_stale_dir"
      fi
      mv "$uv_partial_dir" "$uv_install_dir"
    fi
  fi

  if [[ ! -x "$uv_bin" ]]; then
    echo "ERROR: uv is not executable: $uv_bin" >&2
    return 1
  fi
  uv_version="$($uv_bin --version)"
  if [[ "$uv_version" != "uv $UV_VERSION" ]]; then
    echo "ERROR: expected uv $UV_VERSION, got: $uv_version" >&2
    return 1
  fi

  DROID_DEPTH_UV="$uv_bin"
  UV_CACHE_DIR="${DROID_DEPTH_UV_CACHE_DIR:-$env_parent/uv-cache}"
  export DROID_DEPTH_UV UV_CACHE_DIR
  mkdir -p "$UV_CACHE_DIR"

  if [[ -x "$env_dir/bin/python" ]]; then
    DROID_DEPTH_PYTHON="$env_dir/bin/python"
    export DROID_DEPTH_PYTHON
    echo "uv environment ready: $env_dir"
    echo "uv: $uv_bin ($UV_VERSION)"
    return 0
  fi

  mkdir -p "$env_parent"
  partial_dir="${env_dir}.partial.$$"
  rm -rf -- "$partial_dir"

  echo "Creating isolated uv environment: $env_dir"
  "$uv_bin" venv --python "$base_python" "$partial_dir"
  if [[ ! -x "$partial_dir/bin/python" ]]; then
    echo "ERROR: failed to create uv environment: $partial_dir" >&2
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
  echo "Isolated uv environment created: $env_dir"
  echo "uv: $uv_bin ($UV_VERSION)"
}
