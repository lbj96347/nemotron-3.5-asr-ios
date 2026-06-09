#!/usr/bin/env bash
#
# Compile a .mlpackage into a .mlmodelc using Xcode's coremlcompiler.
#
# The published FluidInference repo already ships compiled .mlmodelc bundles, so
# this is usually a no-op. Keep it for the case where a future conversion step
# (or the ONNX fallback route) produces a .mlpackage that must be compiled before
# bundling into the app. Per the plan, do NOT compile on-device — compile here.
#
# Usage:
#   ./scripts/compile_mlpackage.sh path/to/model.mlpackage [output_dir]
#
set -euo pipefail

PKG="${1:?usage: compile_mlpackage.sh <model.mlpackage> [output_dir]}"
OUT_DIR="${2:-$(dirname "${PKG}")}"

if [[ ! -e "${PKG}" ]]; then
  echo "error: ${PKG} not found" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "Compiling ${PKG} → ${OUT_DIR}"
xcrun coremlcompiler compile "${PKG}" "${OUT_DIR}"
echo "Done: ${OUT_DIR}/$(basename "${PKG%.mlpackage}").mlmodelc"
