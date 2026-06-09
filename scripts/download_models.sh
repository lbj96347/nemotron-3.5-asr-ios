#!/usr/bin/env bash
#
# Download a Nemotron-3.5-ASR CoreML tier into Models/<variant>/<tier>ms/.
#
# The HF repo is public (not gated); every file — including the LFS weight.bin
# blobs — is reachable via the /resolve/main/ URL with a redirect-following
# curl, so this needs nothing beyond curl. (No huggingface_hub / git-lfs / hf.)
#
# Usage:
#   ./scripts/download_models.sh [variant] [tier_ms]
#   ./scripts/download_models.sh multilingual 2240   # default
#
set -euo pipefail

VARIANT="${1:-multilingual}"
TIER="${2:-2240}"
REPO="FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML"
BASE="https://huggingface.co/${REPO}/resolve/main"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${ROOT}/Models/${VARIANT}/${TIER}ms"

echo "Downloading ${VARIANT}/${TIER}ms → ${DEST}"
mkdir -p "${DEST}"

# decoder_joint is the fused default path; decoder + joint are the fallback.
MODULES=(preprocessor encoder decoder_joint decoder joint)
# Files inside each compiled .mlmodelc bundle.
MLMODELC_FILES=(analytics/coremldata.bin coremldata.bin model.mil weights/weight.bin)
# Per-tier loose files.
TIER_FILES=(metadata.json tokenizer.json)

fetch() {  # fetch <remote-rel-path> <local-abs-path>
  local rel="$1" out="$2"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" ]]; then
    echo "  • exists, skipping ${rel}"
    return
  fi
  echo "  ↓ ${rel}"
  curl -fL --retry 3 --progress-bar "${BASE}/${VARIANT}/${TIER}ms/${rel}" -o "${out}.part"
  mv "${out}.part" "${out}"
}

for module in "${MODULES[@]}"; do
  for f in "${MLMODELC_FILES[@]}"; do
    fetch "${module}.mlmodelc/${f}" "${DEST}/${module}.mlmodelc/${f}"
  done
done
for f in "${TIER_FILES[@]}"; do
  fetch "${f}" "${DEST}/${f}"
done

# Top-level manifest (handy reference; ignore failure if absent).
curl -fL --retry 3 -s "${BASE}/manifest.json" -o "${ROOT}/Models/manifest.json" || true

echo
echo "Done. Next:"
echo "  python3 scripts/inspect_model.py Models/${VARIANT}/${TIER}ms --out NemotronASRPoC/ASR/ModelSignatures.json"
echo "  xcodegen generate   # bundle the new Models/ folder"
