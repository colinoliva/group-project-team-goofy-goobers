#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <version> [output-directory]" >&2
  exit 2
fi

VERSION="$1"
OUTPUT_DIR="${2:-dist}"
if [[ ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Version may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_NAME="ti-mmwave-radar-${VERSION}"
PACKAGE_DIR="${OUTPUT_DIR}/${PACKAGE_NAME}"

mkdir -p "${PACKAGE_DIR}/src/prototype"
cp "${ROOT_DIR}/README.md" "${ROOT_DIR}/LICENSE" "${ROOT_DIR}/requirements.txt" "${PACKAGE_DIR}/"
cp -R "${ROOT_DIR}/src/implementation" "${PACKAGE_DIR}/src/"
cp "${ROOT_DIR}/sensors.example.json" "${PACKAGE_DIR}/src/implementation/sensors.json"
cp -R "${ROOT_DIR}/src/prototype/software" "${PACKAGE_DIR}/src/prototype/"

rm -f "${OUTPUT_DIR}/${PACKAGE_NAME}.zip"
(
  cd "${OUTPUT_DIR}"
  zip -qr "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
)
printf '%s\n' "${OUTPUT_DIR}/${PACKAGE_NAME}.zip"
