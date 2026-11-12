#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.2.0}"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${WORKSPACE_ROOT}/dist"
PKG_NAME="kriya-rt-v${VERSION}-aarch64-apple-darwin"
TARGET_DIR="${DIST_DIR}/${PKG_NAME}"

echo "=========================================================================="
echo " Building KriyaRT v${VERSION} Release Package for aarch64-apple-darwin"
echo "=========================================================================="

cd "${WORKSPACE_ROOT}"

# Build release binaries with locked dependencies
echo "==> Compiling release binaries with --locked..."
cargo build --release --locked --target aarch64-apple-darwin -p kriya-server
cargo build --release --locked --target aarch64-apple-darwin --bin kriya_convert -p kriya-server

# Prepare packaging staging directory
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/bin"

cp "${WORKSPACE_ROOT}/target/aarch64-apple-darwin/release/kriya-server" "${TARGET_DIR}/bin/"
cp "${WORKSPACE_ROOT}/target/aarch64-apple-darwin/release/kriya_convert" "${TARGET_DIR}/bin/"
cp "${WORKSPACE_ROOT}/README.md" "${TARGET_DIR}/"

if [ -f "${WORKSPACE_ROOT}/LICENSE" ]; then
    cp "${WORKSPACE_ROOT}/LICENSE" "${TARGET_DIR}/"
fi

# Self-validation: inspect binary architecture & run smoke tests
echo "==> Validating executable architecture..."
file "${TARGET_DIR}/bin/kriya-server" | grep -E "arm64|aarch64" > /dev/null || {
    echo "ERROR: kriya-server binary is not arm64 architecture" >&2
    exit 1
}
file "${TARGET_DIR}/bin/kriya_convert" | grep -E "arm64|aarch64" > /dev/null || {
    echo "ERROR: kriya_convert binary is not arm64 architecture" >&2
    exit 1
}

echo "==> Running binary smoke tests..."
"${TARGET_DIR}/bin/kriya-server" --version > /dev/null
"${TARGET_DIR}/bin/kriya_convert" --version > /dev/null

cd "${DIST_DIR}"
echo "==> Creating tarball..."
tar -czvf "${PKG_NAME}.tar.gz" "${PKG_NAME}"
shasum -a 256 "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256"

echo "==> Verifying SHA256 integrity..."
shasum -a 256 -c "${PKG_NAME}.tar.gz.sha256"

echo "==> Validating tarball contents..."
tar -tzf "${PKG_NAME}.tar.gz"

# Clean up temporary staging directory
rm -rf "${TARGET_DIR}"

echo "=========================================================================="
echo " Package successfully built at:"
echo "   ${DIST_DIR}/${PKG_NAME}.tar.gz"
echo " SHA256 Checksum:"
cat "${DIST_DIR}/${PKG_NAME}.tar.gz.sha256"
echo "=========================================================================="
