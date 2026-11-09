#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.1.0}"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${WORKSPACE_ROOT}/dist"
PKG_NAME="kriya-rt-v${VERSION}-aarch64-apple-darwin"
TARGET_DIR="${DIST_DIR}/${PKG_NAME}"

echo "=========================================================================="
echo " Building KriyaRT v${VERSION} Release Package for aarch64-apple-darwin"
echo "=========================================================================="

cd "${WORKSPACE_ROOT}"

# Build release binaries
echo "==> Compiling release binaries..."
cargo build --release --target aarch64-apple-darwin -p kriya-server
cargo build --release --target aarch64-apple-darwin --bin kriya_convert -p kriya-server

# Prepare packaging staging directory
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/bin"

cp "${WORKSPACE_ROOT}/target/aarch64-apple-darwin/release/kriya-server" "${TARGET_DIR}/bin/"
cp "${WORKSPACE_ROOT}/target/aarch64-apple-darwin/release/kriya_convert" "${TARGET_DIR}/bin/"
cp "${WORKSPACE_ROOT}/README.md" "${TARGET_DIR}/"

if [ -f "${WORKSPACE_ROOT}/LICENSE" ]; then
    cp "${WORKSPACE_ROOT}/LICENSE" "${TARGET_DIR}/"
fi

cd "${DIST_DIR}"
echo "==> Creating tarball..."
tar -czvf "${PKG_NAME}.tar.gz" "${PKG_NAME}"
shasum -a 256 "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256"

echo "=========================================================================="
echo " Package successfully built at:"
echo "   ${DIST_DIR}/${PKG_NAME}.tar.gz"
echo " SHA256 Checksum:"
cat "${DIST_DIR}/${PKG_NAME}.tar.gz.sha256"
echo "=========================================================================="
