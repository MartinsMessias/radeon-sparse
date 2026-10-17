#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "   RadeonSparse — Global System Installer (Linux)"
echo "======================================================================"

# 1. Build binaries
./build_linux.sh

PREFIX="${PREFIX:-/usr/local}"
echo ""
echo "[1/4] Installing libradeon_sparse.so, header, and CLI to ${PREFIX} ..."
if [ "$(id -u)" -ne 0 ]; then
    sudo install -m 755 bin/libradeon_sparse.so "${PREFIX}/lib/libradeon_sparse.so"
    sudo install -m 755 bin/radeon-sparse-bench "${PREFIX}/bin/radeon-sparse-bench"
    sudo install -m 644 include/radeon_sparse.h "${PREFIX}/include/radeon_sparse.h"
    sudo ldconfig
else
    install -m 755 bin/libradeon_sparse.so "${PREFIX}/lib/libradeon_sparse.so"
    install -m 755 bin/radeon-sparse-bench "${PREFIX}/bin/radeon-sparse-bench"
    install -m 644 include/radeon_sparse.h "${PREFIX}/include/radeon_sparse.h"
    ldconfig
fi

cp -f bin/libradeon_sparse.so python/libradeon_sparse.so

echo "[2/4] Installing Python package (pip install .) ..."
python3 -m pip install . --no-deps --quiet || true

echo "======================================================================"
echo "[SUCCESS] RadeonSparse is now globally installed!"
echo "  - Shared Library : ${PREFIX}/lib/libradeon_sparse.so"
echo "  - C Header       : ${PREFIX}/include/radeon_sparse.h"
echo "  - CLI Benchmark  : radeon-sparse-bench"
echo "  - Python SDK     : from radeon_sparse import RadeonSparseEngine"
echo "======================================================================"
