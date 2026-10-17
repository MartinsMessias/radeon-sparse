#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "   Building RadeonSparse Engine + Official Benchmark Suite"
echo "======================================================================"

mkdir -p bin

echo "[1/3] Compiling Shared Library: bin/libradeon_sparse.so ..."
gcc -O3 -fPIC -shared -fvisibility=hidden -Wall -Wextra -Iinclude \
    src/radeon_sparse.c -o bin/libradeon_sparse.so -ldl -lm

echo "[2/3] Compiling Official Benchmark CLI: bin/radeon-sparse-bench ..."
gcc -O3 -Wall -Wextra -Iinclude benchmarks/radeon_sparse_bench.c \
    -Lbin -lradeon_sparse -Wl,-rpath,'$ORIGIN' -lm -o bin/radeon-sparse-bench

echo "[3/3] Compiling C Quickstart Example: bin/use_in_c ..."
gcc -O3 -Wall -Wextra -Iinclude examples/use_in_c.c \
    -Lbin -lradeon_sparse -Wl,-rpath,'$ORIGIN' -lm -o bin/use_in_c

echo "[SUCCESS] Build complete:"
echo "  - bin/libradeon_sparse.so"
echo "  - bin/radeon-sparse-bench"
echo "  - bin/use_in_c"
echo "======================================================================"
