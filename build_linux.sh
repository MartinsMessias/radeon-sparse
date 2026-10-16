#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "   Building RDNA 3 / RDNA 4 AI Engine + Official Benchmark Suite"
echo "======================================================================"

mkdir -p bin

echo "[1/3] Compiling Shared Library: bin/librdna3_ai_engine.so ..."
gcc -O3 -fPIC -shared -fvisibility=hidden -Wall -Wextra -Iinclude \
    src/rdna3_ai_engine.c -o bin/librdna3_ai_engine.so -ldl -lm

echo "[2/3] Compiling Official Benchmark CLI: bin/rdna3-bench ..."
gcc -O3 -Wall -Wextra -Iinclude benchmarks/rdna3_bench.c \
    -Lbin -lrdna3_ai_engine -Wl,-rpath,'$ORIGIN' -lm -o bin/rdna3-bench

echo "[3/3] Compiling C Quickstart Example: bin/use_in_c ..."
gcc -O3 -Wall -Wextra -Iinclude examples/use_in_c.c \
    -Lbin -lrdna3_ai_engine -Wl,-rpath,'$ORIGIN' -lm -o bin/use_in_c

echo "[SUCCESS] Build complete:"
echo "  - bin/librdna3_ai_engine.so"
echo "  - bin/rdna3-bench"
echo "  - bin/use_in_c"
echo "======================================================================"
