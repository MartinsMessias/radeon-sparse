#!/usr/bin/env bash
set -e

echo "========================================================"
echo " Building rdna3-ai-engine for Linux (GCC/Clang)"
echo "========================================================"

CC=${CC:-gcc}
CFLAGS="-O3 -fPIC -Wall -Wextra -I include"
LDFLAGS="-ldl -lm"

echo "[*] Compiling shared library (librdna3_ai_engine.so)..."
$CC -shared $CFLAGS src/rdna3_ai_engine.c -o librdna3_ai_engine.so $LDFLAGS

echo "[*] Compiling C example (use_in_c)..."
$CC $CFLAGS examples/use_in_c.c -L. -lrdna3_ai_engine -Wl,-rpath,. -o use_in_c $LDFLAGS

echo ""
echo "========================================================"
echo " BUILD SUCCESSFUL!"
echo " Generated: librdna3_ai_engine.so and use_in_c"
echo "========================================================"
