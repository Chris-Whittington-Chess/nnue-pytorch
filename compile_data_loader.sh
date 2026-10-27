#!/usr/bin/env bash
set -euo pipefail

# Builds the training_data_loader with profile-guided optimization (PGO).
#
# Workflow:
# 1. Configure and build with PGO instrumentation (PGO_Generate).
# 2. Run the bench target to collect profile data.
# 3. Reconfigure and build the shared library with the collected profile data (PGO_Use).
#
# Usage:
#   ./compile_data_loader.sh [path/to/pgo_input]
# If no argument is provided the default is `.pgo/small.binpack` in the repo root.

ROOT_DIR=$(pwd)
SRC_DIR=${SRC_DIR:-data_loader/cpp/}
BUILD_DIR=${BUILD_DIR:-build}
PGO_DIR=${PGO_DIR:-pgo_data}
PGO_INPUT=${1:-$ROOT_DIR/.pgo/small.binpack}

echo "ROOT_DIR: $ROOT_DIR"
echo "SRC_DIR: $SRC_DIR"
echo "BUILD_DIR: $BUILD_DIR"
echo "PGO_DIR: $PGO_DIR"
echo "PGO_INPUT: $PGO_INPUT"

echo "Cleaning previous build and profile data..."
# The previous DLL is about to be deleted along with $BUILD_DIR. On a box that is training,
# or that may need to RESUME an arm, that window matters: nothing can train without this
# library. Keep a copy outside the build dir, which is where a backup put *inside* it went
# on 2026-10-25.
if [ -f "$BUILD_DIR/training_data_loader.dll" ] || [ -f "$BUILD_DIR/training_data_loader.so" ]; then
  mkdir -p "$ROOT_DIR/.loader-backup"
  cp "$BUILD_DIR"/training_data_loader.* "$ROOT_DIR/.loader-backup/" 2>/dev/null || true
  echo "previous loader backed up to $ROOT_DIR/.loader-backup"
fi

rm -rf "$BUILD_DIR" "$PGO_DIR"

# ENCODER OPTIONS DECIDE THE STAMP EVERY EXPORTED NET CARRIES, and the CMakeLists default
# is NOT production's. SAFECHECK_FEATURES defaults to ON (EngineThreats 7 blocks,
# ThreatsV1nkET7fp); the engine expects OFF (5 blocks, ThreatsV1nkET5fp, fnv1a32 =
# 2520498150). A loader built with the default still produces CORRECT features -- verified
# bit-identical on a deterministic stream -- but stamps every exported net 1800157232, and
# the engine then refuses all of them: 'encoder stamp is not ThreatsV1nkET5fp'. Found on
# PC1 2026-10-25, before a 400-epoch arm was wasted on nets nothing could load.
# Verify after building, with tools/check_loader_stamp.py or by hand:
#     ctypes.CDLL('build/training_data_loader.dll').nnue_encoder_variant_id()
SAFECHECK=${SAFECHECK:--DSAFECHECK_FEATURES=OFF}

echo "Configuring PGO_Generate build..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=PGO_Generate \
  -DPGO_PROFILE_DATA_DIR="$ROOT_DIR/$PGO_DIR" \
  -DPGO_INPUT="$PGO_INPUT" \
  -DLIB_COPY_DIR="$ROOT_DIR" $SAFECHECK

# ---- STOP if the generator is MULTI-CONFIG -------------------------------------------
# Found 2026-10-25 on PC1. This script is written for a SINGLE-config generator (Ninja,
# Makefiles), where CMAKE_BUILD_TYPE selects the build and `cmake --build` needs no
# --config. Under the Visual Studio generator none of that holds:
#   * CMAKE_BUILD_TYPE is IGNORED;
#   * PGO_Generate / PGO_Use are not registered configurations at all
#     (CMAKE_CONFIGURATION_TYPES is Debug;Release;MinSizeRel;RelWithDebInfo);
#   * `cmake --build` with no --config therefore builds DEBUG.
# The script then printed "PGO build complete." and exited 0, having produced a 2.2 MB
# DEBUG dll in build/Debug/ -- not where the python loader looks, with no PGO, and
# catastrophically slower than the 0.5 MB optimised one. Silent, and it would have shown
# up only as unexplained dataloader throughput.
if grep -q '^CMAKE_CONFIGURATION_TYPES' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
  cfgs=$(grep '^CMAKE_CONFIGURATION_TYPES' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)
  echo "ERROR: multi-config generator detected (configurations: $cfgs)." >&2
  echo "       PGO_Generate / PGO_Use are not configurations there, so this script would" >&2
  echo "       silently build Debug. Refusing rather than producing a Debug loader." >&2
  echo "" >&2
  echo "       For PGO install a single-config generator and re-run, e.g.:" >&2
  echo "           cmake -S $SRC_DIR -B $BUILD_DIR -G Ninja $SAFECHECK ..." >&2
  echo "       For an optimised NON-PGO loader right now:" >&2
  echo "           cmake -S $SRC_DIR -B $BUILD_DIR -DLIB_COPY_DIR=\"$ROOT_DIR\" $SAFECHECK" >&2
  echo "           cmake --build $BUILD_DIR --config Release -j" >&2
  echo "           cp $BUILD_DIR/Release/training_data_loader.* $BUILD_DIR/" >&2
  echo "       (the python loader globs ./build/*training_data_loader.*, NOT build/Release)" >&2
  echo "       ^ $SAFECHECK is NOT optional: SAFECHECK_FEATURES defaults to ON in the" >&2
  echo "         CMakeLists, and an ON loader produces entirely correct features but stamps" >&2
  echo "         every net 1800157232 where the engine expects 2520498150 -- and the engine" >&2
  echo "         then refuses all of them, which you discover at the END of the arm." >&2
  exit 1
fi

echo "Building instrumented default targets..."
cmake --build "$BUILD_DIR" -j

echo "Running bench to generate profile data..."
cmake --build "$BUILD_DIR" --target pgo_run -j

# Clang generates .profraw files that must be merged into
# default.profdata before -fprofile-use can use it.
if ls "$PGO_DIR"/*.profraw 1>/dev/null 2>&1; then
  echo "Merging Clang raw profiles..."
  if command -v xcrun >/dev/null 2>&1; then
    xcrun llvm-profdata merge -output="$PGO_DIR/default.profdata" "$PGO_DIR"/*.profraw
  else
    echo "Error: llvm-profdata not found. Can't merge Clang profiles." >&2
    exit 1
  fi
fi

echo "Re-configuring for PGO_Use..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=PGO_Use \
  -DPGO_PROFILE_DATA_DIR="$ROOT_DIR/$PGO_DIR" \
  -DCMAKE_INSTALL_PREFIX="./" \
  -DLIB_COPY_DIR="$ROOT_DIR" $SAFECHECK

echo "Building default targets with profile data..."
cmake --build "$BUILD_DIR" -j

echo "PGO build complete."

rm -rf "$PGO_DIR"
