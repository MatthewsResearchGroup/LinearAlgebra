# LTLT

Algorithms for the `LTL'` factorization of a skew-symmetric matrix: given
skew-symmetric `X`, compute unit lower triangular `L` and skew-symmetric
tridiagonal `T` with `X = L * T * L'`.

`matlab/` holds reference implementations in FLAME partitioning notation;
`src/` holds the C++ that mirrors them line for line, on top of BLIS and
MArray. New variants are derived and tested in MATLAB first, then ported.

Copyright 2024 Southern Methodist University and the University of Texas at
Austin. See [LICENSE](LICENSE).

## Building

    cmake -B build
    cmake --build build -j

Needs CMake 3.23+, Python 3 (for BLIS's build), and a C/C++20 toolchain taking
`-fopenmp` — macOS's `/usr/bin/cc` does not, so set `CMAKE_C_COMPILER` and
`CMAKE_CXX_COMPILER` to one that does. The first build compiles BLIS and
dominates; everything lands under the build directory, so `rm -rf build` is a
full reset.

`NDEBUG` is left undefined on purpose, keeping MArray's bounds checking live at
`-O3`. Add `-DCMAKE_CXX_FLAGS=-DNDEBUG` for timing runs needing the last few
percent.

### Presets

`CMakePresets.json` defines `debug`, `release`, and `bench` (`release` plus
`-DNDEBUG`) with portable settings only. Pick a compiler and generator in
`CMakeUserPresets.json`, which CMake reads alongside it and git ignores:

    cp CMakeUserPresets.json.example CMakeUserPresets.json   # then edit paths
    cmake --preset local-release
    cmake --build --preset local-release

`local-release` inherits its build type from `release` and its compiler from
your `local`. Binaries go to `build/<preset>/build/bin/`; all presets share one
BLIS build under `build/blis/`.

## Running

`ltlt_debug` sweeps every variant and reports residuals, which should be at
machine precision.

`ltlt_perf --help` documents the timing harness. It **appends** to `time.csv` /
`time_<bs>.csv` in the working directory, so delete those between experiment
series or results will be mixed together.

## Optimization steps

Each algorithm is a `template <int Options>` instantiated for the six
cumulative steps in [`src/ltlt.hpp`](src/ltlt.hpp). A new optimization is added
as a new flag bit and a new `STEP_N`, never as an edit that changes the
behaviour of an earlier step.

## MATLAB

Run `Test_LTLt_all` from within `matlab/` to check every variant.
