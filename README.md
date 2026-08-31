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

    cp CMakeUserPresets.json.example CMakeUserPresets.json   # then edit paths
    cmake --preset local-release
    cmake --build --preset local-release

Needs CMake 3.23+, Python 3 (for BLIS's build), and a C/C++20 toolchain that
takes `-fopenmp`. macOS's `/usr/bin/cc` does not, which is why
`CMakeUserPresets.json` names a compiler.

Three build types: `debug`, `release`, and `bench` (`release` plus `-DNDEBUG`).
Each has a `local-*` counterpart adding the compiler and generator from your
`CMakeUserPresets.json`.

Binaries go to `out/<preset>/build/bin/`. All presets share one BLIS build
under `out/blis/`, so only the first configure pays for it, and that first
configure dominates. `rm -rf out` is a full reset, BLIS included.

`NDEBUG` stays undefined under `release`, so MArray's bounds checking is still
live at `-O3`. Use `bench` for timing runs that need the last few percent.

## Running

`ltlt_debug` takes no arguments. It sweeps every variant, size, and step and
reports residuals, which should be at machine precision.

`ltlt_perf --help` documents the timing harness. It **appends** to `time.csv` /
`time_<bs>.csv` in the working directory, so delete those between experiment
series or results get mixed together.

## Optimization steps

Each algorithm is a `template <int Options>` instantiated for the six
cumulative steps in [`src/ltlt.hpp`](src/ltlt.hpp). Add a new optimization as a
new flag bit and a new `STEP_N`; never edit one so that it changes an earlier
step's behaviour.

[`docs/optimization-steps.md`](docs/optimization-steps.md) explains what each
flag switches, how the fused paths reach into BLIS, and what is still unmeasured.
[`docs/ltl-algorithms.md`](docs/ltl-algorithms.md) covers how the variants
themselves differ.

## MATLAB

Run `Test_LTLt_all` from within `matlab/` to check every variant.
