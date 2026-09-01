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

`ltlt_perf` times one algorithm, at one optimization step, over a range of
matrix sizes:

    ltlt_perf <majoralgo> <size_min> <size_max> <stepsize> <repetition> \
              [--minoralgo=<algo>] [--bs=<bs>] [--step=<step>]

`--help` lists the algorithm names. The two "step"s are unrelated: `<stepsize>`
strides the matrix sizes, `--step` picks the optimization step.

`--minoralgo` names the unblocked kernel a block algorithm uses on each diagonal
block, and passing it is also what switches on the blocked path; without it
`<majoralgo>` runs unblocked and `--bs` is ignored. `--bs` is the block size,
default 256. `--step` runs 0-5, default 5. All six instantiations are compiled
in, so the flag costs nothing but a switch at startup.

At each size the harness builds a random `A`, forms the skew-symmetric `B = A -
A'`, and factors a fresh copy `<repetition>` times. It keeps the fastest of
those runs. GFLOPS is the analytic `n^3/3` count over that time, not a measured
one.

Per-rep times and a per-size summary go to stdout, then the
`timer::print_timers()` breakdown of the profiled sections. Results are
**appended** to `time.csv` for unblocked runs and `time_<bs>.csv` for blocked
ones, with columns `NUM_THREADS, MatrixSize, MajorAlgo, MinorAlgo, BlockSize,
Time, GFLOPS`. The header only goes in when the file is empty, so delete those
between experiment series or results get mixed together.

## Optimization steps

Each algorithm is a `template <int Options>` instantiated for the six cumulative
steps in [`src/ltlt.hpp`](src/ltlt.hpp). Add a new optimization as a new flag
bit and a new `STEP_N`; never edit one so that it changes an earlier step's
behaviour.

[`docs/optimization-steps.md`](docs/optimization-steps.md) explains what each
flag switches, how the fused paths reach into BLIS, and what is still
unmeasured. [`docs/ltl-algorithms.md`](docs/ltl-algorithms.md) covers how the
variants themselves differ. [`docs/blas-blis-apis.md`](docs/blas-blis-apis.md)
compares what BLAS and BLIS each expose, and why the fused paths have to reach
as deep into BLIS as they do.

## MATLAB

Run `Test_LTLt_all` from within `matlab/` to check every variant.
