# The optimization ladder

`src/ltlt.hpp:26` holds an unnamed enum: five bit flags and six `STEP_N`
constants built from them. Every algorithm in `src/` is a
`template <int Options>` instantiated at all six steps. `ltlt_debug` runs all
six in sequence; `ltlt_perf --step=N` picks one.

A flag does not name a technique. It names a *layer*, and selects one of two
implementations of the same operation at that layer. Both implementations
compute the same thing; what changes is which code path runs and where that
path lives. The steps are cumulative prefixes of a fixed order, so a sweep over
`--step` attributes a change in GFLOPS to exactly one substitution.

## The five bits

| flag | bit | first in | selects between | lives in |
|---|---|---|---|---|
| `FUSED_L2` | `0x2` | `STEP_1` | MArray's BLAS wrappers vs. hand-written level-2 kernels | `ltlt_blas.cpp` |
| `PARALLEL_L2` | `0x4` | `STEP_2` | serial vs. OpenMP inside those kernels | `ltlt_blas.cpp` |
| `SEPARATE_T` | `0x8` | `STEP_3` | `t` left in L's diagonal slot vs. a literal 1 written there | every `ltlt_*block*.cpp` |
| `FUSED_L3` | `0x1` | `STEP_4` | materialize `T*B` then `gemm` vs. apply `T` inside BLIS's packing | `ltlt_blas.cpp` |
| `BLOCK_RL_VAR1` | `0x10` | `STEP_5` | `ltlt_blockRL` vs. `ltlt_blockRL_var1` | `ltlt_blockRL.cpp`, `ltlt_pivot_blockRL.cpp` |

### FUSED_L2

Gates three functions in `ltlt_blas.cpp`. Only one of them fuses `T`.

`gemv_sktri` genuinely fuses. With the flag off it copies `x`, applies `T` to
the copy with `sktrmv`, and calls `blas::gemv`. With it on, the three-term
stencil `t[j-1]*x[j-1] - t[j]*x[j+1]` is formed a few entries at a time and fed
straight to BLIS's `axpyf` micro-kernel, so `T*x` never exists as an array.

`ger2` fuses two rank-1 updates into one pass over `E` instead of two
`blas::ger` calls, each of which reads and writes all of `E`.

`skr2` replaces `bli_skr2` with a version that walks two columns at a time and
can be threaded. No `T` is involved at all.

So the flag means "our own level-2 kernels instead of the library's".

### PARALLEL_L2

An `if` clause on the nine `#pragma omp parallel` regions in `ltlt_blas.cpp`.
All nine sit inside code that runs only when `FUSED_L2` is set, since each
kernel early-returns to the BLAS path before reaching them. `PARALLEL_L2` on
its own would therefore do nothing. That is why the ladder puts `FUSED_L2`
first, and `BLOCK_RL_VAR1` below has the same dependency on the two bits under
it: a later bit is often meaningless without an earlier one.

The flag controls only the level-2 kernels. Level-3 work is threaded by BLIS
itself at every step, `STEP_0` included. BLIS is configured with `-t openmp`
(`external/CMakeLists.txt`), so both use the same OpenMP runtime and
`OMP_NUM_THREADS` moves both at once.

### SEPARATE_T

Start with the storage. `L` lives inside `X`, shifted one column over
(`X.rebased(1, 1)`, or `X.shifted(1, -1)` when the caller owns the first
column), so column `j` of `X` holds column `j+1` of `L`. Column 0 of `L` is
`e_1` and is never stored. `make_L` and `make_T` in `ltlt.hpp` are the
decoders; read them when the index arithmetic stops making sense, since they
are the only place the layout is written out in full.

The slot where L's unit diagonal would sit is X's subdiagonal, which is exactly
`t`. With the flag off, the algorithms copy that value into the `t` array and
leave the original in place. With it on, they also overwrite the slot with 1.

The point is operand shape. With a real 1 there, the boundary column is a
valid part of a unit-triangular operand and one call covers the whole update.
Without it, that column has to be peeled off.
`src/ltlt_blockLL.cpp:23-48` is the clearest side-by-side: one `gemm_sktri`
over `[R0|r1]`, against a hand-written rank-1 correction plus a `gemv_sktri`
over `R0` plus a `gemm_sktri` over a narrower panel. Same arithmetic, three
calls instead of one, and the level-3 call is smaller.

The residual check is blind to the difference, because `make_L` writes 1 on the
diagonal regardless of what is stored there.

### FUSED_L3

The largest change of the five. Off, `gemm_sktri` copies `b`, applies `T` to
the copy with `sktrmm`, and calls `blas::gemm`: a temporary the size of the
panel, plus a full extra pass over it.

On, it skips the wrapper and drives BLIS's control tree directly. It wraps the
existing buffers as `obj_t` without copying, calls `bli_gemm_cntl_init`, then
replaces the pack-B micro-kernel with the project's `packing()`
(`src/ltlt_packing.cpp`) and attaches a `skparams` carrying `t`'s pointer,
stride, and length. BLIS packs `b` into its level-3 buffer either way;
`packing` applies the tridiagonal stencil during that copy. The multiplication
by `T` becomes free and the temporary disappears.

`gemmt_sktri` does the same for the triangular-output case, marking `c` as
`BLIS_TRIANGULAR` with the requested uplo.

### BLOCK_RL_VAR1

This one changes the schedule; every kernel stays the same. `ltlt_blockRL`
forwards to `ltlt_blockRL_var1`, which peels the first column outside the loop
and then covers each trailing update with a single `gemmt_sktri`.

One asymmetry is worth knowing. `ltlt_blockRL_var1` and
`ltlt_pivot_blockRL_var1` are not templates. They hardcode
`gemmt_sktri<STEP_5>` and write the unit diagonal unconditionally, so they
assume `FUSED_L3` and `SEPARATE_T`. That holds today because
`STEP_5 = STEP_4 | BLOCK_RL_VAR1` and nothing else sets the bit, but the
assumption is invisible to the type system.

## Bit order is not step order

`FUSED_L3` is bit 0 and arrives last. Bits were assigned as they were written;
steps order them by when they were measured. A new optimization takes the next
free bit and a new `STEP_N`, and existing steps never change meaning. That is
what keeps an old `time.csv` comparable to a new one.

## Where this meets BLIS

### The contact surface is small

`blis.h` exposes several hundred symbols. `src/` and `apps/` call about
twenty-five, and they group into five roles.

| role | symbols |
|---|---|
| wrap MArray views as BLIS objects, no copy | `bli_obj_create_with_attached_buffer`, `bli_obj_create_1x1_with_attached_buffer`, `bli_obj_row_stride`, `bli_obj_col_stride`, `bli_obj_dt`, `bli_obj_is_complex`, `bli_obj_set_uplo`, `bli_obj_set_struc` |
| ask the machine description for things | `bli_init`, `bli_gks_query_cntx`, `bli_cntx_get_ukr_dt`, `bli_cntx_get_blksz_def_dt` |
| build and modify a level-3 control tree | `gemm_cntl_t`, `bli_gemm_cntl_init`, `bli_gemm_cntl_set_pack{a,b}_ukr_simple`, `bli_gemm_cntl_set_pack{a,b}_params`, `bli_func_set_dt` |
| run it | `bli_l3_thread_decorator`, `bli_ind_oper_find_avail`, `bli_negsc`, `bli_clock` |
| enums and scalar typedefs | `BLIS_DOUBLE`, `BLIS_LOWER/UPPER/DENSE`, `BLIS_TRIANGULAR`, `BLIS_SKEW_SYMMETRIC`, `BLIS_NAT`, `BLIS_GEMM/GEMMT`, `BLIS_AXPYF_KER`, `BLIS_AXPY2V_KER`, `BLIS_AF`, `BLIS_NO_CONJUGATE`, `dim_t`, `inc_t`, `num_t`, `ind_t` |

Everything else reaches BLIS through MArray's `blas::` wrappers in
`external/marray/marray/blas.h`, which is also where `blas::skr2`,
`blas::gemmt`, and `blas::skr2k` come from: MArray already knows about
skew-symmetric structure, which is why the unfused paths have something to call.

### cntx_t

A `cntx_t` is BLIS's description of the machine. Per datatype it holds a table
of blocksizes, micro-kernel function pointers, and kernel preferences, filled
in at init from whichever sub-configuration `configure` selected.
`bli_gks_query_cntx()` returns the read-only one for the active configuration.

This code reads it and never builds or modifies one. Two uses:

1. `gemv_sktri` fetches a level-1f micro-kernel (`BLIS_AXPYF_KER`) and its
   fusing factor `BLIS_AF`, which is how many columns the kernel consumes per
   call. This is the only place a blocking factor comes from the context:
   `skr2`'s `BS` is a `constexpr 2` and `ger2`'s is a `constexpr 5`. `skr2`
   also fetches `BLIS_AXPY2V_KER` into `kfp_2v` and then never calls it; its
   fused path is hand-written throughout, so `gemv_sktri` is the only kernel
   that actually dispatches into BLIS at level 1.
2. `gemm_sktri` and `gemmt_sktri` pass it to `bli_gemm_cntl_init` and
   `bli_l3_thread_decorator`, so the level-3 loops are blocked for this
   machine's caches.

So every cache blocksize here is inherited. `external/CMakeLists.txt` picks
`firestorm` on Apple silicon and `haswell` on x86 macOS. For `double`,
firestorm gives MR=8, NR=6, MC=256, KC=3072, NC=8184, and inherits AF=8 from
the reference defaults. Changing any of them
means picking a different `BLIS_CONFIG`.

### The packing hook

`packing()` in `src/ltlt_packing.cpp` has exactly the signature of BLIS's
`packm_struc_cxk` reference kernel
(`external/blis/frame/1m/packm/bli_packm_struc_cxk.h`), so BLIS accepts it in
place of its own. BLIS calls it once per panel. The parameters describe the
panel's shape (`panel_dim` by `panel_len`), its padded shape (the `_max` pair),
and its offset within the full operand (`panel_dim_off`, `panel_len_off`).
`skparams` supplies the one thing BLIS cannot know: `t`'s pointer, stride, and
length.

Both call sites pack the B side (`pack_side = 1`), so `panel_len_off` indexes
along the k dimension, which is where `T` lives. The four branches are the
boundary cases of the three-term stencil: a panel at the start has no left
neighbour, one at the end has no right neighbour, one that is both needs both,
and an interior panel needs neither. The closing `set0s_edge` zeroes the
padding out to `panel_dim_max` and `panel_len_max`, as any packing kernel must.

## A strategy for the rest of BLIS

`blis.h` will not be read front to back, and does not need to be. Four rules
keep the picture bounded.

**1. Go by what the code calls.** A BLIS symbol earns a row above when `src/`
or `apps/` calls it. Anything outside that set is out of scope here, however
interesting. To find candidates when the code changes:

    grep -rhoE '\bbli_[a-z0-9_]+|BLIS_[A-Z0-9_]+' src apps | sort -u

The grep matches text, so it over-counts. `skr2` names `BLIS_AXPY2V_KER` and
stores the result in `kfp_2v`, then never calls it, and the symbol shows up in
the output all the same. Before adding a row, check that the value is used and
not just fetched. The output is a flat list, so read it as a diff against the
five-role table above.

**2. Give every number an owner.** There are three:

- the BLIS sub-configuration owns MR, NR, MC, KC, NC, AF, and the
  micro-kernels; they change only by choosing a different `BLIS_CONFIG`;
- the enum owns which implementation of an operation runs;
- the run owns `block_size` (`--bs`), the thread count, the matrix size,
  `-march=native`, and whether `NDEBUG` is defined.

A number that traces to none of them is one this project does not control, and
reasoning about it is guesswork.

**3. One question per sitting, with its experiment attached.** Keep the list
below. An entry is an assumption plus the measurement that would confirm or
kill it, and it gets filled in when a run comes back.

**4. Anchor claims to the CSV.** `ltlt_perf` records thread count, matrix size,
the algorithm pair, block size, time, and GFLOPS, and `--step` selects the
rung. Any speedup asserted here should name the rows that show it.
`timer::print_timers()` breaks a run down per profiled section when a claim
needs localizing further. `PROFILE` is defined unconditionally by
`CMakeLists.txt`, so the timers are always live.

## What the fused paths assume

- `beta` is 1 at every call site. `gemm_sktri` and `gemmt_sktri` return early
  when the k dimension is at most 1, without scaling `c`, which is correct only
  because every caller passes `1.0`. The `n == 0` exit in `gemv_sktri` is the
  same.
- Operands are real doubles, column-major by default.
  `MARRAY_DEFAULT_LAYOUT` is set to `COLUMN_MAJOR` in `ltlt.hpp`. The
  induced-method branches around `bli_ind_oper_find_avail` are boilerplate
  carried over from BLIS's own gemm front-end and are dead here, since nothing
  is ever complex.
- The axpyf fusing factor fits in 16. `gemv_sktri` reads `BS` from the context
  and buffers that many values in `double Txj[16]`, and its row-major branch
  does the same with `yi[16]`. Every sub-configuration in the tree sets AF to 5
  or 8 and the reference default is 8, so this holds; but it is an assumption
  about a value the context supplies at run time.
- `ger2` needs a unit-stride dimension in `E`. The fully general case prints
  and exits.
- `packing` assumes `t` is indexed along k. It offsets `t` by `panel_len_off`,
  which is the right axis only because both call sites pack B.

## Open questions

| question | why it matters | how to settle it |
|---|---|---|
| Does `PARALLEL_L2` help, and above what size? | The L2 kernels and BLIS's L3 loops draw on one OpenMP runtime. They alternate rather than nest, but the thread count is shared. | Fix `OMP_NUM_THREADS`, sweep `n` at `--step=1` against `--step=2` for one algorithm pair. |
| Which blocked/unblocked pairing wins at `STEP_5`, and does the answer move with `--bs`? | The pairing is the one choice the enum does not encode. | Sweep `--bs` across the four pairings at fixed `n`. |
| How much of the `STEP_3` to `STEP_4` gain is the vanished temporary versus the vanished pass over `b`? | Decides whether further work belongs in the packing kernel or in reducing allocations. | Compare the `gemm_sktri` and `blas::gemm` timer sections between the two steps. |
| Would a templated `blockRL_var1` at lower steps be informative? | Currently the schedule change and the two kernel changes it assumes cannot be separated. | Not measurable as written; needs `var1` made a template first. |
| What fraction of achievable bandwidth does the level-1f path reach? | Tells us whether `gemv_sktri` is worth further tuning or is already memory-bound. | `PROFILE_FLOPS` is already instrumented there; compare against a measured stream ceiling. |
