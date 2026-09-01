# BLAS and BLIS: what the APIs offer

This project calls BLIS at three different depths, because the operations it
needs have no BLAS equivalent and the fusion in `FUSED_L3` cannot be expressed
at BLAS's only interface. This note covers what each library offers and where
[`optimization-steps.md`](optimization-steps.md) lands in it.

## BLAS: "level" is arithmetic intensity, and there is one API

The three levels were fixed in 1979, 1988, and 1990. They classify operations
by the ratio of flops to memory traffic, nothing else.

| level | shape | data | flops | ratio |
|---|---|---|---|---|
| 1 | vector-vector (`daxpy`, `ddot`, `idamax`) | O(n) | O(n) | O(1) |
| 2 | matrix-vector (`dgemv`, `dger`, `dsymv`, `dtrsv`) | O(n²) | O(n²) | O(1) |
| 3 | matrix-matrix (`dgemm`, `dsyrk`, `dtrsm`) | O(n²) | O(n³) | O(n) |

Only level 3 has reuse to exploit, which is why it is the only level where
blocking and packing repay their cost, and why algorithms get recast in terms
of level-3 operations wherever possible. That recasting is what separates
`ltlt_blockRL` from `ltlt_unblockRL`.

The interface, by contrast, has no layers. One flat Fortran-77 surface:

- column-major only, one `lda` per matrix
- everything by pointer, because Fortran passes by reference
- `char` flags (`'N'/'T'/'C'`, `'U'/'L'`, `'N'/'U'`)
- datatype welded into the symbol name, so no runtime dispatch
- parallelism set globally, out of band, by environment variable
- a *closed* structure vocabulary: general, symmetric, hermitian, triangular,
  plus banded and packed storage

CBLAS sits on top as a wrapper, adding a row-major enum and scalar returns. It
is not a second layer.

The closed vocabulary is the first hard wall. BLAS has no skew-symmetric
structure and no mechanism to add one.

## BLIS: two independent axes

BLIS separates what the operation is from how you ask for it, and stratifies
both.

### Axis 1: operation levels

From `external/blis/docs/BLISTypedAPI.md`:

| level | what it covers | BLAS analogue |
|---|---|---|
| 1v | vector-vector, plus `axpbyv`, `xpbyv`, `invertv`, `setv` | level 1 |
| 1d | operations on a matrix diagonal: `addd`, `scald`, `setd`, `shiftd` | none |
| 1m | elementwise on matrices: `copym`, `addm`, `scalm`, `setm` | none |
| 1f | fused kernels: `axpyf`, `dotxf`, `dotxaxpyf`, `axpy2v` | none |
| 2 | matrix-vector | level 2 |
| 3 | matrix-matrix, plus `gemmt` | level 3 |
| utility | `asumv`, `norm1/f/i`, `randm`, `sumsqv`, `mkherm`, `printm` | partial |
| microkernels | `gemm_ukr`, `trsm_ukr`, `gemmtrsm_ukr`, pack kernels | none |

Level-1f has no BLAS counterpart at all. Level-2 operations are
bandwidth-bound, so `axpyf` performs several columns' worth of `axpy` in one
pass and loads `y` once instead of *f* times. `src/ltlt_blas.cpp:291` and
`:495` pull `BLIS_AXPYF_KER` and `BLIS_AXPY2V_KER` straight out of the context;
that is the whole of `FUSED_L2`.

BLIS also publishes its bottom edge, the microkernels, as a contract you may
replace. BLAS's blocked implementation is opaque.

This fork adds the skew-symmetric family (`skmv`, `skr2`, `skr2k`, `shmv`) at
levels 2 and 3, which is why it is vendored rather than linked.

### Axis 2: API layers

Outermost to innermost. BLAS has nothing corresponding to this axis.

| # | layer | entry point | what it buys |
|---|---|---|---|
| 1 | BLAS compat | `dgemm_` | drop-in for existing code |
| 2 | CBLAS compat | `cblas_dgemm` | row-major C interface |
| 3 | typed | `bli_dgemm` | both strides, split conj/trans, enum parameters |
| 4 | object | `bli_gemm` on `obj_t` | runtime datatype, structure, uplo, diagonal offset |
| 5 | expert | `_ex` suffix | `cntx_t*` (kernels, blocksizes), `rntm_t*` (per-call threading) |
| 6 | control tree | `gemm_cntl_t` + `bli_l3_thread_decorator` | substitute pack kernels and loop variants |
| 7 | sandbox / addon | build-time | replace level-3 entirely |

Layer 3, typed, keeps one function per datatype but changes the contract. Each
operand carries *both* a row and a column stride, so column-major, row-major,
and general-strided all work unmodified, and a transpose costs a stride swap.
`trans_t` and `conj_t` are separate parameters, so conjugation without
transposition is expressible; BLAS conflates the two in `'C'`.

Layer 4, object, moves every attribute inside `obj_t`: datatype, dimensions,
both strides, structure, uplo, conj/trans flags, diagonal offset, packing
state. Datatype becomes a runtime field rather than part of the symbol name,
which is what makes mixed-datatype `gemm` and the induced complex methods
expressible, hence `bli_ind_oper_find_avail` in `ltlt_blas.cpp`. Attaching an
existing buffer is free.

Layer 5, expert, adds `cntx_t*` and `rntm_t*`, with `NULL` for either
recovering basic behaviour (`external/blis/docs/BLISTypedAPI.md:117`). The
`rntm_t` matters when `gemm` is called from inside an already-parallel region:
BLAS forces a global threading decision, BLIS lets one call site request its
own.

Layer 6, the control tree, is undocumented. `bli_gemm_cntl_init` builds a
`gemm_cntl_t` and `bli_l3_thread_decorator` runs it
(`external/blis/frame/3/bli_l3_decor.h:41`). The tree is the five-loop nest
reified as data: which variant runs at each loop, which pack kernel is called,
what parameters reach it.

## Where this project sits

| step | layer used | why |
|---|---|---|
| unfused level-3 | 3, via MArray's `blas::gemm` | copy `b`, `sktrmm` the copy, then `gemm` (`src/ltlt_blas.cpp:66`) |
| `FUSED_L2` | operation level 1f, kernels fetched from `cntx_t` | `T*x` never materializes |
| `FUSED_L3` | 4 and 6 | own pack-B kernel plus `skparams` |

`src/ltlt.hpp:16` defines `MARRAY_USE_BLIS`, which routes MArray's `blas::`
overloads onto the typed API instead of Fortran BLAS symbols. Without that
define the same calls fall through to `dgemm_` and still land in BLIS via layer
1, losing the stride flexibility on the way.

`FUSED_L3` is what the layering is for. BLIS packs `B` into its level-3 buffer
either way; replacing the pack kernel applies the tridiagonal `T` during a copy
that was already going to happen, so the temporary and the extra pass both
disappear. No BLAS interface can express that, at any level or through any
vendor extension, which is why the unfused path pays for a panel-sized
temporary.

## Head to head

| | BLAS | BLIS |
|---|---|---|
| storage | column-major, one `lda`; separate banded/packed routines | both strides, any layout; no banded or packed |
| transpose | flags on each routine | stride swap, free |
| conj vs. trans | conflated in `'C'` | independent parameters |
| datatype | in the symbol name | in the name, or a runtime field |
| threading | global, environment variable | global, or per-call via `rntm_t` |
| structure | fixed, closed | open; this fork added skew-symmetric |
| extension point | none | microkernels, control tree, sandboxes, addons |
| stability | frozen since 1990 | public API stable, internals are not |

The costs run the other direction as the layers descend. The object API does
per-call setup the typed API skips, and layer 6 carries no ABI guarantee.
`gemm_cntl_t` and `bli_gemm_cntl_init` do not exist in released BLIS 0.9.0,
though `external/blis/CHANGELOG` is tagged 0.9.0: the vendored tree is a fork
carrying post-0.9 development work plus the skew-symmetric additions. Vendoring
is the price of using layer 6.

One thing BLAS has that BLIS dropped on purpose: banded and packed storage. The
compat layer forwards `dgbmv`, but the native API has no equivalent.
