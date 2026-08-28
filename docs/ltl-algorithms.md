# How the variants differ

Every routine in `src/` computes the same factorization: given skew-symmetric
`X`, find unit lower triangular `L` and skew-symmetric tridiagonal `T` with
`X = L * T * L'`. What differs is the schedule, meaning when each piece of
arithmetic happens and how much data it touches at a time. This note covers
those choices. The code and the MATLAB reference cover the rest.

## Three axes

The variants are a cross product. A file name tells you where its routine
sits.

Right-looking or left-looking. A right-looking algorithm finishes the current
column and immediately pushes its contribution out over the whole remaining
matrix. A left-looking one does nothing until it has to: before touching a
column it reaches back and pulls in everything owed to it. Same factorization,
same flop count, different memory traffic.

Unblocked or blocked. Unblocked algorithms advance a column at a time using
vector and matrix-vector operations. Blocked ones advance a panel at a time:
hand the panel to an unblocked routine, then express the trailing update as a
matrix-matrix product, which is where the hardware runs fastest. The two levels
compose, since any unblocked variant can factor the panel for any blocked one,
so the question worth asking is which pairings win. The debug driver tries all
of them. One variant sits between the two levels and advances two columns per
step: still vector-level arithmetic, but half as many passes over the trailing
matrix.

Pivoted or not. Without pivoting the factorization breaks down when a pivot
comes out small or zero. The pivoted variants pick a pivot each step and
permute symmetrically so the skew structure survives, paying speed and a
permutation vector for stability. Each one mirrors an unpivoted routine closely
enough to be read alongside it.

## The layer underneath

The factorizations are written against a handful of operations that have no
BLAS equivalent, because the middle factor is both tridiagonal and skew and no
library exposes that shape. Keeping those in one place is what lets the
factorizations stay close to their MATLAB originals: they read as partitioned
linear algebra instead of loops, and the tuning happens below them.

That layer is also where the algorithms stop being purely mathematical.
Applying `T` to a panel can ride along with data movement a matrix-matrix
product already performs, so it costs almost nothing. The trick only exists one
level below the factorization, and it accounts for a good share of the speedup.

## Why they all stay

There is no single winner to find here. The point is to measure: what
left-looking buys over right-looking at a given size, what a blocked algorithm
gains from each choice of panel factorization, what pivoting costs. So old
variants stay, and each is instantiated at all six optimization steps (see the
README) instead of being replaced by its improved form.
