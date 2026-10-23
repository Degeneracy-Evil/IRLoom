# Pinned Affine cleanup correction

`affine-cleanup-positional.patch` applies only to LLVM/MLIR
`26eb4285b56edd8c897642078d91f16ff0fd3472`'s
`mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp` (Apache-2.0 with LLVM exception).
The original source, binaries and archives remain untouched.

The old cleanup builder replaces operands by **SSA value**, reading a live init
operand range. Shared initial values collapse independent carries; an initial
value reused as a bound can also rewrite that bound. The fix first redirects
external result users, then assigns each cleanup init **by operand position**.
No action, program, seed or node-ID blacklist is used.

```bash
bash scripts/prepare-correctness.sh
```

This recompiles one copied translation unit and links it before the matching
historical archive and bridge object, into
`benchmark-data/architecture/compilers/cleanup-positional-v1`. The link map checks
that the old LoopUtils archive member was not extracted. It is not a full LLVM
bootstrap and does not patch the historical `polygeist-opt` executable.

`affine-cleanup.mlir` is a source-independent integer reproducer. Its core is
the two-carry `same_init` function; other functions control distinct initial
values, bound aliases and constant odd/even trip counts. The C++ runner checks
only positive sizes 1–300. The historical official affine-unroll pass also
reproduces the defect without IRLoom's bridge or frontend. These are diagnostic
fixtures, never training programs or performance labels.

`inspect-loops.cpp` is a read-only structural classifier using the same SDK.
It records precise cleanup need, initializer aliases and structural operation
paths. It does not feed additional features to the models.

`wrong-output.cpp` deliberately corrupts only private candidate executables in
the guarded integration audit. It is never a benchmark or an oracle.
