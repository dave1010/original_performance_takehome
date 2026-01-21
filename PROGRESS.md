# Progress

## Progress entries
- SIMD vectorization with batched VLIW scheduling, pre-broadcasted hash constants, and single-core configuration — 16465 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Processed blocks in groups of two to reuse vector scratch while packing more SIMD ops per bundle — 8785 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Processed blocks in groups of three to further increase ILP and VLIW packing — 7665 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Processed blocks in groups of four to push VLIW packing further — 6737 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Processed blocks in groups of five to improve packing density — 6577 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Processed blocks in groups of six for the best packing so far — 5745 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Tried grouping eight blocks (worse packing) — 6289 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Moved the rounds loop inside the block group to reuse precomputed addresses — 5565 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Retested grouping five blocks with the loop reordering (still worse) — 6367 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Retested grouping seven blocks with the loop reordering (worse) — 6783 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Loaded idx/val once per block group and stored only after all rounds — 4605 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Replaced idx = idx*2 + choice with a single multiply_add after computing choice — 4509 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Dropped unused zero broadcast constant — 4508 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Added a dependency-aware, cross-engine scheduler to pack ready load/valu/alu/store slots into the same bundles while respecting scratch/memory hazards and pause barriers, replacing phase-based emission with mixed bundles — 2620 cycles (python perf_takehome.py Tests.test_kernel_cycles).
- Added a per-block hash/update helper and restructured the round loop to interleave each block’s gather with the previous block’s hash/update stage for explicit software pipelining across blocks — 2680 cycles (python perf_takehome.py Tests.test_kernel_cycles).
