Paged attention needs explicit L0 operand reuse together with compiler-managed Cube/Vector task stages. This branch provides the frontend capabilities used by the final BF16 Paged MLA kernels.

- Carry the prerequisite L0 API from #2982 with its original author preserved.
- Refine execution scheduling and shared allocation planning.
- Accept BF16 L0 operands with FP32 accumulation and reject mixed operand types.
- Add ND2NZ row windows with static diagnostics and FP16/BF16 round-trip coverage.
- Expose per-kernel optimized-metaop selection and automatic synchronization control. Dynamic cross-core flag IDs are already supported by the current upstream base.

The operator-specific O2 experiment environment variable is replaced by the public compile option. Existing defaults remain CANN-version dependent.

Tests include frontend contracts, resident-Q reuse, ND2NZ windows, and option cache keys. The final MLA kernels passed full-length and ragged output/LSE checks with the packaged compiler. Fresh builds and tests of these rebased upstream branches are running. This PR remains a draft while that validation completes.

The matching AscendNPU-IR branch adds fixed-stage lowering, the prerequisite L0 operations, and BF16 templates.
