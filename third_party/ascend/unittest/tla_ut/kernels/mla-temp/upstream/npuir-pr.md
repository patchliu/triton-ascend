Paged attention needs fixed-stage Cube/Vector overlap and explicit BF16 L0 matrix products with FP32 accumulation. This branch carries the complete compiler support used by the final Paged MLA implementation.

- Validate and adapt TLA stages to preload scopes, reusing existing synchronization.
- Integrate the L0 operations from !4678 with the current upstream memory planner.
- Preserve the upstream pool-fallback random seed while enabling L0 planning.
- Instantiate BF16 matrix templates and select their matching ABI entry points.
- Include stage validation, preload boundary, memory planning, conversion, FP16/BF16, and mixed-type rejection tests.

The final kernels passed full-length and ragged output/LSE checks with the previously packaged compiler. A fresh build of this branch against current master is running. Targeted lit tests will be recorded before this draft is marked ready.

Requires the matching Triton Ascend branch for the frontend APIs. No LLVM submodule changes are included.
