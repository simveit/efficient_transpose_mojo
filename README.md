# Effective matrix transpose

Improve matrix transpose step by step in Mojo.

Please see [my blogpost](http://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) for a detailed explanation.

## Performance Comparison

| Kernel | Bandwidth (GB/s) | % of Max Bandwidth | Implementation |
|--------|------------------|-------------------|----------------|
| transpose_naive | 1056.08 | 32.0025% | Mojo |
| transpose_swizzle | 1437.55 | 43.5622% | Mojo |
| transpose_swizzle_batched | 2775.49 | 84.1056% | Mojo |