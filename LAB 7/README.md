# Assignment 7b

## Files Included
Key Concepts Covered
CUDA kernel design and execution
Thread and block configuration
Static device memory using device variables
Memory transfer using cudaMemcpyToSymbol and cudaMemcpyFromSymbol
Parallel computation using atomic operations
GPU vs CPU performance comparison
CUDA event-based timing
Theoretical vs measured memory bandwidth
Profiling using nvprof
Results Summary

Problem 1:
Both iterative and formula methods produced the same result (524800), validating correctness.

Problem 2:
CPU merge sort performed faster than GPU for small input size (N = 1000). GPU overhead dominates for smaller datasets.

Problem 3:
Kernel execution time ≈ 0.019 ms
Theoretical Bandwidth ≈ 320 GB/s
Measured Bandwidth ≈ 145–165 GB/s
Efficiency ≈ 45%–51%


Conclusion

The assignment demonstrates effective use of CUDA for parallel computation.
It highlights the advantages of GPU acceleration for large-scale problems and shows how memory bandwidth and profiling impact real-world performance.

