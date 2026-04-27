# Assignment 8: GPU Accelerated Machine Learning


## Files Included

| File                        | Exercise | Description |
|-----------------------------|----------|-------------|
| 'Q1`  | Problem 1 | Vector operations, bandwidth benchmark, launch config analysis |
| `Q2`  | Problem 2 | Shared memory reduction, bank conflicts, warp shuffle |
| `Q3`  | Problem 3 | Activations (Sigmoid, Tanh, Leaky ReLU), ReLU backward, BCE & Cross-Entropy loss |
| `Q4`  | Problem 4 | Tiled GEMM, MaxPool2x2, BatchNorm inference |
| `Q5`  | Problem 5 | MNIST CNN training pipeline simulation |
And a detailed report with outputs.

Summary of Work Done

-Problem 1: Full bandwidth benchmark + launch configuration analysis (block sizes 64–1024)
-Problem 2: Naive, Shared Memory Tree, and Warp Shuffle reductions + bank conflict timing experiment
-Problem 3: Complete activation suite + BCE & Cross-Entropy loss kernels with verification
-Problem 4: Tiled MatMul, MaxPool2x2, and BatchNorm inference kernels
-Problem 5: Simulated full MNIST CNN training loop showing decreasing loss and increasing accuracy

