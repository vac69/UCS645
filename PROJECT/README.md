Course: Parallel and Distributed Computing
High-Performance Grid-Based Environmental Spread Modeling Using Multithreaded CPU Parallelism


Parallel and Distributed Computing is essential for accelerating large-scale computational
simulations. Many natural phenomena, such as heat diffusion and forest fire propagation, can
be modeled using grid-based numerical techniques where each cell updates its state based on
neighboring values over discrete time steps.


This project presents the design and implementation of a parallel simulation framework using
C++ and OpenMP to model spatial diffusion and combustion dynamics on a two-dimensional
grid. The environment is discretized into cells representing temperature values or fire states
(empty, vegetation, burning). At each iteration, cell states are updated according to predefined
mathematical or probabilistic rules.

Since cell updates within a time step are independent, OpenMP is used to parallelize
computations across multiple CPU cores. The system compares sequential and parallel
implementations to evaluate execution time, speedup, and scalability under varying grid sizes
and thread counts.

The objective of this project is to demonstrate performance improvement through shared-
memory parallel programming and to analyze scalability behavior in grid-based simulations.
