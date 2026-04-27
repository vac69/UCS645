UCS645 – Parallel & Distributed Computing
Assignment 5: MPI (Blocking vs Non-Blocking Communication)
📌 Introduction

This repository contains implementations of multiple MPI programs designed to demonstrate core concepts of parallel and distributed computing. 
The focus is on communication techniques, performance analysis, and task distribution strategies.

⚙️ Concepts Covered
Blocking vs Non-blocking communication in MPI
Collective operations such as MPI_Bcast and MPI_Reduce
Performance measurement using MPI_Wtime()
Dynamic load balancing using the master-worker model
📁 Project Structure
Task	Source File	Description
Q1	q1_daxpy.c	Parallel DAXPY computation with speedup analysis
Q2	q2_broadcast.c	Comparison of custom broadcast vs MPI_Bcast
Q3	q3_dotproduct.c	Distributed dot product with scaling discussion
Q4	q4_primes.c	Prime number generation using master-worker model
Q5	q5_perfect.c	Perfect number detection using dynamic distribution
—	Makefile	Simplified build and execution commands
—	Assignment_5_UCS645.pdf	Detailed written report
▶️ Compilation & Execution
Using Makefile
make        # Compile all programs
make run1   # Execute Question 1
make run2   # Execute Question 2
make run3   # Execute Question 3
make run4   # Execute Question 4
make run5   # Execute Question 5
Manual Compilation
mpicc -o program_name file_name.c
mpirun --allow-run-as-root --oversubscribe -np 4 ./program_name
📊 Key Insights
Built-in MPI functions like MPI_Bcast are more efficient due to optimized internal implementations (e.g., tree-based communication).
Increasing the number of processes improves performance up to a limit, after which communication overhead dominates.
Parallel efficiency decreases as processes increase if workload is not sufficiently large.
The master-worker approach is effective for problems with uneven or unpredictable workloads.
✅ Summary

This assignment highlights how parallel programming can significantly enhance computational performance. 
However, achieving optimal results depends on factors such as communication cost, workload balance, and algorithm design.
 Efficient use of MPI primitives plays a crucial role in scalability.
