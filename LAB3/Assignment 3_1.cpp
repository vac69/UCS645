#include <iostream>
#include <format>
#include <random>
#include <omp.h>
#include "correlate.hpp"

auto generate_random_number() -> double {
        static thread_local std::mt19937 mt{std::random_device{}()};
        static thread_local std::uniform_real_distribution<double> range{-1.0, 1.0};
        return range(mt);
}

auto main(int argc, char** argv) -> int {
        if(argc != 3) {
                std::cout<<std::format("Usage: ./correlate_matrix <number_of_rows> <number_of_columns>\n");
                return -1;
        }
        int rows{std::atoi(argv[1])};
        int cols{std::atoi(argv[2])};
        int max_threads{omp_get_max_threads()};
        std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
        std::vector<double> flat_array(rows * cols);

        #pragma omp parallel for collapse(2)
        for(int i = 0; i<rows; ++i) {
                for(int j = 0; j<cols; ++j) {
                        double num = generate_random_number();
                        matrix[i][j] = num;
                        flat_array[i*cols + j] = num;
                }
        }

        std::cout<<std::format("Using No Threading\n");
        auto seq_time{correlate_matrix_sequential(matrix)};
        std::cout<<std::format("Execution time: {:.2f}ms\n\n",seq_time.count());

        std::cout<<std::format("Using Threading with 2D Heap Allocated Array\n\n");
        for(int num_threads{2}; num_threads<=max_threads; num_threads += 2){
                std::cout<<std::format("With {} threads\n", num_threads);
                omp_set_num_threads(num_threads);
                auto unoptimized_par_time{correlate_matrix_parallel_2d_array(matrix)};
                std::cout<<std::format("Execution time: {:.2f}ms\n", unoptimized_par_time.count());
                std::cout<<std::format("Speed Up: {:.2f}x\n\n", seq_time.count()/unoptimized_par_time.count());
        }

        std::cout<<std::format("Using Threading with Flat Array\n\n");
        for(int num_threads{2}; num_threads<=max_threads; num_threads += 2){
                std::cout<<std::format("With {} threads\n", num_threads);
                omp_set_num_threads(num_threads);
                auto optimized_par_time{correlate_matrix_parallel_2d_array(matrix)};
                std::cout<<std::format("Execution time: {:.2f}ms\n", optimized_par_time.count());
                std::cout<<std::format("Speed Up: {:.2f}x\n\n", seq_time.count()/optimized_par_time.count());
        }
        return 0;
}
