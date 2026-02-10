#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>

struct Particle {
    double x, y, z;
};

int main() {
    const int N = 2000;
    const double epsilon = 1.0;
    const double sigma = 1.0;

    std::vector<Particle> pos(N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);

    // Initialize particles (example)
    for (int i = 0; i < N; i++) {
        pos[i] = {drand48(), drand48(), drand48()};
    }

    double total_energy = 0.0;
    double start = omp_get_wtime();

    int threads = omp_get_max_threads();
    std::vector<std::vector<double>> fx_local(threads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> fy_local(threads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> fz_local(threads, std::vector<double>(N, 0.0));

    #pragma omp parallel reduction(+:total_energy)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = pos[i].x - pos[j].x;
                double dy = pos[i].y - pos[j].y;
                double dz = pos[i].z - pos[j].z;

                double r2 = dx*dx + dy*dy + dz*dz;
                double inv_r2 = 1.0 / r2;
                double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                double inv_r12 = inv_r6 * inv_r6;

                // Potential energy
                double pe = 4 * epsilon * (inv_r12 - inv_r6);
                total_energy += pe;

                // Force magnitude
                double fmag = 24 * epsilon * (2*inv_r12 - inv_r6) * inv_r2;

                fx_local[tid][i] += fmag * dx;
                fy_local[tid][i] += fmag * dy;
                fz_local[tid][i] += fmag * dz;

                fx_local[tid][j] -= fmag * dx;
                fy_local[tid][j] -= fmag * dy;
                fz_local[tid][j] -= fmag * dz;
            }
        }
    }

    // Merge thread-local forces
    for (int t = 0; t < threads; t++) {
        for (int i = 0; i < N; i++) {
            fx[i] += fx_local[t][i];
            fy[i] += fy_local[t][i];
            fz[i] += fz_local[t][i];
        }
    }

    double end = omp_get_wtime();

    std::cout << "Total Potential Energy = " << total_energy << "\n";
    std::cout << "Execution Time = " << end - start << " seconds\n";

    return 0;
}
