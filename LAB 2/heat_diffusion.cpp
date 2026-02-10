#include <omp.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    const int NX = 500;
    const int NY = 500;
    const int STEPS = 500;
    const double alpha = 0.1;

    vector<vector<double>> temp(NX, vector<double>(NY, 0.0));
    vector<vector<double>> temp_new(NX, vector<double>(NY, 0.0));

    // Initial condition: heat source at center
    temp[NX / 2][NY / 2] = 100.0;

    double start = omp_get_wtime();

    // Time stepping loop
    for (int t = 0; t < STEPS; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                temp_new[i][j] =
                    temp[i][j] +
                    alpha * (
                        temp[i + 1][j] + temp[i - 1][j] +
                        temp[i][j + 1] + temp[i][j - 1] -
                        4.0 * temp[i][j]
                    );
            }
        }
        temp.swap(temp_new);
    }

    // Reduction: total heat in the plate
    double total_heat = 0.0;

    #pragma omp parallel for reduction(+:total_heat)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            total_heat += temp[i][j];
        }
    }

    double end = omp_get_wtime();

    cout << "Final center temperature = "
         << temp[NX / 2][NY / 2] << endl;
    cout << "Total Heat Energy = " << total_heat << endl;
    cout << "Execution Time = " << end - start << " seconds" << endl;

    return 0;
}
