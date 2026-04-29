#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>

using namespace std;

const int N = 300;
const int STEPS = 100;

vector<vector<double>> grid(N, vector<double>(N));
vector<vector<double>> newGrid(N, vector<double>(N));

void initialize() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = rand() % 100;
}

void updateParallel() {
    #pragma omp parallel for
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            newGrid[i][j] = (
                grid[i][j] +
                grid[i-1][j] +
                grid[i+1][j] +
                grid[i][j-1] +
                grid[i][j+1]
            ) / 5.0;
        }
    }
    grid = newGrid;
}

int main() {
    initialize();

    double start = omp_get_wtime();

    for (int step = 0; step < STEPS; step++) {
        updateParallel();
    }

    double end = omp_get_wtime();

    cout << "Execution Time: " << (end - start) << endl;

    return 0;
}
