#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <chrono>

using namespace std;
using Grid = vector<vector<int>>;

const int EMPTY = 0;
const int TREE = 1;
const int BURNING = 2;

Grid initialize_grid(int rows, int cols, double density = 0.7) {
    Grid g(rows, vector<int>(cols, EMPTY));
    srand(42);  // fixed seed
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (static_cast<double>(rand()) / RAND_MAX < density) {
                g[i][j] = TREE;
            }
        }
    }
    if (rows > 0 && cols > 0) g[0][cols/2] = BURNING;
    return g;
}

bool has_burning_neighbor(const Grid& g, int x, int y, int rows, int cols) {
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d], ny = y + dy[d];
        if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && g[nx][ny] == BURNING)
            return true;
    }
    return false;
}

void update_grid_sequential(const Grid& current, Grid& next, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if (current[i][j] == BURNING) next[i][j] = EMPTY;
            else if (current[i][j] == TREE)
                next[i][j] = has_burning_neighbor(current, i, j, rows, cols) ? BURNING : TREE;
            else next[i][j] = EMPTY;
        }
}

void update_grid_parallel(const Grid& current, Grid& next, int rows, int cols) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if (current[i][j] == BURNING) next[i][j] = EMPTY;
            else if (current[i][j] == TREE)
                next[i][j] = has_burning_neighbor(current, i, j, rows, cols) ? BURNING : TREE;
            else next[i][j] = EMPTY;
        }
}

double run_simulation(int rows, int cols, int max_steps, int threads, bool use_parallel) {
    omp_set_num_threads(threads);
    Grid current = initialize_grid(rows, cols);
    Grid next(rows, vector<int>(cols, EMPTY));

    auto start = chrono::high_resolution_clock::now();
    for (int step = 0; step < max_steps; ++step) {
        if (use_parallel)
            update_grid_parallel(current, next, rows, cols);
        else
            update_grid_sequential(current, next, rows, cols);
        current.swap(next);
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

int main() {
    cout << fixed << setprecision(4);
    vector<int> sizes = {200, 400, 600};
    vector<int> threads_list = {1, 2, 4, 8};
    int max_steps = 200;

    cout << "Grid Size | Threads | Parallel | Time (s) | Speedup\n";
    cout << "----------|---------|----------|----------|--------\n";

    for (int size : sizes) {
        double seq_time = run_simulation(size, size, max_steps, 1, false);
        for (int t : threads_list) {
            double par_time = run_simulation(size, size, max_steps, t, true);
            double speedup = seq_time / par_time;
            cout << size << "x" << size << " | " << setw(7) << t
                 << " | Yes      | " << setw(8) << par_time
                 << " | " << setw(7) << speedup << endl;
        }
        cout << size << "x" << size << " | " << setw(7) << 1
             << " | No       | " << setw(8) << seq_time
             << " | 1.0000" << endl;
    }
    return 0;
}
