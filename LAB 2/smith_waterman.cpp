#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int score(char a, char b) {
    return (a == b) ? 2 : -1;
}

int main() {
    string A = "ACACACTA";
    string B = "AGCACACA";

    int N = A.size();
    int M = B.size();
    int gap = 2;

    vector<vector<int>> H(N + 1, vector<int>(M + 1, 0));
    int max_score = 0;

    double start = omp_get_wtime();

    // Wavefront parallelization
    for (int k = 2; k <= N + M; k++) {
        #pragma omp parallel for reduction(max:max_score) schedule(static)
        for (int i = 1; i <= N; i++) {
            int j = k - i;
            if (j >= 1 && j <= M) {
                int match  = H[i-1][j-1] + score(A[i-1], B[j-1]);
                int delete_gap = H[i-1][j] - gap;
                int insert_gap = H[i][j-1] - gap;

                H[i][j] = max({0, match, delete_gap, insert_gap});
                max_score = max(max_score, H[i][j]);
            }
        }
        #pragma omp barrier
    }

    double end = omp_get_wtime();

    cout << "Maximum Alignment Score = " << max_score << endl;
    cout << "Execution Time = " << end - start << " seconds" << endl;

    return 0;
}
