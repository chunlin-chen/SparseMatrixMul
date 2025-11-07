#include "SparseMatrix.hpp"
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

vector<vector<int>> generateSparseMatrix(int N, double sparsity) {
    vector<vector<int>> mat(N, vector<int>(N, 0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if ((rand() / (double)RAND_MAX) > sparsity)
                mat[i][j] = rand() % 10 + 1;
    return mat;
}

int main(int argc, char* argv[]) {

    srand(2025);
    int N = stoi(argv[1]);
    double sparsity = stod(argv[2]);


    cout << "Matrix size: " << N << "x" << N << endl;
    cout << "Sparsity: " << sparsity * 100 << "%\n";

    auto matA = generateSparseMatrix(N, sparsity);
    auto matB = generateSparseMatrix(N, sparsity);

    // ===== Dense (baseline) =====
    auto start_dense = chrono::high_resolution_clock::now();
    vector<vector<double>> matDense(N, vector<double>(N, 0.0));
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            for (int k=0;k<N;k++)
                matDense[i][j] += matA[i][k]*matB[k][j];
    auto end_dense = chrono::high_resolution_clock::now();
    double time_dense = chrono::duration<double, milli>(end_dense - start_dense).count();

    // ===== Convert to Sparse =====
    SparseMatrixCSR A(N, N), B(N, N);
    for (auto &row : matA) {
        vector<pair<int,double>> sparseRow;
        for (int j=0;j<N;++j)
            if (row[j] != 0) sparseRow.push_back({j, (double)row[j]});
        A.addRow(sparseRow);
    }
    for (auto &row : matB) {
        vector<pair<int,double>> sparseRow;
        for (int j=0;j<N;++j)
            if (row[j] != 0) sparseRow.push_back({j, (double)row[j]});
        B.addRow(sparseRow);
    }

    // ===== Single-thread Sparse =====
    auto start_single = chrono::high_resolution_clock::now();
    auto C_single = A.multiplySparse(B);
    auto end_single = chrono::high_resolution_clock::now();
    double time_single = chrono::duration<double, milli>(end_single - start_single).count();

    // ===== Multi-thread Sparse =====
    auto start_parallel = chrono::high_resolution_clock::now();
    auto C_parallel = A.multiplySparseParallel(B);
    auto end_parallel = chrono::high_resolution_clock::now();
    double time_parallel = chrono::duration<double, milli>(end_parallel - start_parallel).count();

    // ===== Output result =====
    cout << "Dense time:         " << time_dense << " ms\n";
    cout << "Sparse single:      " << time_single << " ms\n";
    cout << "Sparse parallel:    " << time_parallel << " ms\n";
    cout << "Speedup vs dense:   " << time_dense / time_parallel << "x\n";
    cout << "Speedup vs single:  " << time_single / time_parallel << "x\n";

    // ===== Verify correctness =====
    auto C_full = C_parallel.toFullMatrix();
    bool pass = true;
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            if (abs(C_full[i][j] - matDense[i][j]) > 1e-6) {
                cout << "Mismatch at (" << i << "," << j << ")\n";
                pass = false;
                break;
            }

    if (pass) cout << "PASS" << endl;
    else cout << "FAIL" << endl;

    return 0;
}
