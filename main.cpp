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
    // Argument validation
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <matrix_size> <sparsity> [seed]" << endl;
        cout << "Example: " << argv[0] << " 1024 0.9 42" << endl;
        cout << "\nParameters:" << endl;
        cout << "  matrix_size : Dimension N for NxN matrices (e.g., 512, 1024, 2048)" << endl;
        cout << "  sparsity    : Fraction of zeros, range 0.0-1.0 (0.9 = 90% sparse)" << endl;
        cout << "  seed        : Random seed for reproducibility (default: 2025)" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    double sparsity = stod(argv[2]);
    int seed = (argc >= 4) ? stoi(argv[3]) : 2025;

    // Input validation
    if (sparsity < 0.0 || sparsity >= 1.0) {
        cout << "Error: sparsity must be in range [0.0, 1.0)" << endl;
        cout << "Example: 0.9 means 90% zeros (10% nonzeros)" << endl;
        return 1;
    }

    srand(seed);
    
    cout << "Random seed: " << seed << endl;
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
