# SparseMatrixMul

Sparse matrix multiplication project using CSR (Compressed Sparse Row) format

Supports:
- Dense baseline multiplication
- Sparse matrix multiplication (CSR)
- Single-thread and Multi-thread (OpenMP)

```bash
g++ -std=c++17 -O2 -fopenmp main.cpp SparseMatrix.cpp -o sparse_test

./sparse_test <size> <sparsity>

Example:
./sparse_test 512 0.8
```
