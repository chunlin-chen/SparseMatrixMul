# Sparse Matrix Multiplication (SpGEMM)

High-performance sparse matrix-matrix multiplication using CSR format with OpenMP parallelization.

## Features

- CSR (Compressed Sparse Row) sparse matrix storage
- Single-threaded baseline implementation
- Multi-threaded OpenMP parallelization with dynamic scheduling
- Dense matrix baseline for correctness verification
- Automatic performance benchmarking and speedup analysis

## Build
```bash
g++ -std=c++17 -O2 -fopenmp main.cpp SparseMatrix.cpp -o sparse_test
```

## Usage
```bash
./sparse_test <matrix_size> <sparsity>
```

Parameters:
- `matrix_size`: Dimension N for NxN matrices (e.g., 512, 1024, 2048)
- `sparsity`: Fraction of zero elements, range 0.0-1.0 (0.9 = 90% zeros)

## Example
```bash
./sparse_test 1024 0.9
```

Output:
```
Matrix size: 1024x1024
Sparsity: 90.0%
Dense time:         12847.3 ms
Sparse single:      245.7 ms
Sparse parallel:    42.1 ms
Speedup vs dense:   305.2x
Speedup vs single:  5.8x
PASS
```

## Implementation Details

Uses CSR format with three arrays:
- `values[]` - nonzero element values
- `colIndex[]` - column indices of nonzeros
- `rowPtr[]` - starting index for each row

Parallelization strategy:
- OpenMP parallel for with dynamic scheduling
- Private hash maps per thread for lock-free accumulation
- Each thread processes independent output rows

## Requirements

- C++17 compiler (g++ 7.0+)
- OpenMP support
