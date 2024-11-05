# matmul_kernel

Run C code:
```clang -O3 -mno-avx512f -march=native -DTEST -DNITER=100 matmul_kernel.c -o matmul_kernel.out && ./matmul_kernel.out```
Run mojo code:
```magic run mojo build matmul_kernel.mojo && ./matmul_kernel```
