// clang-17 -O2 -mno-avx512f -march=native -DNITER=100 -fopenmp matmul_parallel.c -o matmul_parallel.out && ./matmul_parallel.out
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEM_ALIGN 64

#define MR 16
#define NR 6
#define NTHREADS 8

#define MC MR* NTHREADS * 1
#define NC NR* NTHREADS * 80
#define KC 1000

#ifndef MDIM
#define MDIM 1500
#endif

#ifndef NDIM
#define NDIM 1200
#endif

#ifndef KDIM
#define KDIM 1800
#endif

#ifndef NITER
#define NITER 100
#endif

#define min(x, y) ((x) < (y) ? (x) : (y))

// Uncomment this section if <float.h> isn't supported in your environment
#define DBL_MAX 1.7976931348623157e+308
#define DBL_MIN 2.2250738585072014e-308

static float blockA_packed[MC * KC] __attribute__((aligned(MEM_ALIGN)));
static float blockB_packed[NC * KC] __attribute__((aligned(MEM_ALIGN)));


void print_row(float* mat, const int row_index, const int cols) {
    for (int j = 0; j < 3; j++) {
        printf("%f, ", mat[row_index * cols + j]);
    }
    printf("..., ");
    for (int j = cols - 3; j < cols; j++) {
        printf("%f, ", mat[row_index * cols + j]);
    }
    printf("\n");
}

// Main function to pretty print the matrix
void pretty_print(float* mat, const int rows, const int cols) {
    // Print the first 3 rows
    for (int i = 0; i < 3 && i < rows; i++) {
        print_row(mat, i, cols);
    }

    // Print ellipsis if there are more than 6 rows
    if (rows > 6) {
        printf("...\n");
    }

    // Print the last 3 rows if there are more than 3 rows
    for (int i = rows - 3; i < rows; i++) {
        if (i >= 3) {
            print_row(mat, i, cols);
        }
    }
}

void print_C_buffer(__m256 C_buffer[], int size) {
    // Each __m256 contains 8 floats, so we need to print all elements within each __m256.
    float temp[8];  // Temporary array to store values from __m256 registers

    for (int i = 0; i < size; i++) {
        // Store values of C_buffer[i] into a float array
        _mm256_storeu_ps(temp, C_buffer[i]);

        // Print each element in the __m256 register
        printf("C_buffer[%d]: ", i);
        for (int j = 0; j < 8; j++) {
            printf("%f ", temp[j]);
        }
        printf("\n");
    }
}

void print_m256(__m256 vec) {
    float temp[8];  // Temporary array to store the 8 float values in the __m256 register
    _mm256_storeu_ps(temp, vec);  // Store the contents of the __m256 register into the array

    printf("Values in __m256 register: ");
    for (int i = 0; i < 8; i++) {
        printf("%f ", temp[i]);
    }
    printf("\n");
}

void print_mask(const __m256i mask) {
    int values[8];
    _mm256_storeu_si256((__m256i*)values, mask);

    printf("Mask values: ");
    for (int i = 0; i < 8; i++) {
        printf("%u ", values[i]);  // Prints in decimal as signed integers
    }
    printf("\n");
}

void pack_panelB(float* B, float* blockB_packed, const int nr, const int kc, const int K) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * K + p];
        }
        for (int j = nr; j < NR; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB(float* B, float* blockB_packed, const int nc, const int kc, const int K) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for (int j = 0; j < nc; j += NR) {
        const int nr = min(NR, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

void pack_panelA(float* A, float* blockA_packed, const int mr, const int kc, const int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < MR; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA(float* A, float* blockA_packed, const int mc, const int kc, const int M) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for (int i = 0; i < mc; i += MR) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M) {
    __m256 C_buffer[12]; // 6x2 becomes a linear array of size 12
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_masks[2];

    if (m != 16) {
        const unsigned int bit_mask = 65535;
        packed_masks[0] = _mm256_setr_epi32(
            bit_mask << (m + 15), bit_mask << (m + 14), bit_mask << (m + 13), bit_mask << (m + 12),
            bit_mask << (m + 11), bit_mask << (m + 10), bit_mask << (m + 9), bit_mask << (m + 8));
        packed_masks[1] = _mm256_setr_epi32(
            bit_mask << (m + 7), bit_mask << (m + 6), bit_mask << (m + 5), bit_mask << (m + 4),
            bit_mask << (m + 3), bit_mask << (m + 2), bit_mask << (m + 1), bit_mask << m);

        for (int j = 0; j < n; j++) {
            C_buffer[j * 2] = _mm256_maskload_ps(&C[j * M], packed_masks[0]);
            C_buffer[j * 2 + 1] = _mm256_maskload_ps(&C[j * M + 8], packed_masks[1]);
        }


    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[j * 2] = _mm256_loadu_ps(&C[j * M]);
            C_buffer[j * 2 + 1] = _mm256_loadu_ps(&C[j * M + 8]);
        }
    }

    for (int p = 0; p < k; p++) {
        a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        // Unrolled multiplication and accumulation for 6 elements in blockB_packed
        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0]);
        C_buffer[1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[2] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[2]);
        C_buffer[3] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[4] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[4]);
        C_buffer[5] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[5]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[6] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[6]);
        C_buffer[7] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[7]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[8] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[8]);
        C_buffer[9] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[9]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[10] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[10]);
        C_buffer[11] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[11]);

        blockA_packed += 16;
        blockB_packed += 6;
    }

    if (m != 16) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_masks[0], C_buffer[j * 2]);
            _mm256_maskstore_ps(&C[j * M + 8], packed_masks[1], C_buffer[j * 2 + 1]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[j * 2]);
            _mm256_storeu_ps(&C[j * M + 8], C_buffer[j * 2 + 1]);
        }
    }

}

void matmul_parallel(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        kernel_16x6(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                                    &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
                    }
                }
            }
        }
    }
}

void matmul_naive(float* A, float* B, float* C, const int M, const int N, const int K) {
  #pragma omp parallel for collapse(2) num_threads(NTHREADS)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int p = 0; p < K; p++) {
                C[j * M + i] += A[p * M + i] * B[j * K + p];
            }
        }
    }
}

void print_mat(float* mat, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", mat[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_rand(float* mat, const int M, const int N) {
    int total = M * N;
    for (int i = 0; i < M * N; i++) {
        // *mat++ = rand() / (float)RAND_MAX;
        mat[i] = (float)i / (total - 1);
    }
}

void init_const(float* mat, const float value, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            *mat++ = value;
        }
    }
}

void compare_mats(float* mat1, float* mat2, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(mat1[j * M + i] - mat2[j * M + i]) > 1e-4) {
                printf("MISMATCH! Element[%d][%d] %f != %f\n", i, j, mat1[j * M + i],
                       mat2[j * M + i]);
                return;
            }
        }
    }
    printf("MATCH!\n");
    return;
}

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
    const int M = MDIM;
    const int N = NDIM;
    const int K = KDIM;
    float* A = (float*)_mm_malloc(M * K * sizeof(float), MEM_ALIGN);
    float* B = (float*)_mm_malloc(K * N * sizeof(float), MEM_ALIGN);
    float* C = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_ref = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    init_rand(A, M, K);
    init_rand(B, K, N);

#ifdef TEST
    matmul_naive(A, B, C_ref, M, N, K);
#endif
    double FLOP = 2 * (double)M * N * K;
    double gflops_sum = 0.0, gflops_min = DBL_MAX, gflops_max = DBL_MIN;

    for (int i = 0; i < NITER; i++) {
        init_const(C, 0.0, M, N);
        uint64_t start = timer();
        matmul_parallel(A, B, C, M, N, K);
        uint64_t end = timer();

        double exec_time = (end - start) * 1e-9; // Execution time in seconds
        double FLOPS = FLOP / exec_time;        // Floating-point operations per second
        double GFLOPS = FLOPS / 1e9;            // GFLOPS

        // Accumulate GFLOPS for statistics
        gflops_sum += GFLOPS;
        if (GFLOPS < gflops_min) gflops_min = GFLOPS;
        if (GFLOPS > gflops_max) gflops_max = GFLOPS;

        printf("Exec. time = %.3f ms\n", exec_time * 1000);
        printf("GFLOPS = %.3f\n", GFLOPS);
#ifdef TEST
        compare_mats(C, C_ref, M, N);
#endif
        printf("\n");
    }

    // Compute average GFLOPS
    double gflops_avg = gflops_sum / NITER;

    // Print statistics
    printf("Average GFLOPS = %.3f\n", gflops_avg);
    printf("Max GFLOPS = %.3f\n", gflops_max);
    printf("Min GFLOPS = %.3f\n", gflops_min);

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0;
}
