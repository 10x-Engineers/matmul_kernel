// clang-17 -O2 -mno-avx512f -march=native -DTEST -DNITER=1000 matmul_kernel.c -o matmul_kernel.out && ./matmul_kernel.out
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEM_ALIGN 64

// #define MR 16
// #define NR 6

#define MR 8
#define NR 8

#ifndef MDIM
#define MDIM MR * 62
#endif

#ifndef NDIM
#define NDIM NR * 166
#endif

#ifndef KDIM
#define KDIM 1000
#endif

#ifndef NITER
#define NITER 1
#endif

void print_C_buffer(__m256 C_buffer[], int index) {
    float values[8];  // Temporary array to hold the values
    _mm256_storeu_ps(values, C_buffer[index]);  // Store C_buffer[index] into the array

    // Print each element
    printf("C_buffer[%d]: ", index);
    for (int i = 0; i < 8; i++) {
        printf("%f ", values[i]);
    }
    printf("\n");
}

void print_avx_register(__m256 reg) {
    float temp[8]; // Temporary array to store register contents
    _mm256_storeu_ps(temp, reg); // Store register into array
    for (int i = 0; i < 8; i++) {
        printf("%f ", temp[i]); // Print each element
    }
    printf("\n");
}

void kernel_8x8(float* blockA, float* blockB, float* C, const int M, const int K) {
    __m256 C_buffer[8];     // Buffer for storing intermediate results for each row of C
    __m256 b_packFloat8;    // Temporary register for broadcasting an element from B
    __m256 a_packFloat8;    // Temporary register for 8 elements from a row in A

    // Initialize C_buffer with the current values of C (for accumulation)
    for (int i = 0; i < 8; i++) {
        C_buffer[i] = _mm256_loadu_ps(&C[i * M]); // Load 8 floats from each row of C
    }

    // Perform matrix multiplication and accumulation
    for (int p = 0; p < K; p++) {
        // Load a row of A into a_packFloat8
        a_packFloat8 = _mm256_loadu_ps(&blockA[p * M]);

        // Manually unroll the inner loop for each column of C
        b_packFloat8 = _mm256_broadcast_ss(&blockB[0 * K + p]);
        C_buffer[0] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[0]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[1 * K + p]);
        C_buffer[1] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[1]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[2 * K + p]);
        C_buffer[2] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[2]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[3 * K + p]);
        C_buffer[3] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[3]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[4 * K + p]);
        C_buffer[4] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[4]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[5 * K + p]);
        C_buffer[5] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[5]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[6 * K + p]);
        C_buffer[6] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[6]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[7 * K + p]);
        C_buffer[7] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[7]);
    }

    // Store results back to C
    for (int i = 0; i < 8; i++) {
        _mm256_storeu_ps(&C[i * M], C_buffer[i]); // Store 8 floats back to each row of C
    }
}

void matmul_kernel(float* A, float* B, float* C, const int M, const int N, const int K) {
    assert(M % MR == 0);
    assert(N % NR == 0);
    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            kernel_8x8(&A[i], &B[j * K], &C[j * M + i], M, K);
        }
    }
}

void matmul_naive(float* A, float* B, float* C, const int M, const int N, const int K) {
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
    for (int i = 0; i < M * N; i++) {
        *mat++ = rand() / (float)RAND_MAX;
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
            if (fabsf(mat1[j * M + i] - mat2[j * M + i]) > 1e-3) {
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

    printf("FLOP:%f\n", FLOP);

    for (int i = 0; i < NITER; i++) {
        init_const(C, 0.0, M, N);
        uint64_t start = timer();
        matmul_kernel(A, B, C, M, N, K);
        uint64_t end = timer();

        double exec_time = (end - start) * 1e-9;
        double FLOPS = FLOP / exec_time;

        printf("Exec. time = %.3fms\n", exec_time * 1000);
        printf("GFLOPS = %.3f\n", FLOPS / 1e9);
#ifdef TEST
        compare_mats(C, C_ref, M, N);
#endif
        printf("\n");
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0;
}