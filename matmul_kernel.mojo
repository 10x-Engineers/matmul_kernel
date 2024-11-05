from random import random_float64, seed
from time import now
from utils.numerics import inf
from python import Python, PythonObject
from builtin.math import abs
from sys import simdwidthof

alias NUM_ITER = 100

alias M = 8*62
alias N = 8*166
alias K = 1000
# alias nelts = simdwidthof[DType.float32] ()
alias nelts = 8

fn pretty_print(mat: UnsafePointer[Float32], rows: Int, cols: Int) raises:
    def print_row(row_index: Int):
        for j in range(3):
            print(mat[row_index * cols + j], end=", ")
        print("...", end=", ")
        for j in range(cols - 3, cols):
            print(mat[row_index * cols + j], end=", ")
        print()

    for i in range(3):
        print_row(i)

    if rows > 6:
        print("...")

    for i in range(rows - 3, rows):
        print_row(i)

fn print_list[T: DType] (x: List[SIMD[T, 1]]):
    for i in range(len(x)):
        print(x[i], end=" ")
    print()

fn matmul_naive(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M:Int, N:Int, K:Int):
    for i in range(M):
        for j in range(N):
            for p in range(K):
                C[j * M + i] += A[p * M + i] * B[j * K + p]

fn init_rand(mat: UnsafePointer[Float32], M:Int, N:Int):
    for i in range(M*N):
        random_float64_value = random_float64()
        random_float32_value = random_float64_value.cast[DType.float32]()
        mat[i] = random_float32_value

fn kernel_8x8(blockA: UnsafePointer[Float32], blockB: UnsafePointer[Float32], C: UnsafePointer[Float32], M:Int, K:Int):
    C_buffer = List[SIMD[DType.float32, nelts]] (0,0,0,0,0,0,0,0)
    var b_packFloat8: SIMD[DType.float32, nelts]
    var a_packFloat8: SIMD[DType.float32, nelts]
    for i in range(8):
        C_buffer.append(C.load[width=nelts] (i * M))

    for p in range(K):
        a_packFloat8 = blockA.load[width = nelts] (p * M)
        
        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](0 * K + p))
        C_buffer[0] = a_packFloat8.fma(b_packFloat8, C_buffer[0])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](1 * K + p))
        C_buffer[1] = a_packFloat8.fma(b_packFloat8, C_buffer[1])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](2 * K + p))
        C_buffer[2] = a_packFloat8.fma(b_packFloat8, C_buffer[2])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](3 * K + p))
        C_buffer[3] = a_packFloat8.fma(b_packFloat8, C_buffer[3])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](4 * K + p))
        C_buffer[4] = a_packFloat8.fma(b_packFloat8, C_buffer[4])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](5 * K + p))
        C_buffer[5] = a_packFloat8.fma(b_packFloat8, C_buffer[5])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](6 * K + p))
        C_buffer[6] = a_packFloat8.fma(b_packFloat8, C_buffer[6])

        b_packFloat8 = SIMD[DType.float32, nelts] (blockB.load[width = 1](7 * K + p))
        C_buffer[7] = a_packFloat8.fma(b_packFloat8, C_buffer[7])

    for i in range(8):
        C.store[width = nelts](i * M, C_buffer[i])

fn matmul_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M:Int, N:Int, K:Int):
    for i in range(0,M,8):
        for j in range(0,N,8):
            kernel_8x8(A+i, B +(j*K), C + (j*M + i), M, K)
    
fn compare_mat(mat1: UnsafePointer[Float32], mat2: UnsafePointer[Float32], M:Int, N:Int):
    for i in range(M):
        for j in range(N):
            if abs(mat1[j * M + i] - mat2[j * M + i]) > 1e-3:
                print("Mismatch")
    print("Match")

fn main() raises:
    print(nelts)
    seed()
    
    print("BENCHMARK STARTED")

    FLOP = 2 * M * N * K
    A = UnsafePointer[Float32].alloc(M * K)
    B = UnsafePointer[Float32].alloc(K * N)
    C = UnsafePointer[Float32].alloc(M * N)
    C_ref = UnsafePointer[Float32].alloc(M * N)

    init_rand(A, M, K)
    init_rand(B, K, N)

    matmul_naive(A, B, C_ref, M, N, K)
    pretty_print(C_ref, M, N)

    print("FLOP:", FLOP)

    for _ in range(NUM_ITER):
        start = now()
        matmul_kernel(A, B, C, M, N, K)
        end = now()
        
        exec_time = (end - start) * 1e-9

        FLOPS = FLOP / exec_time

        print("Exec. time (ms): ", exec_time * 1000)
        print("GFLOPS: ", FLOPS / 1e9)

        # print(C[0])
        # pretty_print(C, M, N)

        compare_mat(C, C_ref, M, N)


