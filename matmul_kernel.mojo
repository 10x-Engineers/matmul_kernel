from random import random_float64, seed
from time import now
from utils.numerics import inf
from python import Python, PythonObject
from builtin.math import abs
from sys import simdwidthof
from sys.intrinsics import masked_load, masked_store


alias NUM_ITER = 100


alias MR = 16
alias NR = 6

alias M = 1013
alias N = 1022
alias KDIM = 1011

alias nelts = simdwidthof[DType.float32] ()

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

fn print_C_Buffer(C_buffer: UnsafePointer[Float32]):
    for i in range(12):
        print("C_buffer["+str(i)+"]:", C_buffer.load[width = nelts](nelts*i))


fn matmul_naive(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M:Int, N:Int, K:Int):
    for i in range(M):
        for j in range(N):
            for p in range(K):
                C[j * M + i] += A[p * M + i] * B[j * K + p]

fn init_rand(mat: UnsafePointer[Float32], M:Int, N:Int):
    total = M * N
    for i in range(M*N):
        random_float64_value = random_float64()
        random_float32_value = random_float64_value.cast[DType.float32]()
        mat[i] = random_float32_value
        mat[i] = (i / (total - 1)).cast[DType.float32]()

fn compare_mat(mat1: UnsafePointer[Float32], mat2: UnsafePointer[Float32], M:Int, N:Int):
    for i in range(M):
        for j in range(N):
            if abs(mat1[j * M + i] - mat2[j * M + i]) > 1e-1:
                print("Mismatch")
                return
    print("Match")


fn pack_blockA(A:UnsafePointer[Float32], blockA_packed:UnsafePointer[Float32], m:Int, M:Int, K:Int):
    for p in range(K):
        for i in range(m):
            blockA_packed[p * MR + i] = A[p * M + i]
        
        for i in range(m, MR):
            blockA_packed[p * MR + i] = 0.0

fn pack_blockB(B: UnsafePointer[Float32], blockB_packed: UnsafePointer[Float32], n: Int, N: Int, K: Int):
    for p in range(K):
        for j in range(n):
            blockB_packed[p * NR + j] = B[j * K + p]
        
        for j in range(n, NR):
            blockB_packed[p * NR + j] = 0.0

fn mask_values(inp: UInt32) raises -> UInt32:
    g = bin(inp)
    if len(g) < 34:
        return 0
    else:
        return int((g)[2:3])

fn kernel_16x6(blockA_packed: UnsafePointer[Float32], blockB_packed: UnsafePointer[Float32], C: UnsafePointer[Float32], 
               m:Int, n:Int, M:Int, N:Int, K:Int, count:Int) raises:
    
    mask = UnsafePointer[UInt32].alloc(16)
    C_buffer = UnsafePointer[Float32].alloc(96)

    var a0_packFloat8: SIMD[DType.float32, nelts]
    var a1_packFloat8: SIMD[DType.float32, nelts]

    if m !=16:
        var bit_mask:UInt32 = 65535
        mask[0] = mask_values(bit_mask << (m + 15))
        mask[1] = mask_values(bit_mask << (m + 14))
        mask[2] = mask_values(bit_mask << (m + 14))
        mask[3] = mask_values(bit_mask << (m + 12))
        mask[4] = mask_values(bit_mask << (m + 11))
        mask[5] = mask_values(bit_mask << (m + 10))
        mask[6] = mask_values(bit_mask << (m + 9))
        mask[7] = mask_values(bit_mask << (m + 8))
        mask[8] = mask_values(bit_mask << (m + 7))
        mask[9] = mask_values(bit_mask << (m + 6))
        mask[10] = mask_values(bit_mask << (m + 5))
        mask[11] = mask_values(bit_mask << (m + 4))
        mask[12] = mask_values(bit_mask << (m + 3))
        mask[13] = mask_values(bit_mask << (m + 2))
        mask[14] = mask_values(bit_mask << (m + 1))
        mask[15] = mask_values(bit_mask << (m))


        for j in range(n):
            C_buffer.store[width=nelts](j * 2 * 8, masked_load[nelts](C + j * M, mask.load[width = nelts](0) != 0, 0))
            C_buffer.store[width=nelts]((j * 2 + 1) * 8, masked_load[nelts](C + j * M + 8, mask.load[width = nelts](8) != 0, 0))
    else:
        for j in range(n):
            C_buffer.store[width = nelts] (j*2*8, C.load[width=nelts] (j * M))
            C_buffer.store[width = nelts]((j * 2 + 1) * 8, C.load[width = nelts](j * M + 8))

    a0 = 0
    a1 = 0
    for _ in range(K):
        a0_packFloat8 = blockA_packed.load[width = nelts] (a0)
        a1_packFloat8 = blockA_packed.load[width = nelts] (a0+8)
        
        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1])
        C_buffer.store[width=nelts](0, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](0)))
        C_buffer.store[width=nelts](8, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](8)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+1])
        C_buffer.store[width=nelts](16, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](16)))
        C_buffer.store[width=nelts](24, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](24)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+2])
        C_buffer.store[width=nelts](32, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](32)))
        C_buffer.store[width=nelts](40, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](40)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+3])
        C_buffer.store[width=nelts](48, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](48)))
        C_buffer.store[width=nelts](56, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](56)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+4])
        C_buffer.store[width=nelts](64, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](64)))
        C_buffer.store[width=nelts](72, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](72)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+5])
        C_buffer.store[width=nelts](80, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](80)))
        C_buffer.store[width=nelts](88, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](88)))
        
        a0 += 16
        a1 += 6
      
    if m!=16:
        for j in range(n):
            mask1 = SIMD[DType.bool, 8] ()
            mask2 = SIMD[DType.bool, 8] ()
            new_c = 0
            for x in range(8):
                if mask[x] == 1:
                    mask1[x] = True
                else:
                    mask1[x] = False
            for x in range(8,16):
                if mask[x] == 1:
                    mask2[new_c] = True
                else:
                    mask2[new_c] = False
                new_c+=1
            
            masked_store[8](C_buffer.load[width=8](8 * 2 * j), C + j*M, mask1)
            masked_store[8](C_buffer.load[width=8](8 * (2 * j + 1)), C + j*M+8, mask2)

    else:
        for j in range(n):
            C.store[width=8](j * M, C_buffer.load[width=8](8 * 2 * j))
            C.store[width=8](j * M + 8, C_buffer.load[width=8](8 * (2 * j + 1)))

   
fn matmul_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M:Int, N:Int, K:Int) raises:
    blockA_packed = UnsafePointer[Float32, alignment=64].alloc(MR * KDIM)
    blockB_packed = UnsafePointer[Float32, alignment=64].alloc(NR * KDIM)
    for i in range(0,M,MR):
        m = min(MR, M - i)
        pack_blockA(A+i, blockA_packed, m, M, KDIM)
        count = 0
        for j in range(0,N,NR):
            n = min(NR, N - j)
            pack_blockB(B+ (j * KDIM), blockB_packed, n, N, KDIM)
            kernel_16x6(blockA_packed, blockB_packed, C + (j*M + i), m, n, M, N, K, count)
            count +=1

fn main() raises:
    print(nelts)
    seed()
    
    print("BENCHMARK STARTED")

    FLOP = 2 * M * N * KDIM
    A = UnsafePointer[Float32, alignment=64].alloc(M * KDIM)
    B = UnsafePointer[Float32, alignment=64].alloc(KDIM * N)
    C_ref = UnsafePointer[Float32, alignment=64].alloc(M * N)

    init_rand(A, M, KDIM)
    init_rand(B, KDIM, N)

    blockA_packed = UnsafePointer[Float32, alignment=64].alloc(MR * KDIM)
    blockB_packed = UnsafePointer[Float32, alignment=64].alloc(NR * KDIM)

    C = UnsafePointer[Float32, alignment=64].alloc(M * N)
    for i in range(0,M,MR):
        m = min(MR, M - i)
        pack_blockA(A+i, blockA_packed, m, M, KDIM)
        count = 0
        for j in range(0,N,NR):
            n = min(NR, N - j)
            pack_blockB(B+ (j * KDIM), blockB_packed, n, N, KDIM)
            kernel_16x6(blockA_packed, blockB_packed, C + (j*M + i), m, n, M, N, KDIM, count)
            
            count +=1


    matmul_naive(A, B, C_ref, M, N, KDIM)
    pretty_print(C_ref, M, N)

    print("FLOP:", FLOP)

    for _ in range(NUM_ITER):
        C = UnsafePointer[Float32, alignment=64].alloc(M * N)
        start = now()
        matmul_kernel(A, B, C, M, N, KDIM)
        end = now()
        
        exec_time = (end - start) * 1e-9

        FLOPS = FLOP / exec_time

        print("Exec. time (ms): ", exec_time * 1000)
        print("GFLOPS: ", FLOPS / 1e9)

        compare_mat(C, C_ref, M, N)
