from random import random_float64, seed
from time import now
from utils.numerics import inf
from python import Python, PythonObject
from builtin.math import abs
from sys import simdwidthof
from sys.intrinsics import masked_load, masked_store
from algorithm import parallelize
from memory.unsafe_pointer import UnsafePointer
import math
from algorithm import reduction
from memory import stack_allocation

alias NUM_ITER = 100

alias MR = 16
alias NR = 6
alias NTHREADS = 8

alias M = 741
alias N = 2048
alias KDIM = 741

alias MC = MR * NTHREADS * 1
alias NC = NR * NTHREADS * 80
alias KC = 1000

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
    # total = M * N
    for i in range(M*N):
        random_float64_value = random_float64()
        random_float32_value = random_float64_value.cast[DType.float32]()
        mat[i] = random_float32_value
        # mat[i] = (i / (total - 1)).cast[DType.float32]()

fn init_const(mat: UnsafePointer[Float32], m:Int, n:Int):
    for i in range(m*n):
        mat[i] = 0.0


fn compare_mat(mat1: UnsafePointer[Float32], mat2: UnsafePointer[Float32], M:Int, N:Int):
    for i in range(M):
        for j in range(N):
            if abs(mat1[j * M + i] - mat2[j * M + i]) > 1e-4:
                print("Mismatch:", i, j, mat1[j * M + i], mat2[j * M + i])
                return
    print("Match")

fn pack_panelB(B: UnsafePointer[Float32], blockB_packed: UnsafePointer[Float32], nr:Int, kc:Int, K:Int):
    for p in range(kc):
        for j in range(nr):
            blockB_packed[p * NR + j] = B[j * K + p]
        
        for j in range(nr, NR):
            blockB_packed[p * NR + j] = 0.0

fn pack_blockB(B: UnsafePointer[Float32], blockB_packed: UnsafePointer[Float32], nc: Int, kc: Int, K: Int):
    @parameter
    fn pack_row1(thread_id: Int):
        j = thread_id * NR
        nr = min(NR, nc - j)
        pack_panelB(B + (j * K), blockB_packed + (j * kc), nr, kc, K)
    
    iterations = (nc + NR - 1) // NR
    parallelize[pack_row1] (iterations, NTHREADS)


fn pack_panelA(A: UnsafePointer[Float32], blockA_packed: UnsafePointer[Float32], mr:Int, kc:Int, M:Int):
    for p in range(kc):
        for i in range(mr):
            blockA_packed[p * MR + i] = A[p * M + i]
        
        for j in range(mr, MR):
            blockA_packed[p * MR + j] = 0.0

fn pack_blockA(A: UnsafePointer[Float32], blockA_packed: UnsafePointer[Float32], mc:Int, kc:Int, K:Int):
    @parameter
    fn pack_row2(thread_id: Int):
        i = thread_id * MR
        mr = min(MR, mc-i)
        pack_panelA(A + (i), blockA_packed + (i*kc), mr, kc, K)
    
    iterations = (mc + MR - 1) // MR
    parallelize[pack_row2] (iterations, NTHREADS)

fn mask_values(inp: UInt32) -> UInt32:
    g = bin(inp)
    if len(g) < 34:
        return 0
    else:
        bit_str = g[2:3]
        if bit_str == "1":
            return 1
        else:
            return 0

fn kernel_16x6(blockA_packed: UnsafePointer[Float32], blockB_packed: UnsafePointer[Float32], C: UnsafePointer[Float32], 
               m:Int, n:Int, k:Int, M:Int, thread_id:Int):
    
    mask = stack_allocation[16, UInt32] ()
    C_buffer = stack_allocation[96, Float32] ()
    
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
            C_buffer.store(j * 2 * 8, masked_load[nelts](C + j * M,
                                                                      mask.load[width = nelts](0) != 0, 0))
            C_buffer.store((j * 2 + 1) * 8, masked_load[nelts](C + j * M + 8,
                                                                          mask.load[width = nelts](8) != 0, 0))
        
    else:
        for j in range(n):
            C_buffer.store(j*2*8, C.load[width=nelts] (j * M))
            C_buffer.store((j * 2 + 1) * 8, C.load[width = nelts](j * M + 8))

    a0 = 0
    a1 = 0
    for _ in range(k):
        a0_packFloat8 = blockA_packed.load[width = nelts] (a0)
        a1_packFloat8 = blockA_packed.load[width = nelts] (a0+8)
        
        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1])
        C_buffer.store(0, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](0)))
        C_buffer.store(8, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](8)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+1])
        C_buffer.store(16, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](16)))
        C_buffer.store(24, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](24)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+2])
        C_buffer.store(32, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](32)))
        C_buffer.store(40, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](40)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+3])
        C_buffer.store(48, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](48)))
        C_buffer.store(56, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](56)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+4])
        C_buffer.store(64, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](64)))
        C_buffer.store(72, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](72)))

        b_packFloat8 = SIMD[DType.float32, 1](blockB_packed[a1+5])
        C_buffer.store(80, a0_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](80)))
        C_buffer.store(88, a1_packFloat8.fma(b_packFloat8, C_buffer.load[width=nelts](88)))
        
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
            C.store(j * M, C_buffer.load[width=nelts](8 * 2 * j))
            C.store(j * M + 8, C_buffer.load[width=nelts](8 * (2 * j + 1)))


fn matmul_parallel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M:Int, N:Int, K:Int):
    blockA_packed = UnsafePointer[Float32, alignment=64].alloc(MC * KC)
    blockB_packed = UnsafePointer[Float32, alignment=64].alloc(NC * KC)
    for j in range(0,N,NC):
        nc = min(NC, N - j)
        for p in range(0,K,KC):
            kc = min(KC, K - p)
            pack_blockB(B+ (j * K + p), blockB_packed, nc, kc, K)
            for i in range(0, M, MC):
                mc = min(MC, M - i)
                pack_blockA(A+ (p * M + i), blockA_packed, mc, kc, M)

                @parameter
                fn pack(thread_id:Int):
                    jr = thread_id * NR
                    nr = min(NR, nc - jr)
                    for ir in range(0, mc, MR):
                        mr = min(MR, mc - ir)
                        kernel_16x6(blockA_packed +(ir * kc), blockB_packed + (jr * kc), C + ((j + jr) * M + (i + ir)), 
                                   mr, nr, kc, M, thread_id)

                iterations = (nc + NR - 1) // NR
                parallelize[pack] (iterations, NTHREADS)
    
    blockA_packed.free()
    blockB_packed.free()

fn main() raises:
    seed()
    FLOP = 2 * M * N * KDIM
    A = UnsafePointer[Float32, alignment=64].alloc(M * KDIM)
    B = UnsafePointer[Float32, alignment=64].alloc(KDIM * N)
    C_ref = UnsafePointer[Float32, alignment=64].alloc(M * N)
    C = UnsafePointer[Float32, alignment=64].alloc(M * N)

    init_const(C_ref,M,N)
    init_rand(A, M, KDIM)
    init_rand(B, KDIM, N)

    print("Matmul naive")
    matmul_naive(A, B, C_ref, M, N, KDIM)
    print("FLOP:", FLOP)

    var gflops_list = List[Float64] ()
    
    print("Benchmark started")
    for _ in range(NUM_ITER):
        init_const(C,M,N)
        start = now()
        matmul_parallel(A, B, C, M, N, KDIM)
        end = now()
        
        exec_time = (end - start) * 1e-9

        FLOPS = FLOP / exec_time

        print("Exec. time (ms): ", exec_time * 1000)
        print("GFLOPS: ", FLOPS / 1e9)

        compare_mat(C, C_ref, M, N)

        gflops_list.append(FLOPS / 1e9)

    sum = 0.0
    max = 0.0
    min = 1000000000000000.0
    for i in range(len(gflops_list)):
        sum+= gflops_list[i]
        if gflops_list[i] > max:
            max = gflops_list[i]
        if gflops_list[i] < min:
            min = gflops_list[i]

    average_gflops = sum / NUM_ITER
    max_gflops = max
    min_gflops = min

    print("Average GFLOPS:", average_gflops)
    print("Max GFLOPS:" ,max_gflops)
    print("Min GFLOPS:", min_gflops)

    A.free()
    B.free()
    C_ref.free()
    C.free()

    # print(M,N,KDIM)
