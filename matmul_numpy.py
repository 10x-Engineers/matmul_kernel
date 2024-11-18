import numpy as np
import time

# Matrix size
N = 200  # Change this as needed

# Generate random matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Warm-up to ensure fair timing
np.matmul(A, B)

# Number of iterations
iterations = 100
operations = 2 * (N ** 3)  # Total floating-point operations per multiplication

# List to store GFLOPS for each iteration
gflops_list = []

for _ in range(iterations):
    # Start timing
    start_time = time.perf_counter()
    # Perform matrix multiplication
    C = np.matmul(A, B)
    # End timing
    end_time = time.perf_counter()
    
    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time
    
    # Calculate GFLOPS
    GFlops = operations / (elapsed_time * 1e9)
    
    # Append the result to the list
    gflops_list.append(GFlops)
    
    # Print execution time and GFLOPS for this iteration
    print(f"Exec. time: {elapsed_time * 1000:.3f} ms")
    print(f"GFLOPS: {GFlops:.2f} GFlops")
    print()

# Calculate average, max, and min GFLOPS
average_gflops = sum(gflops_list) / iterations
max_gflops = max(gflops_list)
min_gflops = min(gflops_list)

print(f"Average GFLOPS: {average_gflops:.2f}")
print(f"Max GFLOPS: {max_gflops:.2f}")
print(f"Min GFLOPS: {min_gflops:.2f}")
