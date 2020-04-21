#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <omp.h>

#define MAX_ITER 10
#define BLOCK_SIZE 1024

inline double square(double x) { return x * x; }

inline void swap(double **x, double **y) {
  double* tmp;
  tmp = *x;
  *x = *y;
  *y = tmp;
}

// the final result is in `U_new`
void jacobi_ref(double *ref_old, double *ref_new, size_t N) {
  double h_sq = square(1.00 / (N + 1));

  for(size_t iter = 0; iter < MAX_ITER; iter++) {
    #pragma omp parallel for
    for (size_t i = 1; i <= N; i++) {
      for (size_t j = 1; j <= N; j++) {
        ref_new[i*(N + 2) + j] =
          (h_sq
           + ref_old[(i - 1)*(N + 2) + j    ]
           + ref_old[(i    )*(N + 2) + j - 1]
           + ref_old[(i + 1)*(N + 2) + j    ]
           + ref_old[(i    )*(N + 2) + j + 1])/4;
      }
    }

    swap(&ref_old, &ref_new);
  }
}


__global__
void jacobi(double *ref_old, double *ref_new, size_t N) {
  double h_sq = (1.00 / (N + 1)) * (1.00 / (N + 1));
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  for (size_t iter = 0; iter < MAX_ITER; iter++) {
    ref_new[y*(N + 2) + x] =
      (h_sq
       + ref_old[(y - 1)*(N + 2) + x    ]
       + ref_old[(y    )*(N + 2) + x - 1]
       + ref_old[(y + 1)*(N + 2) + x    ]
       + ref_old[(y    )*(N + 2) + x + 1])/4;

    double* tmp = ref_old;
    ref_old = ref_new;
    ref_new = tmp;

    __syncthreads();
  }
}


int main() {
  size_t N = (1UL << 15);
  size_t num_elems = (N + 2) * (N + 2);
  size_t size = num_elems * sizeof(double);

  // call ref
  double *U_old_ref = (double*) calloc(num_elems, sizeof(double));
  double *U_new_ref = (double*) calloc(num_elems, sizeof(double));

  double start_time_ref = omp_get_wtime();
  jacobi_ref(U_old_ref, U_new_ref, N);
  double total_time_ref = omp_get_wtime() - start_time_ref;

  printf("--\n");

  // call actual
  double *U_old, *U_new;

  cudaMalloc(&U_old, size);
  cudaMalloc(&U_new, size);

  double* U_answer = (double*) malloc(size);
  cudaMemset(U_old, 0, size);
  cudaMemset(U_new, 0, size);

  double start_time = omp_get_wtime();
  jacobi<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(U_old, U_new, N);
  cudaDeviceSynchronize();
  cudaMemcpy(U_answer, U_new, size, cudaMemcpyDeviceToHost);
  double total_time = omp_get_wtime() - start_time;

  // time
  printf("Time: %f -> ref: %f\n", total_time, total_time_ref / total_time);

  // error
  double err = 0;
  for (size_t i = 0; i < num_elems; i++) {
    // printf("%lu\n", i);
    err += fabs(U_answer[i] - U_new_ref[i]);
  }
  printf("Error: %f\n", err * 1e-9);

  // free
  cudaFree(U_old);
  cudaFree(U_new);
  free(U_answer);

  return 0;
}
