#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <algorithm>


#define BLOCK_SIZE 1024
#define BLOCK_SIZE_RT 32

// C = A * B
__global__
void inner_product(double* C, const double* A, const double* B, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] += A[idx] * B[idx];
    }
}

void test_inner_product() {
    size_t N = (1UL << 20);
    double *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(double));
    cudaMallocManaged(&B, N * sizeof(double));
    cudaMallocManaged(&C, N * sizeof(double));
    double* C_ref = (double*) malloc(N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        A[i] = ((double) rand()) / 2;
        B[i] = ((double) rand()) / 3;
        C[i] = 0;
        C_ref[i] = A[i] * B[i];
    }

    double start_time = omp_get_wtime();
    inner_product<<<(N/BLOCK_SIZE) + 1,BLOCK_SIZE>>>(C,A,B,N);
    cudaDeviceSynchronize();
    double end_time = omp_get_wtime();

    printf("Time: %f\n", end_time - start_time);

    double err = 0;
    for (size_t i = 0; i < N; i++)
        err += fabs(C[i]-C_ref[i]);
    printf("Error = %f\n", err);
}

__global__
void matrix_product(double *C, double *A, double *B, size_t N) {
    size_t id_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (id_x >= N || id_y >= N) return;

    double sum = 0;
    for (size_t k = 0; k < N; k++) {
        sum += A[id_y * N + k] * B[k * N + id_x];
    }
    C[id_y * N + id_x] = sum;
}

void matrix_product_ref(double *C, double *A, double *B, size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

void test_matrix_product() {
    size_t N = (1UL << 10);
    size_t size = N * N * sizeof(double);
    double* A = (double*) malloc(size);
    double* B = (double*) malloc(size);
    double* C = (double*) malloc(size);
    double* C_ref = (double*) malloc(size);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = i * 0.6 + j;
            B[i * N + j] = ((double)i / (j + 1)) + j * 0.7;
            C[i*N + j] = 0;
            C_ref[i*N + j] = 0;
        }
    }

    double start_time_ref = omp_get_wtime();
    matrix_product_ref(C_ref, A, B, N);
    double end_time_ref = omp_get_wtime();
    printf("--\n");


    double *Ad, *Bd, *Cd;
    cudaMalloc(&Ad, size);
    cudaMalloc(&Bd, size);
    cudaMalloc(&Cd, size);

    dim3 gridSize(N/BLOCK_SIZE_RT, N/BLOCK_SIZE_RT);
    dim3 blockSize(BLOCK_SIZE_RT, BLOCK_SIZE_RT);

    double start_time = omp_get_wtime();
    cudaMemcpyAsync(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(Bd, B, size, cudaMemcpyHostToDevice);
    matrix_product<<<gridSize, blockSize>>>(Cd,Ad,Bd,N);
    cudaMemcpyAsync(C, Cd, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double end_time = omp_get_wtime();


    double gpu_time = end_time - start_time;
    double ref_time = end_time_ref - start_time_ref;
    printf("Time: %f = Time_ref/%f\n", gpu_time, (ref_time / gpu_time));
    printf("GPU bandwidth = %f GB/s\n", (3.0 * size) / gpu_time / 1e9);


    double err = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            err += fabs(C[i * N + j]-C_ref[i * N + j]);
    printf("Error = %f\n", err);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(C_ref);
}



int main() {

    test_matrix_product();

    return 0;
}

