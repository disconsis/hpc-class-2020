#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NTHREADS 4

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}


void scan_omp(long* prefix_sum, const long* A, long n) {
  #pragma omp parallel num_threads(NTHREADS)
  {
      int tidx = omp_get_thread_num();

      int start = (n * tidx)/NTHREADS;
      int end = std::min(n, (n * (tidx + 1))/NTHREADS);
      prefix_sum[start] = 0;

      for (int i = start + 1; i < end; i++) {
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      }
  }

  for (int t = 1; t < NTHREADS; t++) {
      int start = (n * t)/NTHREADS;
      int end = std::min(n, (n * (t + 1))/NTHREADS);

      long to_add = prefix_sum[start - 1] + A[start - 1];
      for (int i = start; i < end; i++) {
          prefix_sum[i] += to_add;
      }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
