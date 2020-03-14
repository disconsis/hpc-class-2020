#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>
#include "utils.h"
using namespace std;

#define N 10000
#define MAX_ITER 10
#define THREADS 16

double u_arr[N + 2][N + 2] = {0};
double u_new_arr[N + 2][N + 2] = {0};

inline double square(double x) { return x * x; };
double h_sq = square(1.00 / (N + 1));

inline void gauss_update(double (*u_old)[N + 2],
                         double (*u_new)[N + 2],
                         long i_start,
                         long j_start) {
  #pragma omp parallel for num_threads(THREADS)
  for (auto i = i_start; i <= N; i += 2) {
    for (auto j = j_start; j <= N; j += 2) {
      u_new[i][j] = (h_sq
                     + u_old[i - 1][j]
                     + u_old[i][j - 1]
                     + u_old[i + 1][j]
                     + u_old[i][j + 1]) / 4;
    }
  }
}

void gauss() {
  double(*u_old)[N + 2] = u_arr;
  double(*u_new)[N + 2] = u_new_arr;


  for (auto iter = 0; iter < MAX_ITER; iter++) {
    // reds first
    // odd*odd and even*even
    gauss_update(u_old, u_new, 1, 1);
    gauss_update(u_old, u_new, 2, 2);

    // then black
    // even*odd and odd*even
    gauss_update(u_old, u_new, 2, 1);
    gauss_update(u_old, u_new, 1, 2);

    swap(u_old, u_new);
  }
}

int main() {
#ifdef _OPENMP
  cout << "OpenMP enabled\n";
  cout << "Threads: " << THREADS << "\n";
#else
  cout << "OpenMP disabled\n";
#endif
  cout << "Size: " << N << endl;

  Timer t;
  t.tic();
  gauss();
  double time = t.toc();

  cout << "Time: " << time << endl;

  return 0;
}
