#include <iostream>
#include <cstdio>
#include <utility>
#include <cmath>
#include "utils.h"
#include <omp.h>
using namespace std;


#define N 10000
#define MAX_ITER 10

double u_arr[N + 2][N + 2] = {0};
double u_new_arr[N + 2][N + 2] = {0};

inline double square(double x) { return x * x; };

void jacobi() {
  double (*u_old)[N + 2] = u_arr;
  double(*u_new)[N + 2] = u_new_arr;

  double h_sq = square(1.00 / (N + 1));

  for(auto iter = 0; iter < MAX_ITER; iter++) {
    #pragma omp parallel for
    for (auto i = 1; i <= N; i++) {
      for (auto j = 1; j <= N; j++) {
        u_new[i][j] = (h_sq
                       + u_old[i - 1][j    ]
                       + u_old[i    ][j - 1]
                       + u_old[i + 1][j    ]
                       + u_old[i    ][j + 1])/4;
      }
    }

    swap(u_old, u_new);
  }
}



int main() {

  Timer t;
  t.tic();
  jacobi();
  double time = t.toc();

  #ifdef _OPENMP
  cout << N << " " << omp_get_max_threads() << " " << time << "\n";
  #endif

  return 0;
}
