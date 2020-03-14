#include <iostream>
#include <cstdio>
#include <utility>
#include <cmath>
using namespace std;


#define N 16
#define MAX_ITER 5000
#define MIN_RESIDUAL_FRAC 1e6

inline double square(double x) { return x * x; };

double residual(double (*u)[N + 2]) {
  double res = 0;

  for (auto i = 1; i <= N; i++) {
    for (auto j = 1; j <= N; j++) {
      res += square(4 * u[i][j]
                    - u[i - 1][j    ]
                    - u[i    ][j - 1]
                    - u[i + 1][j    ]
                    - u[i    ][j + 1] - 1);
    }
  }

  return sqrt(res);
}

string to_string(double (*u)[N + 2]) {
  string repr = "";
  for (auto i = 1; i <= N; i++) {
    for (auto j = 1; j <= N; j++) {
      repr += to_string(u[i][j]) + " ";
    }
    repr += "\n";
  }
  return repr;
}

void jacobi(double u_arr[N + 2][N + 2]) {
  double u_new_arr[N + 2][N + 2] = { 0 };

  double (*u_old)[N + 2] = u_arr;
  double(*u_new)[N + 2] = u_new_arr;

  double h_sq = square(1.00 / (N + 1));

  double init_residual = residual(u_old);
  double threshold_residual = init_residual / MIN_RESIDUAL_FRAC;
  double last_residual = init_residual;

  printf("Initial residual: %f\n", init_residual);
  printf("Threshold residual: %f\n", threshold_residual);

  for(auto iter = 0; iter < MAX_ITER; iter++) {
    // update
    for (auto i = 1; i <= N; i++) {
      for (auto j = 1; j <= N; j++) {
        u_new[i][j] = (h_sq
                       + u_old[i - 1][j    ]
                       + u_old[i    ][j - 1]
                       + u_old[i + 1][j    ]
                       + u_old[i    ][j + 1])/4;
      }
    }

    // check end conditions
    double residual_new = residual(u_new);
    printf("Residual: %10.20f\n", residual_new);

    if (residual_new > last_residual) {
      printf("[!] Diverging. stopping.\n");
      return;
    } else if (residual_new <= threshold_residual) {
      printf("[*] Threshold residual reached\n");
      return;
    } else {
      last_residual = residual_new;
      swap(u_old, u_new);
    }
  }
}



int main() {
  double u[N + 2][N + 2] = { 0 };
  jacobi(u);

  return 0;
}
