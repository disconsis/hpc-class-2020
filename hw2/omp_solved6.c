/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

/** Solution
 *
 * There were a couple of bugs in this.
 * First of all, it didn't compile since dotprod didn't return anything.
 * So I took sum as an in-variable.
 *
 * Secondly, dotprod was called from a parallel omp section, thus each
 * thread executed it independently, which is probably not the intended behaviour.
 * So I moved the parallel section inside the function.
 *
 * I assume the question had something to do with sum being a shared variable,
 * but since the code didn't compile as-is I didn't really know what was expected
 * in that scenario.
 */

float a[VECLEN], b[VECLEN];

void dotprod (float& sum)
{
int i,tid;

#pragma omp parallel private(tid)
{
  tid = omp_get_thread_num();

  #pragma omp for reduction(+ : sum)
  for (i=0; i<VECLEN; i++) {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
  }
}
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

dotprod(sum);

printf("Sum = %f\n",sum);

}

