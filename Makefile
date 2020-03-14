CC=g++ -ggdb

.PHONY: all
all: val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 MMult1 jacobi2D-omp gs2D-omp

val_test01_solved: val_test01_solved.cpp
	$(CC) -o val_test01_solved val_test01_solved.cpp

val_test02_solved: val_test02_solved.cpp
	$(CC) -o val_test02_solved val_test02_solved.cpp

omp_solved2: omp_solved2.c
	$(CC) -o omp_solved2 -fopenmp omp_solved2.c

omp_solved3: omp_solved3.c
	$(CC) -o omp_solved3 -fopenmp omp_solved3.c

omp_solved4: omp_solved4.c
	$(CC) -o omp_solved4 -fopenmp omp_solved4.c

omp_solved5: omp_solved5.c
	$(CC) -o omp_solved5 -fopenmp omp_solved5.c

omp_solved6: omp_solved6.c
	$(CC) -o omp_solved6 -fopenmp omp_solved6.c

MMult1: MMult1.cpp
	g++ -fopenmp -O3 -march=native MMult1.cpp -o MMult1

jacobi2D-omp: jacobi2D-omp.cpp
	g++ -o jacobi2D-omp -ggdb -fopenmp -march=native jacobi2D-omp.cpp # -O2

gs2D-omp: gs2D-omp.cpp
	g++ -o gs2D-omp -ggdb -fopenmp -march=native gs2D-omp.cpp # -O2
