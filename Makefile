CC=g++ -ggdb

.PHONY: all
all: val_test01_solved val_test02_solved omp_solved2

val_test01_solved: val_test01_solved.cpp
	$(CC) -o val_test01_solved val_test01_solved.cpp

val_test02_solved: val_test02_solved.cpp
	$(CC) -o val_test02_solved val_test02_solved.cpp

omp_solved2: omp_solved2.c
	$(CC) -o omp_solved2 -fopenmp omp_solved2.c
