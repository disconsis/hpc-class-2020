CC=g++

.PHONY: all
all: val_test01_solved

val_test01_solved: val_test01_solved.cpp
	$(CC) -o val_test01_solved val_test01_solved.cpp
