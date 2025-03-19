CC=gcc
CFLAGS=-I. -lm -O3 -fopenmp -lopenblas

nn: nn.c
	$(CC) -o nn nn.c $(CFLAGS)

nn_cu: nn.cu
	nvcc -arch=compute_70 -o nn_cu nn.cu -I. -lm -O3 -lcublas

clean:
	rm nn nn_cu