# Compilation Instructions

## CPU Code

The source code for the CPU implementation is contained in the `nn.c` file. Macros at the top of the file can be changed/commented out to control thread count, tile size, batch size, and which implementation of matrix-matrix multiplication (hand-written vs. OpenBLAS) to use. Training parameters (number of epochs, learning rate) are set at the bottom of the file in the main function and can be edited there. The code can be compiled with `make nn` and run with `./nn`.

## GPU Code

The source code for the GPU implementation is contained in the `nn.cu` file. Similarly, macros at the top of the file control tile size, threads per block, block count, batch size, and which implementation of matrix-matrix multiplication is used. Training parameters are set in main again at the bottom of the file. Code can be compiled with `make nn_cu` and run with `srun ./nn_cu` (or, if using the UChicago Midway RCC Cluster, `sbatch batchfile_cu`). The Makefile and batchfile (`batchfile_cu`) are set to compile/run the CUDA code for an Nvidia Tesla V100 core. Targetting different hardware can be done by editting the Makefile/batchfile.