#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include "mnist.h"
#include "cuda.h"
#include "gputimer.h"

#define CUBLAS
#include <cublas_v2.h>
// Comment out if not using CUBLAS ^

#define TILE_SIZE 1
#define BATCH_SIZE 500
#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK 256

float loss[50] = {0};
int lossIdx = 0; 

/* =============================================================== */
/* CUDA stuff --- for doing matrix-matrix multiplication on device */
/* =============================================================== */

// Stores A x B + C in D. Ignores C if NULL. Assumes A is IxJ, B is JxK, C and D are IxK.
__global__ void mmult(int I, int J, int K, float *A, float *B, float *C, float *D) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread does 1 tile --- don't have to worry about shared memory
    // Write a __host__ setup function that does all the malloc-ing
    int num_blocks_width = ceil((float)K / TILE_SIZE);
    int num_blocks_height = ceil((float)I / TILE_SIZE);
    if (id >= num_blocks_height * num_blocks_width) {
        // One thread per tile --- extra threads return right away.
        return;
    }

    // Thread is responsible for writing to D[ilb:iub][klb:kub]
    int ilb = (id / num_blocks_width) * TILE_SIZE;
    int iub = min(ilb + TILE_SIZE, I);
    int klb = (id % num_blocks_width) * TILE_SIZE;
    int kub = min(klb + TILE_SIZE, K);

    // printf("Thread %d is responsible for D[%d:%d][%d:%d]\n", id, ilb, iub, klb, kub);
    
    // Load block of C into D
    for (int i = ilb; i < iub; i++) {
        for (int k = klb; k < kub; k++) {
            if (C != NULL) {
                D[i*K + k] = C[i*K + k];
            } else {
                D[i*K + k] = 0;
            }
        }
    }
    // Tiled multiplication of A and B
    int num_Jblocks = ceil((float)J / TILE_SIZE);
    for (int jblock = 0; jblock < num_Jblocks; jblock++) {
        for (int j = jblock * TILE_SIZE; j < min((jblock+1) * TILE_SIZE, J); j++) {
            for (int i = ilb; i < iub; i++) {
                for (int k = klb; k < kub; k++) {
                    D[i*K + k] += A[i*J + j] * B[j*K + k];
                }
            }
        }
    }
}

// Same as mmult but stores the transpose of the result
__global__ void mmultT(int I, int J, int K, float *A, float *B, float *C, float *D) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread does 1 tile --- don't have to worry about shared memory
    // Write a __host__ setup function that does all the malloc-ing
    int num_blocks_width = ceil((float)K / TILE_SIZE);
    int num_blocks_height = ceil((float)I / TILE_SIZE);
    if (id >= num_blocks_height * num_blocks_width) {
        // One thread per tile --- extra threads return right away.
        return;
    }

    // Thread is responsible for writing to D[ilb:iub][klb:kub]
    int ilb = (id / num_blocks_width) * TILE_SIZE;
    int iub = min(ilb + TILE_SIZE, I);
    int klb = (id % num_blocks_width) * TILE_SIZE;
    int kub = min(klb + TILE_SIZE, K);
    
    // Load block of C into D
    for (int i = ilb; i < iub; i++) {
        for (int k = klb; k < kub; k++) {
            if (C != NULL) {
                D[k*I + i] = C[i*I + k];
            } else {
                D[k*I + i] = 0;
            }
        }
    }
    // Tiled multiplication of A and B
    int num_Jblocks = ceil((float)J / TILE_SIZE);
    for (int jblock = 0; jblock < num_Jblocks; jblock++) {
        for (int j = jblock * TILE_SIZE; j < min((jblock+1) * TILE_SIZE, J); j++) {
            for (int i = ilb; i < iub; i++) {
                for (int k = klb; k < kub; k++) {
                    D[k*I + i] += A[i*J + j] * B[j*K + k];
                }
            }
        }
    }
}

#ifndef CUBLAS

// Does all the CudaMalloc-ing and calls the kernel
__host__ void mmult_setup(int I, int J, int K, float *A, float *B, float *C, float *D, bool transpose_res) {
    // printf("mmult_setup called with params %d %d %d\n", I, J, K);

    float *Adev, *Bdev, *Cdev, *Ddev;
    cudaMalloc((void **)&Adev, I*J*sizeof(float));
    cudaMalloc((void **)&Bdev, J*K*sizeof(float));
    if (C == NULL) {
        Cdev = NULL;
    } else {
        cudaMalloc((void **)&Cdev, I*K*sizeof(float));
    }
    cudaMalloc((void **)&Ddev, I*K*sizeof(float));
    
    // Write data to host (this may need to be optimized away haha)
    cudaMemcpy(Adev, A, I*J*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdev, B, J*K*sizeof(float), cudaMemcpyHostToDevice);
    if (Cdev != NULL) {
        cudaMemcpy(Cdev, C, I*K*sizeof(float), cudaMemcpyHostToDevice);
    }

    // Call device function
    if (transpose_res) {
        // printf("Calling mmultT on device...\n");
        mmultT<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(I, J, K, Adev, Bdev, Cdev, Ddev);
    } else {
        // printf("Calling mmult on device...\n");
        mmult<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(I, J, K, Adev, Bdev, Cdev, Ddev);
    }

    // Aggregate result back on device
    cudaMemcpy(D, Ddev, I*K*sizeof(float), cudaMemcpyDeviceToHost);

    // Free stuff
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
    cudaFree(Ddev);
}

#else

__host__ void mmult_setup(int I, int J, int K, float *A, float *B, float *C, float *D, bool transpose_res) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // printf("Cudamalloc-ing...\n");
    float *Adev, *Bdev, *Cdev;
    cudaMalloc((void **)&Adev, I*J*sizeof(float));
    cudaMalloc((void **)&Bdev, J*K*sizeof(float));
    cudaMalloc((void **)&Cdev, I*K*sizeof(float));

    // printf("CudaMemcpy-ing...\n");
    cudaMemcpy(Adev, A, I*J*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdev, B, J*K*sizeof(float), cudaMemcpyHostToDevice);
    if (C == NULL) {
        cudaMemset(Cdev, 0, I*K*sizeof(float));
    } else {
        cudaMemcpy(Cdev, C, I*K*sizeof(float), cudaMemcpyHostToDevice);
    }

    float alpha = 1.0;
    float beta = 1.0;

    if (transpose_res) {
        // printf("Calling Sgemm transpose...\n");
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, I, K, J, &alpha, Adev, J, Bdev, K, &beta, Cdev, I);
    } else {
        // printf("Calling Sgemm normal...\n");
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, I, J, &alpha, Bdev, K, Adev, J, &beta, Cdev, K);
    }

    // if (I == 256) {
    //     exit(0);
    // }

    // printf("Freeing device structures...\n");
    cudaMemcpy(D, Cdev, I*K*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
    cublasDestroy(handle);

    // exit(0);
}

#endif

/* =================================================== */
/* Everything below here should be host-callable only! */
/* =================================================== */

/* ================= */
/* Utility functions */
/* ================= */

__host__ float **dmatrix(int n, int m) {
    float *data = (float *)calloc(n*m, sizeof(float));
    float **M = (float **)malloc(n*sizeof(float *));
    for (int i = 0; i < n; i++) {
        M[i] = &data[i*m];
    }
    return M;
}

__host__ void dmatrix_free(float **M) {
    free(M[0]);
    free(M);
}

__host__ void ReLU(float **X, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            X[i][j] = X[i][j] > 0 ? X[i][j] : 0;
        }
    }
}

__host__ void softmax(float **X, int n, int m) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            X[j][i] = exp(X[j][i]);
            sum += X[j][i];
        }
        for (int j = 0; j < n; j++) {
            X[j][i] /= sum;
        }
    }
}

__host__ void shuffle(mnist_data *data, int size) {
    for (int i = 0; i < size-1; i++) {
        int j = i + rand() / (RAND_MAX / (size - i) + 1);
        mnist_data t = data[j];
        data[j] = data[i];
        data[i] = t;
    }
}


/* ====================================== */
/* Code for implementing the Layer struct */
/* ====================================== */

typedef struct Layer_ {
    struct  Layer_ *prev, *next;                       // Previous and next layers
    int     m, n, batch_size;                          // Number of neurons in previous and current layer
    float   **W, **B, **A, **Err, **Wgrad;             // Weights, biases, activations, gradients, and errors
    void   (*activation)(float**, int, int);           // Activation function
} Layer;

// Makes layer with n neurons, assuming prev layer had m neurons
__host__ Layer *new_layer(int m, int n, int batch_size, void (*activation)(float**, int, int)) {
    Layer *res = (Layer *)malloc(sizeof(Layer));
    res->m = m;
    res->n = n;
    res->batch_size = batch_size;
    res->W = dmatrix(n, m);
    res->B = dmatrix(n, batch_size);
    res->A = dmatrix(n, batch_size);
    res->Err = dmatrix(batch_size, n);
    res->Wgrad = dmatrix(n, m);
    res->activation = activation;

    memset(res->B[0], 0, n * batch_size * sizeof(float));

    float ub = 1/sqrt(m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res->W[i][j] = (float)rand()/(float)(RAND_MAX) * 2 * ub - ub;
        }
    }

    return res;
}

__host__ void free_layer(Layer *l) {
    dmatrix_free(l->W);
    dmatrix_free(l->B);
    dmatrix_free(l->A);
    dmatrix_free(l->Err);
    dmatrix_free(l->Wgrad);
    free(l);
}

// Assumes the previous layer's activations are set
__host__ void activate(Layer *l, int max_batch_size) {
    mmult_setup(l->n, l->m, l->batch_size, l->W[0], l->prev->A[0], l->B[0], l->A[0], false);
    // if (l->n == 256) {
    //     printf("Just did multiplication for activation layer 1 -> 2\n");
    //     exit(0);
    // }
    l->activation(l->A, l->n, l->batch_size);
}

// Calculates errors for final layer
__host__ void calculate_error_output(Layer *l, int *labels) {
    for (int i = 0; i < l->batch_size; i++) {
        for (int j = 0; j < l->n; j++) {
            l->Err[i][j] = l->A[j][i];
            if (labels[i] == j) {
                l->Err[i][j] -= 1;
            }
        }
    }
}

// Calculates errors for hidden and input layers
__host__ void calculate_error(Layer *l) {
    mmult_setup(l->batch_size, l->next->n, l->n, l->next->Err[0], l->next->W[0], NULL, l->Err[0], false);
    // Perform Hadamard product
    for (int i = 0; i < l->batch_size; i++) {
        for (int j = 0; j < l->n; j++) {
            if (l->A[j][i] <= 0) {
                l->Err[i][j] = 0;
            }
        }
    }
}

// Calculate batch-average gradient and update bias and weights
__host__ void update(Layer *l, float lr, int max_batch_size) {
    // Calculate weight gradient
    mmult_setup(l->m, l->batch_size, l->n, l->prev->A[0], l->Err[0], NULL, l->Wgrad[0], true);
    // Update weights and biases
    for (int i = 0; i < l->n; i++) {
        for (int j = 0; j < l->m; j++) {
            l->W[i][j] -= lr / l->batch_size * l->Wgrad[i][j];
        }
    }
    for (int i = 0; i < l->batch_size; i++) {
        for (int j = 0; j < l->n; j++) {
            for (int k = 0; k < l->batch_size; k++) {
                l->B[j][k] -= lr / l->batch_size * l->Err[i][j];
            }
        }
    }
}

/* ========================================================================================  */
/* DONE WITH LAYER CODE. From here on down writing functions for dealing with entire models. */
/* ========================================================================================  */

typedef struct Model_ {
    int     depth;      // Number of layers
    Layer   **layers;   // Array to hold layers
    int     *labels;
    int max_batch_size;
} Model;

__host__ Model *create_model(int depth, int sizes[], int max_batch_size) {
    Model *model = (Model *)malloc(sizeof(Model));
    Layer **layers = (Layer **)malloc(sizeof(Layer*) * depth);
    // Create input layer
    layers[0] = new_layer(0, sizes[0], max_batch_size, ReLU);
    for (int i = 1; i < depth-1; i++) {
        layers[i] = new_layer(sizes[i-1], sizes[i], max_batch_size, ReLU);
        layers[i]->prev = layers[i-1];
        layers[i-1]->next = layers[i];
    }
    // Create output layer with softmax
    layers[depth-1] = new_layer(sizes[depth-2], sizes[depth-1], max_batch_size, softmax);
    layers[depth-1]->prev = layers[depth-2];
    layers[depth-2]->next = layers[depth-1];

    model->depth = depth;
    model->layers = layers;
    model->labels = (int *)malloc(sizeof(int) * max_batch_size);
    model->max_batch_size = max_batch_size;

    return model;
}

__host__ void free_model(Model *model) {
    for (int i = 0; i < model->depth; i++) {
        free_layer(model->layers[i]);
    }
    free(model->layers);
    free(model);
}

__host__ void forward(Model *m, mnist_data *data, int batch_size) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    // Set batch size
    for (int i = 0; i < m->depth; i++) {
        m->layers[i]->batch_size = batch_size;
    }
    // Feed input into first layer
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                m->layers[0]->A[j*28 + k][i] = (float)data[i].data[j][k];
            }
        }
    }
    // Forward propagation
    for (int i = 1; i < m->depth; i++) {
        activate(m->layers[i], m->max_batch_size);
    }
}

__host__ void backward(Model *m, mnist_data *data, int batch_size, float lr) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    // Read in labels
    for (int i = 0; i < batch_size; i++) {
        m->labels[i] = data[i].label;
    }
    // Calculate errors in output layer
    calculate_error_output(m->layers[m->depth-1], m->labels);
    // Calculate errors for hidden layers
    for (int i = m->depth-2; i > 0; i--) {
        calculate_error(m->layers[i]);
    }
    // Perform gradient step
    for (int i = 1; i < m->depth; i++) {
        update(m->layers[i], lr, m->max_batch_size);
    }
}

__host__ void calculate_cross_entropy(Model *m, int batch_size, int size, mnist_data *data) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    Layer *output = m->layers[m->depth-1];
    int num_batches = size / batch_size;
    float sum = 0;
    for (int i = 0; i < num_batches; i++) {
        forward(m, &data[i*batch_size], batch_size);
        for (int j = 0; j < batch_size; j++) {
            sum -= log(output->A[data[i*batch_size + j].label][j]);
        }
    }
    if (lossIdx >= 50) {
        return;
    }
    loss[lossIdx++] =  sum / (num_batches * batch_size);
}

__host__ int count_errors(Model *m, int batch_size) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    // Assumes that labels are already loaded
    Layer *output = m->layers[m->depth-1];
    int count = 0;
    for (int i = 0; i < batch_size; i++) {
        int pred = 0;
        float maxval = 0;
        for (int j = 0; j < output->n; j++) {
            if (output->A[j][i] > maxval) {
                pred = j;
                maxval = output->A[j][i];
            }
        }
        if (pred != m->labels[i]) {
            count++;
        }
    }
    return count;
}

// Performs a single epoch of SGD
__host__ void batch_SGD(Model *m, int batch_size, int train_size, mnist_data *data, float lr) {
    int num_batches = train_size / batch_size; // Round down so we don't have to adjust bsize
    for (int i = 0; i < num_batches; i++) {
        forward(m, &data[i*batch_size], batch_size);
        backward(m, &data[i*batch_size], batch_size, lr);
    }
}

__host__ void test_accuracy(Model *m, int size, mnist_data *test_set, int batch_size) {
    int errors = 0;
    Layer *output = m->layers[m->depth-1];
    for (int i = 0; i < size; i += batch_size) {
        int bsize = fmin(batch_size, size-i);
        forward(m, &test_set[i], bsize);
        for (int j = 0; j < bsize; j++) {
            int pred = 0;
            float maxval = 0;
            for (int k = 0; k < output->n; k++) {
                if (output->A[k][j] > maxval) {
                    maxval = output->A[k][j];
                    pred = k;
                }
            }
            if (pred != test_set[i+j].label) {
                errors += 1;
            }
        }
    }
    printf("Test accuracy: %4.2f%%\n", (1 - ((float)errors / size)) * 100);
}


int main() {
    // Create model w/ 4 layers: 784 -> 128 -> 256 -> 10
    // Have to malloc because nvcc doesn't like taking addresses of temp arrays :(
    int *sizes = (int *)malloc(sizeof(int) * 4);
    sizes[0] = 784;
    sizes[1] = 128;
    sizes[2] = 256;
    sizes[3] = 10;
    Model *model = create_model(4, sizes, BATCH_SIZE);
    free(sizes);
    
    mnist_data *train_data, *test_data;
    unsigned int train_cnt, test_cnt;
    int ret, epochs;
    float lr;
    GpuTimer trainTimer, infTimer, totalTimer;
    srand(time(NULL));

    printf("Loading MNSIT data...\n");
    if (ret = mnist_load("../MNIST/train-images.idx3-ubyte", "../MNIST/train-labels.idx1-ubyte", &train_data, &train_cnt)) {
        printf("An error occured: %d\n", ret);
        exit(0);
    } else {
        printf("Training set size: %d\n", train_cnt); 
    }
    if (ret = mnist_load("../MNIST/t10k-images.idx3-ubyte", "../MNIST/t10k-labels.idx1-ubyte", &test_data, &test_cnt)) {
        printf("An error occured: %d\n", ret);
        exit(0);
    } else {
        printf("Testing set size: %d\n", test_cnt); 
    }

    // Training phase
    epochs = 50;
    lr = 0.1;
    printf("\nTraining parameters:\nBatch size: %d\nEpochs: %d\nLearning rate: %5.4f\n", BATCH_SIZE, epochs, lr);
    totalTimer.Start();
    trainTimer.Start();
    for (int epoch = 1; epoch < epochs+1; epoch++) {
        printf("\nEpoch %d:\n", epoch);
        shuffle(train_data, train_cnt);
        batch_SGD(model, BATCH_SIZE, 50000, train_data, lr);
        calculate_cross_entropy(model, BATCH_SIZE, 10000, &train_data[50000]);
    }
    trainTimer.Stop();
    printf("\nTraining time: %g(s)\n", trainTimer.Elapsed() / 1000);
    printf("Grind rate (training): %.2f(samples/s)\n", (float)((train_cnt / BATCH_SIZE) * BATCH_SIZE * epochs) / (float)trainTimer.Elapsed() * 1000);

    // Testing phase
    printf("\nValidating on test set:\n");
    infTimer.Start();
    test_accuracy(model, test_cnt, test_data, BATCH_SIZE);
    infTimer.Stop();
    printf("Inference time: %g(s)\n", infTimer.Elapsed() / 1000);
    printf("Grind rate (inference): %.2f(samples/s)\n", (float)((test_cnt / BATCH_SIZE) * BATCH_SIZE) / infTimer.Elapsed() * 1000);
    totalTimer.Stop();
    printf("\nTotal time: %g(s)\n", totalTimer.Elapsed() / 1000);
    printf("Grind rate (overall): %.2f(samples/s)\n", (float)((train_cnt / BATCH_SIZE) * BATCH_SIZE * epochs + (test_cnt / BATCH_SIZE) * BATCH_SIZE) / totalTimer.Elapsed() * 1000);

    // Printing out losses in array form for copy-pasting
    printf("Loss: [");
    for (int i = 0; i < 50; i++) {
        printf("%.7f", loss[i]);
        if (i < 49) {
            printf(", ");
        }
    }
    printf("]\n");

    free(train_data);
    free(test_data);
    free_model(model);
    return 0;
}