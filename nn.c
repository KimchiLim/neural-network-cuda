#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include "mnist.h"

// #define CBLAS
// #include <cblas.h>
// // ^ Comment out if not using CBLAS ^

#define BLOCK_SIZE 8
#define BATCH_SIZE 500
#define CORE_COUNT 16

float loss[50] = {0};
int lossIdx = 0;

/* ========================= */
/* General utility functions */
/* ========================= */

float **dmatrix(int n, int m) {
    float *data = (float *)calloc(n*m, sizeof(float));
    float **M = (float **)malloc(n*sizeof(float *));
    for (int i = 0; i < n; i++) {
        M[i] = &data[i*m];
    }
    return M;
}

void dmatrix_free(float **M) {
    free(M[0]);
    free(M);
}

void ReLU(float **X, int n, int m) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            X[i][j] = X[i][j] > 0 ? X[i][j] : 0;
        }
    }
}

void softmax(float **X, int n, int m) {
    #pragma omp parallel for
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

void shuffle(mnist_data *data, int size) {
    for (int i = 0; i < size-1; i++) {
        int j = i + rand() / (RAND_MAX / (size - i) + 1);
        mnist_data t = data[j];
        data[j] = data[i];
        data[i] = t;
    }
}

#ifndef CBLAS

// Stores A x B + C in D. Ignores C if NULL. Assumes A is IxJ, B is JxK, C and D are IxK.
void mmult(int I, int J, int K, float **A, float **B, float **C, float **D) {
    int Iblocks = ceil((float)I / BLOCK_SIZE);
    int Jblocks = ceil((float)J / BLOCK_SIZE);
    int Kblocks = ceil((float)K / BLOCK_SIZE);
    
    // Assumes D has shape IxK
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < Iblocks; x++) {
        // #pragma omp parallel for num_threads(4)
        for (int z = 0; z < Kblocks; z++) {
            // Load block of C into D
            for (int i = x * BLOCK_SIZE; i < fmin((x+1) * BLOCK_SIZE, I); i++) {
                for (int k = z * BLOCK_SIZE; k < fmin((z+1) * BLOCK_SIZE, K); k++) {
                    if (C != NULL) {
                        D[i][k] = C[i][k];
                    } else {
                        D[i][k] = 0;
                    }
                }
            }
            // Now do tiled multiplication of A and B
            for (int y = 0; y < Jblocks; y++) {
                for (int i = x * BLOCK_SIZE; i < fmin((x+1) * BLOCK_SIZE, I); i++) {
                    for (int j = y * BLOCK_SIZE; j < fmin((y+1) * BLOCK_SIZE, J); j++) {
                        for (int k = z * BLOCK_SIZE; k < fmin((z+1) * BLOCK_SIZE, K); k++) {
                            D[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
}

// mmult but stores the transpose of the result
void mmultT(int I, int J, int K, float **A, float **B, float **C, float **D) {
    int Iblocks = ceil((float)I / BLOCK_SIZE);
    int Jblocks = ceil((float)J / BLOCK_SIZE);
    int Kblocks = ceil((float)K / BLOCK_SIZE);

    // Assumes D has shape KxI
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < Iblocks; x++) {
        // #pragma omp parallel for num_threads(4)
        for (int z = 0; z < Kblocks; z++) {
            // Load block of C into D
            for (int i = x * BLOCK_SIZE; i < fmin((x+1) * BLOCK_SIZE, I); i++) {
                for (int k = z * BLOCK_SIZE; k < fmin((z+1) * BLOCK_SIZE, K); k++) {
                    if (C != NULL) {
                        D[k][i] = C[i][k];
                    } else {
                        D[k][i] = 0;
                    }
                }
            }
            // Now do tiled multiplication of A and B
            for (int y = 0; y < Jblocks; y++) {
                for (int i = x * BLOCK_SIZE; i < fmin((x+1) * BLOCK_SIZE, I); i++) {
                    for (int j = y * BLOCK_SIZE; j < fmin((y+1) * BLOCK_SIZE, J); j++) {
                        for (int k = z * BLOCK_SIZE; k < fmin((z+1) * BLOCK_SIZE, K); k++) {
                            D[k][i] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
}

#endif

// Adds B to A. Assumes both are nxm matrices
void madd(int n, int m, float **A, float **B) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] += B[i][j];
        }
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
Layer *new_layer(int m, int n, int batch_size, void (*activation)(float**, int, int)) {
    Layer *res = malloc(sizeof(Layer));
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

void free_layer(Layer *l) {
    dmatrix_free(l->W);
    dmatrix_free(l->B);
    dmatrix_free(l->A);
    dmatrix_free(l->Err);
    dmatrix_free(l->Wgrad);
    free(l);
}

// Assumes the previous layer's activations are set
void activate(Layer *l, int max_batch_size) {
    #ifndef CBLAS
    mmult(l->n, l->m, l->batch_size, l->W, l->prev->A, l->B, l->A);
    #else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l->n, l->batch_size, l->m, 1, l->W[0], l->m, l->prev->A[0], max_batch_size, 0, l->A[0], max_batch_size);
    madd(l->n, l->batch_size, l->A, l->B);
    #endif
    l->activation(l->A, l->n, l->batch_size);
}

// Calculates errors for final layer
void calculate_error_output(Layer *l, int *labels) {
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
void calculate_error(Layer *l) {
    #ifndef CBLAS
    mmult(l->batch_size, l->next->n, l->n, l->next->Err, l->next->W, NULL, l->Err);
    #else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l->batch_size, l->n, l->next->n, 1, l->next->Err[0], l->next->n, l->next->W[0], l->n, 0, l->Err[0], l->n);
    #endif
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
void update(Layer *l, float lr, int max_batch_size) {
    // Calculate weight gradient
    #ifndef CBLAS
    mmultT(l->m, l->batch_size, l->n, l->prev->A, l->Err, NULL, l->Wgrad);
    #else
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, l->n, l->m, l->batch_size, 1, l->Err[0], l->n, l->prev->A[0], max_batch_size, 0, l->Wgrad[0], l->m);
    #endif
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

Model *create_model(int depth, int sizes[], int max_batch_size) {
    Model *model = malloc(sizeof(Model));
    Layer **layers = malloc(sizeof(Layer*) * depth);
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
    model->labels = malloc(sizeof(int) * max_batch_size);
    model->max_batch_size = max_batch_size;

    return model;
}

void free_model(Model *model) {
    for (int i = 0; i < model->depth; i++) {
        free_layer(model->layers[i]);
    }
    free(model->layers);
    free(model);
}

void forward(Model *m, mnist_data *data, int batch_size) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    // Set batch size
    for (int i = 0; i < m->depth; i++) {
        m->layers[i]->batch_size = batch_size;
    }
    // Feed input into first layer
    #pragma omp parallel for num_threads(16)
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

void backward(Model *m, mnist_data *data, int batch_size, float lr) {
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

float calculate_cross_entropy(Model *m, int batch_size) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    // Assumes that labels are already loaded
    Layer *output = m->layers[m->depth-1];
    float sum = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(16)
    for (int i = 0; i < batch_size; i++) {
        sum -= log(output->A[m->labels[i]][i]);
    }
    return sum / batch_size;
}

void calculate_loss(Model *m, int batch_size, int size, mnist_data *data) {
    Layer *output = m->layers[m->depth-1];
    int num_batches = ceil((float)size / batch_size);
    float sum = 0;
    for (int i = 0; i < num_batches; i++) {
        int bsize = (i < num_batches-1) ? batch_size : size - (batch_size * (num_batches-1));
        forward(m, &data[i*batch_size], bsize);
        for (int j = 0; j < batch_size; j++) {
            sum -= log(output->A[data[i*batch_size + j].label][j]);
        }
    }
    if (lossIdx >= 50) {
        return;
    }
    loss[lossIdx++] =  sum / (num_batches * batch_size);
}

int count_errors(Model *m, int batch_size) {
    if (batch_size > m->max_batch_size) {
        printf("Batch size (%d) is larger than maximum allowed (%d). Exiting...\n", batch_size, m->max_batch_size);
        exit(0);
    }
    // Assumes that labels are already loaded
    Layer *output = m->layers[m->depth-1];
    int count = 0;
    #pragma omp parallel for reduction(+:count) num_threads(16)
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
void batch_SGD(Model *m, int batch_size, int train_size, mnist_data *data, float lr) {
    int num_batches = ceil((float)train_size / batch_size);
    for (int i = 0; i < num_batches; i++) {
        int bsize = (i < num_batches-1) ? batch_size : train_size - (batch_size * (num_batches-1));
        forward(m, &data[i*batch_size], bsize);
        backward(m, &data[i*batch_size], bsize, lr);
        // if ((i+1) % 100 == 0) {
        //     float cross_entropy = calculate_cross_entropy(m, bsize);
        //     int errors = count_errors(m, bsize);
        //     printf("\nBatch %d\nCross Entropy: %f\nErrors: %d\n", i+1, cross_entropy, errors);
        // }
    }
}

void test_accuracy(Model *m, int size, mnist_data *test_set, int batch_size) {
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
    int max_batch_size = 512;
    Model *model = create_model(4, (int[]){784, 128, 256, 10}, max_batch_size);
    mnist_data *train_data, *test_data;
    unsigned int train_cnt, test_cnt;
    int ret, epochs;
    float lr, total;
    double stTrain, ftTrain, stInf, ftInf;
    omp_set_num_threads(CORE_COUNT);
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
    epochs = 1;
    lr = 0.1;
    printf("\nTraining parameters:\nBatch size: %d\nEpochs: %d\nLearning rate: %5.4f\n", BATCH_SIZE, epochs, lr);
    stTrain = omp_get_wtime();
    for (int epoch = 1; epoch < epochs+1; epoch++) {
        printf("\nEpoch %d:\n", epoch);
        shuffle(train_data, train_cnt);
        batch_SGD(model, BATCH_SIZE, 50000, train_data, lr);
        calculate_loss(model, BATCH_SIZE, 10000, &train_data[50000]);
    }
    ftTrain = omp_get_wtime();
    total = (float)(ftTrain - stTrain);
    printf("\nTraining time: %f(s)\n", total);
    printf("Grind rate (training): %.2f(samples/s)\n", (float)(train_cnt * epochs) / total);

    // Testing phase
    printf("\nValidating on test set:\n");
    stInf = omp_get_wtime();
    test_accuracy(model, test_cnt, test_data, BATCH_SIZE);
    ftInf = omp_get_wtime();
    total = (float)(ftInf - stInf);
    printf("Inference time: %f(s)\n", total);
    printf("Grind rate (inference): %.2f(samples/s)\n", (float)(test_cnt) / total);
    printf("Grind rate (overall): %.2f(samples/s)\n", (float)(test_cnt + train_cnt * epochs) / (ftInf - stTrain));

    // Printing out losses in array form for copy-pasting
    printf("\nLoss: [");
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