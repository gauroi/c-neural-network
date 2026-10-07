#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix* mat);

void save_weights(Matrix* mat, const char* filename);
void load_weights(Matrix* mat, const char* filename);

void rand_matrix(Matrix* mat, float min, float max);

Matrix matrix_multiply(const Matrix* a, const Matrix* b);

void ReLU(Matrix* matrix);
void softmax(Matrix* matrix);

#endif