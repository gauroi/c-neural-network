#ifndef MATRIX_H
#include <stdio.h>
#define MATRIX_H

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix* mat);

void save_weights(const Matrix* mat, FILE* file);
void load_weights(Matrix* mat, FILE* file);

void rand_matrix(Matrix* mat, float min, float max);
void zero_matrix(Matrix* mat);

Matrix matrix_multiply(const Matrix* a, const Matrix* b);
Matrix matrix_transpose(const Matrix* matrix);

void ReLU(Matrix* matrix);
void backward_ReLU(Matrix* gradient, const Matrix* response);
void softmax(Matrix* matrix);

void matrix_update(Matrix* matrix, const Matrix* gradient, float lr);

int matrix_argmax(const Matrix* mat, int row);

#endif