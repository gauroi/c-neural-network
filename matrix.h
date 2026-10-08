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

Matrix matrix_multiply(const Matrix* a, const Matrix* b);

void ReLU(Matrix* matrix);
void softmax(Matrix* matrix);

void matrix_update(Matrix* matrix, const Matrix* gradient, float lr);

#endif