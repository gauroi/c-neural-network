#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix create_matrix(int rows, int cols)
void free_matrix(Matrix* mat)

#endif