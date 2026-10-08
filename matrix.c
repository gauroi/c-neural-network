#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


Matrix create_matrix(int rows, int cols)
{
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;

    mat.data = malloc(rows * cols * sizeof(float));
    if (mat.data == NULL) {
        printf("malloc error.\n");
        exit(1);
    }

    return mat;
}


void free_matrix(Matrix* mat)
{
    if (mat->data != NULL) {
        free(mat->data);
        mat->data = NULL;
    }
}


void save_weights(const Matrix* mat, FILE* file)
{
    if (file == NULL) {
        printf("Open file error!\n");
        exit(3);
    }
    fwrite(&(mat->rows), sizeof(int), 1, file);
    fwrite(&(mat->cols), sizeof(int), 1, file);
    fwrite(mat->data, sizeof(*mat->data), mat->rows * mat->cols, file);
}


void load_weights(Matrix* mat, FILE* file)
{
    if (file == NULL) {
        printf("Open file error!\n");
        exit(3);
    }
    fread(&(mat->rows), sizeof(int), 1, file);
    fread(&(mat->cols), sizeof(int), 1, file);

    mat->data = malloc(mat->rows * mat->cols * sizeof(*mat->data));

    fread(mat->data, sizeof(*mat->data), mat->rows * mat->cols, file);
}


void rand_matrix(Matrix* mat, float min, float max)
{
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = min + ((float)rand() / (float)RAND_MAX) * (max - min); 
    }
}


Matrix matrix_multiply(const Matrix* a, const Matrix* b)
{
    if (a->cols != b->rows) {
        printf("matrix multiply error!\n");
        exit(4);
    }

    Matrix matrix = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int d = 0; d < a->cols; d++) {
                sum += a->data[i * a->cols + d] * b->data[d * b->cols + j];
            }
            matrix.data[i * matrix.cols + j] = sum;
        }
    }

    return matrix;
}


void ReLU(Matrix* matrix) 
{
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = matrix->data[i] < 0 ? 0.0f : matrix->data[i];
    }
}


void softmax(Matrix* matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        float sum = 0.0f;

        for (int j = 0; j < matrix->cols; j++) {
            int index = i * matrix->cols +j;
            matrix->data[index] = expf(matrix->data[index]);
            sum += matrix->data[index];
        }

        for (int d = 0; d < matrix->cols; d++) {
            int index = i * matrix->cols + d;
            matrix->data[index] = matrix->data[index] / sum;
        }
    }
}