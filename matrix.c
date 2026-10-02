#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>


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


void save_weights(Matrix* mat, const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Open file error!\n");
        exit(3);
    }
    fwrite(&(mat->rows), sizeof(int), 1, file);
    fwrite(&(mat->cols), sizeof(int), 1, file);
    fwrite(mat->data, sizeof(*mat->data), mat->rows * mat->cols, file);

    printf("weights are saved.\n");
    fclose(file);
}


void load_weights(Matrix* mat, const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Open file error!\n");
        exit(3);
    }
    fread(&(mat->rows), sizeof(int), 1, file);
    fread(&(mat->cols), sizeof(int), 1, file);

    mat->data = malloc(mat->rows * mat->cols * sizeof(*mat->data));

    fread(mat->data, sizeof(*mat->data), mat->rows * mat->cols, file);

    printf("weights are loaded.\n");
    fclose(file);
}