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
        printf("malloc error.");
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