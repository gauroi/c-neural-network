#ifndef LINEAR_H
#define LINEAR_H

#define HIDDEN_DIM 64

#include "matrix.h"
#include "embedding.h"
#include <stddef.h>


typedef struct {
    Matrix weights;
    Matrix biases;
    Matrix gradient_weights;
    Matrix gradient_biases;
} LinearLayer;


LinearLayer create_linear_layer(size_t in_features, size_t out_features);
void free_linear_layer(LinearLayer* linearLayer);

void save_linear_layer(const LinearLayer* linearLayer, FILE* file);
void load_linear_layer(LinearLayer* linearLayer, FILE* file);

Matrix forward_linear(LinearLayer* linearLayer, const Matrix* input);
Matrix backward_linear(LinearLayer* linearLayer, const Matrix* input, const Matrix* gradient_out);

#endif