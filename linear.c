#include "linear.h"
#include "matrix.h"
#include "embedding.h"
#include <stddef.h>


LinearLayer create_linear_layer(size_t in_features, size_t out_features)
{
    LinearLayer layer;
    layer.weights = create_matrix(in_features, out_features);
    layer.biases = create_matrix(1, out_features);
    layer.gradient_weights = create_matrix(in_features, out_features);
    layer.gradient_biases = create_matrix(1, out_features);

    rand_matrix(&layer.weights, -0.1, 0.1);
    rand_matrix(&layer.biases, -0.1, 0.1);


    return layer;
}


void free_linear_layer(LinearLayer* linearLayer)
{
    free_matrix(&linearLayer->weights);
    free_matrix(&linearLayer->biases);
    free_matrix(&linearLayer->gradient_weights);
    free_matrix(&linearLayer->gradient_biases);
}


void save_linear_layer(const LinearLayer* linearLayer, FILE* file)
{
    save_weights(&linearLayer->weights, file);
    save_weights(&linearLayer->biases, file);
}


void load_linear_layer(LinearLayer* linearLayer, FILE* file)
{
    load_weights(&linearLayer->weights, file);
    load_weights(&linearLayer->biases, file);
}


Matrix forward_linear(LinearLayer* linearLayer, const Matrix* input)
{
    Matrix response = matrix_multiply(input, &linearLayer->weights);

    for (int i = 0; i < response.rows; i++) {
        for (int j = 0; j < response.cols; j++) {
            response.data[i * response.cols + j] += linearLayer->biases.data[j];
        }
    }

    return response;
}


Matrix backward_linear(LinearLayer* linearLayer, const Matrix* input, const Matrix* gradient_out)
{
    zero_matrix(&linearLayer->gradient_biases);
    for (int i = 0; i < gradient_out->rows; i++) {
        for (int j = 0; j < gradient_out->cols; j++) {
            linearLayer->gradient_biases.data[j] += gradient_out->data[i * gradient_out->cols + j];
        }
    }

    Matrix input_temp = matrix_transpose(input);
    free_matrix(&linearLayer->gradient_weights);
    linearLayer->gradient_weights = matrix_multiply(&input_temp, gradient_out);
    free_matrix(&input_temp);

    Matrix weights_temp = matrix_transpose(&linearLayer->weights);
    Matrix gradient_input = matrix_multiply(gradient_out, &weights_temp);
    free_matrix(&weights_temp);
    
    return gradient_input;
}