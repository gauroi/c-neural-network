#include "loss.h"
#include "matrix.h"
#include <math.h>

float cross_entropy_loss(const Matrix* logits, const int* targets, int symbol_count)
{
    float total_loss = 0.0f;
    for (int i = 0; i < symbol_count; i++) {
        float loss = logits->data[i * logits->cols + targets[i]];

        total_loss += -logf(loss + 1e-15f);
    }

    return total_loss / symbol_count;
}


Matrix backward_softmax(const Matrix* logits, const int* tokens, int symbols_count)
{
    Matrix gradient = create_matrix(logits->rows, logits->cols);
    for (int i = 0; i < logits->rows * logits->cols; i++) {
        gradient.data[i] = logits->data[i];
    }

    for (int i = 0; i < symbols_count; i++) {
        int indx = tokens[i];
        gradient.data[i * gradient.cols + indx] -= 1.0f;
    }

    for (int i = 0; i < gradient.rows * gradient.cols; i++) {
        gradient.data[i] /= symbols_count;
    }

    return gradient;
}