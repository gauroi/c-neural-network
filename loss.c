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