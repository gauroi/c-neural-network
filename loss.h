#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include <math.h>

float cross_entropy_loss(const Matrix* logits, const int* targets, int symbol_count);
Matrix backward_softmax(const Matrix* logits, const int* tokens, int symbols_count);

#endif