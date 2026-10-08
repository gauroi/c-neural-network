#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include <math.h>

float cross_entropy_loss(const Matrix* logits, const int* targets, int symbol_count);

#endif