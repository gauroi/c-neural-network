#ifndef TRAIN_H
#define TRAIN_H

#include "matrix.h"
#include "embedding.h"
#include "linear.h"
#include "tokenizer.h"
#include "loss.h"
#include <stdio.h>

void train_network(EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer, Tokenizer* tkz, const char* text, int cycles, float lr);

#endif