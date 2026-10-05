#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "matrix.h"
#define EMBEDDING_DIM 16

typedef struct {
    Matrix mat;
    size_t emb_dim; //16
} EmbeddingLayer;


EmbeddingLayer create_embedding_layer(int vocab_size, size_t emb_dim);
void free_embedding_layer(EmbeddingLayer* emb_layer);

Matrix forward_embedding(EmbeddingLayer* emb_layer, int* tokens, size_t tokens_count);

#endif