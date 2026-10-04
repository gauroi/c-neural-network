#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "matrix.h"

typedef struct {
    Matrix mat;
    size_t emb_dim;
} EmbeddingLayer;


EmbeddingLayer create_embedding_layer(int vocab_size, size_t emb_dim);
void free_embedding_layer(EmbeddingLayer* emb_layer);

#endif