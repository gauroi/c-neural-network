#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "matrix.h"
#include <stdlib.h>

#define EMBEDDING_DIM 16

typedef struct {
    Matrix mat;
    size_t emb_dim; //16
} EmbeddingLayer;


EmbeddingLayer create_embedding_layer(int vocab_size, size_t emb_dim);
void free_embedding_layer(EmbeddingLayer* emb_layer);

void save_embedding_layer(const EmbeddingLayer* emb_layer, FILE* file);
void load_embedding_layer(EmbeddingLayer* emb_layer, FILE* file);

Matrix forward_embedding(EmbeddingLayer* emb_layer, const int* tokens, size_t tokens_count);
void backward_embedding(EmbeddingLayer* emb_layer, const Matrix* gradient_embeddings, const int *tokens, int symbols_count, float lr);

#endif