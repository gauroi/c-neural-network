#include "embedding.h"
#include "matrix.h"
#include <stddef.h>
#include <string.h>


EmbeddingLayer create_embedding_layer(int vocab_size, size_t emb_dim)
{
    EmbeddingLayer embLayer;
    embLayer.emb_dim = emb_dim;

    embLayer.mat = create_matrix(vocab_size, emb_dim);
    rand_matrix(&embLayer.mat, -0.1, 0.1);

    return embLayer;
}


void free_embedding_layer(EmbeddingLayer* embLayer)
{
    free_matrix(&embLayer->mat);
}


Matrix forward_embedding(EmbeddingLayer* emb_layer, const int* tokens, size_t tokens_count)
{
    Matrix embeddings = create_matrix(tokens_count, EMBEDDING_DIM);

    float* d = embeddings.data;

    for (int i = 0; i < tokens_count; i++) {
        float* s = emb_layer->mat.data + (tokens[i] * EMBEDDING_DIM);

        memcpy(d,s, EMBEDDING_DIM * sizeof(float));

        d += EMBEDDING_DIM;
    }

    return embeddings;
}