#include "embedding.h"
#include "matrix.h"



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