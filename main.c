#include "matrix.h"
#include "tokenizer.h"
#include "embedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(void)
{
    Tokenizer tokenizer = create_tokenizer("abcdefghijklmnopqrstuvwxyz ,.?");
    EmbeddingLayer embLayer = create_embedding_layer(tokenizer.vocab_size, EMBEDDING_DIM);

    size_t tokens_count = 0;

    printf("enter message: ");
    char text[256];
    if (fgets(text, sizeof(text), stdin)) {
        text[strcspn(text, "\n")] = '\0';
    }

    int* tokens = encode(&tokenizer, text, &tokens_count);
    Matrix embeddings = forward_embedding(&embLayer, tokens, tokens_count);

    for (int i = 0; i < embeddings.rows; i++) {
        for (int j = 0; j < embeddings.cols; j++) {
            printf("%f ", embeddings.data[i * embeddings.cols + j]);
        }
        puts("");
    }

    free_matrix(&embeddings);
    free(tokens);
    free_embedding_layer(&embLayer);
    free_tokenizer(&tokenizer);

    return 0;
}