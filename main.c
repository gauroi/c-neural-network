#include "matrix.h"
#include "linear.h"
#include "tokenizer.h"
#include "embedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


int main(void)
{
    Tokenizer tokenizer = create_tokenizer("abcdefghijklmnopqrstuvwxyz ,.?");
    EmbeddingLayer embLayer = create_embedding_layer(tokenizer.vocab_size, EMBEDDING_DIM);
    LinearLayer linearLayer = create_linear_layer(EMBEDDING_DIM, HIDDEN_DIM);

    size_t tokens_count = 0;

    printf("enter message: ");
    char text[256];
    if (fgets(text, sizeof(text), stdin)) {
        text[strcspn(text, "\n")] = '\0';
    }

    for (int i = 0; text[i] != '\0'; i++) {
        text[i] = tolower((unsigned char)text[i]);
    }

    int* tokens = encode(&tokenizer, text, &tokens_count);
    Matrix embeddings = forward_embedding(&embLayer, tokens, tokens_count);

    Matrix response = forward_linear(&linearLayer, &embeddings);
    ReLU(&response);

    for (int i = 0; i < tokens_count; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            printf("%f ", response.data[i * response.cols + j]);
        }
        puts("");
    }


    free_matrix(&embeddings);
    free(tokens);
    free_linear_layer(&linearLayer);
    free_embedding_layer(&embLayer);
    free_tokenizer(&tokenizer);

    return 0;
}