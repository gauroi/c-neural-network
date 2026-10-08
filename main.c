#include "matrix.h"
#include "linear.h"
#include "tokenizer.h"
#include "embedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


void save_network(const char* filename, EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer)
{
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Open file error!\n");
        exit(3);
    }

    save_embedding_layer(emb_layer, file);
    save_linear_layer(hidden_layer, file);
    save_linear_layer(output_layer, file);

    fclose(file);
    printf("weights saved successfully.\n");
}


void load_network(const char* filename, EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer)
{
    FILE* file = fopen(filename, "rb");
    if (file == NULL){
        printf("Open file error!\n");
        exit(3);
    }

    load_embedding_layer(emb_layer, file);
    load_linear_layer(hidden_layer, file);
    load_linear_layer(output_layer, file);

    fclose(file);
    printf("weights loaded successfully.\n\n");
}

int main(void)
{
    Tokenizer tokenizer = create_tokenizer("abcdefghijklmnopqrstuvwxyz ,.?");
    EmbeddingLayer embLayer = create_embedding_layer(tokenizer.vocab_size, EMBEDDING_DIM);
    LinearLayer linearLayer = create_linear_layer(EMBEDDING_DIM, HIDDEN_DIM);
    LinearLayer outputLayer = create_linear_layer(HIDDEN_DIM, tokenizer.vocab_size);

    load_network("weights.bin", &embLayer, &linearLayer, &outputLayer);

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

    Matrix logits = forward_linear(&outputLayer, &response);
    softmax(&logits);

    for (int i = 0; i < tokens_count; i++) {
        for(int j = 0; j < tokenizer.vocab_size; j++) {
            printf("%f ", logits.data[i * logits.cols + j]);
        }
        puts("");
    }


    save_network("weights.bin", &embLayer, &linearLayer, &outputLayer);

    free(tokens);
    free_matrix(&embeddings);
    free_matrix(&logits);
    free_linear_layer(&outputLayer);
    free_linear_layer(&linearLayer);
    free_embedding_layer(&embLayer);
    free_tokenizer(&tokenizer);

    return 0;
}