#include "matrix.h"
#include "network.h"
#include "linear.h"
#include "train.h"
#include "loss.h"
#include "tokenizer.h"
#include "embedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


char* generate_text(int length, EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer, Tokenizer* tkz, const char* text)
{
    
    size_t tokens_count = 0;
    int* tokens = encode(tkz, text, &tokens_count);

    char* text_response = malloc(1000 * sizeof(*text_response));

    for (int i = 0; i < length; i++) {
        Matrix embeddings = forward_embedding(emb_layer, tokens, tokens_count);
        Matrix response = forward_linear(hidden_layer, &embeddings);
        ReLU(&response);
        Matrix logits = forward_linear(output_layer, &response);
        softmax(&logits);

        int current_symbol = matrix_argmax(&logits, logits.rows - 1);
        
        text_response[i] = tkz->vocab[current_symbol];

        tokens = realloc(tokens, sizeof(*tokens) * (tokens_count + 1));
        tokens[tokens_count] = current_symbol;
        tokens_count++;

        free_matrix(&embeddings);
        free_matrix(&response);
        free_matrix(&logits);
    }
    text_response[length] = '\0';
    free(tokens);
    return text_response;
}

int main(void)
{
    Tokenizer tokenizer = create_tokenizer("abcdefghijklmnopqrstuvwxyz ,.?");
    EmbeddingLayer embLayer = create_embedding_layer(tokenizer.vocab_size, EMBEDDING_DIM);
    LinearLayer linearLayer = create_linear_layer(EMBEDDING_DIM, HIDDEN_DIM);
    LinearLayer outputLayer = create_linear_layer(HIDDEN_DIM, tokenizer.vocab_size);

    load_network("weights.bin", &embLayer, &linearLayer, &outputLayer);

    // size_t tokens_count = 0;

    // printf("enter message: ");
    // char text[256];
    // if (fgets(text, sizeof(text), stdin)) {
    //     text[strcspn(text, "\n")] = '\0';
    // }

    // for (int i = 0; text[i] != '\0'; i++) {
    //     text[i] = tolower((unsigned char)text[i]);
    // }

    // int* tokens = encode(&tokenizer, text, &tokens_count);
    // Matrix embeddings = forward_embedding(&embLayer, tokens, tokens_count);

    // Matrix response = forward_linear(&linearLayer, &embeddings);
    // ReLU(&response);

    // Matrix logits = forward_linear(&outputLayer, &response);
    // softmax(&logits);

    train_network(&embLayer, &linearLayer, &outputLayer, &tokenizer, "hello", 20000, 0.01f);

    save_network("weights.bin", &embLayer, &linearLayer, &outputLayer);

    // free(tokens);
    // free_matrix(&embeddings);
    // free_matrix(&response);
    // free_matrix(&logits);

    free_linear_layer(&outputLayer);
    free_linear_layer(&linearLayer);
    free_embedding_layer(&embLayer);
    free_tokenizer(&tokenizer);

    return 0;
}