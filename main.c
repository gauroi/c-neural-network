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

#define CONTEXT_SIZE 4

char* generate_text(int length, EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer, Tokenizer* tkz, const char* text)
{
    
    size_t tokens_count = 0;
    int* tokens = encode(tkz, text, &tokens_count);

    char* text_response = malloc((length + 1) * sizeof(*text_response));

    for (int i = 0; i < length; i++) {
        Matrix embeddings = forward_embedding(emb_layer, tokens, tokens_count);

        Matrix flattened_input = create_matrix(1, CONTEXT_SIZE * EMBEDDING_DIM);
        zero_matrix(&flattened_input);

        int start_token = (int)tokens_count - CONTEXT_SIZE;

        for (int t = 0; t < CONTEXT_SIZE; t++) {
            int current_token_indx = start_token + t;
            if (current_token_indx >= 0) {
                float* src = embeddings.data + (current_token_indx * EMBEDDING_DIM);
                float* dst = flattened_input.data + (t * EMBEDDING_DIM);
                memcpy(dst, src, EMBEDDING_DIM * sizeof(float)); 
            }
        }

        Matrix response = forward_linear(hidden_layer, &flattened_input);
        ReLU(&response);
        Matrix logits = forward_linear(output_layer, &response);
        softmax(&logits);

        int current_symbol = matrix_argmax(&logits, 0);
        
        text_response[i] = tkz->vocab[current_symbol];

        tokens = realloc(tokens, sizeof(*tokens) * (tokens_count + 1));
        tokens[tokens_count] = current_symbol;
        tokens_count++;

        free_matrix(&embeddings);
        free_matrix(&flattened_input);
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
    LinearLayer linearLayer = create_linear_layer(CONTEXT_SIZE * EMBEDDING_DIM, HIDDEN_DIM);
    LinearLayer outputLayer = create_linear_layer(HIDDEN_DIM, tokenizer.vocab_size);

    load_network("weights.bin", &embLayer, &linearLayer, &outputLayer);

    //train_network(&embLayer, &linearLayer, &outputLayer, &tokenizer, "hello", 20000, 0.01f);

    char* buff = generate_text(30, &embLayer, &linearLayer, &outputLayer, &tokenizer, "hel");
    printf("%s\n", buff);

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