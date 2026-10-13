#include "train.h"
#include "matrix.h"
#include "embedding.h"
#include "linear.h"
#include "tokenizer.h"
#include "loss.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void train_network(EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer, Tokenizer* tkz, const char* text, int cycles, float lr, int context_size)
{
    size_t tokens_count = 0;
    int* tokens = encode(tkz, text, &tokens_count);

    if (tokens_count <= context_size) {
        printf("text is too short!\n");
        free(tokens);
        exit(8);
    }

    int windows_count = (int)tokens_count - context_size;

    for (int it = 0; it < cycles; it++) {
        Matrix embeddings = forward_embedding(emb_layer, tokens, tokens_count);

        Matrix gradient_embeddings = create_matrix(embeddings.rows, embeddings.cols);
        zero_matrix(&gradient_embeddings);

        zero_matrix(&hidden_layer->gradient_weights);
        zero_matrix(&hidden_layer->gradient_biases);
        zero_matrix(&output_layer->gradient_weights);
        zero_matrix(&output_layer->gradient_biases);

        float total_loss = 0.0f;
        for (int s = 0; s < windows_count; s++) {
            Matrix flattened_input = create_matrix(1, context_size * EMBEDDING_DIM);

            for (int t = 0; t < context_size; t++) {
                int current_token_indx = s + t;
                float* src = embeddings.data + (current_token_indx * EMBEDDING_DIM);
                float* dst = flattened_input.data + (t * EMBEDDING_DIM);
                memcpy(dst, src, EMBEDDING_DIM * sizeof(float));
            }

            Matrix response = forward_linear(hidden_layer, &flattened_input);
            ReLU(&response);
            Matrix logits = forward_linear(output_layer, &response);
            softmax(&logits);
            
            int target_token = tokens[s + context_size];
            total_loss += cross_entropy_loss(&logits, &target_token, 1);

            Matrix gradient_logits = backward_softmax(&logits, &target_token, 1);
            Matrix gradient_response = backward_linear(output_layer, &response, &gradient_logits);
            backward_ReLU(&gradient_response, &response);

            Matrix gradient_flattened = backward_linear(hidden_layer, &flattened_input, &gradient_response);

            for (int t = 0; t < context_size; t++) {
                int current_token_indx = s + t;
                float* src = gradient_flattened.data + (t * EMBEDDING_DIM);
                float* dst = gradient_embeddings.data + (current_token_indx * EMBEDDING_DIM);
                for (int w = 0; w < EMBEDDING_DIM; w++) {
                    dst[w] += src[w];
                }
            }

            free_matrix(&gradient_flattened);
            free_matrix(&gradient_logits);
            free_matrix(&gradient_response);
            free_matrix(&logits);
            free_matrix(&response);
            free_matrix(&flattened_input);
        } 

        if (it % 20 == 0 || it == cycles - 1) {
            printf("%4d / %4d | loss: %f\n", it, cycles, total_loss / windows_count);
        }

        backward_embedding(emb_layer, &gradient_embeddings, tokens, tokens_count, lr);

        matrix_update(&output_layer->weights, &output_layer->gradient_weights, lr);
        matrix_update(&output_layer->biases, &output_layer->gradient_biases, lr);
        matrix_update(&hidden_layer->weights, &hidden_layer->gradient_weights, lr);
        matrix_update(&hidden_layer->biases, &hidden_layer->gradient_biases, lr);
        free_matrix(&gradient_embeddings);
        free_matrix(&embeddings);
    }

    free(tokens);
    printf("training has be successfully!\n");
}
