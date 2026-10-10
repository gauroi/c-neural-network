#include "train.h"
#include "matrix.h"
#include "embedding.h"
#include "linear.h"
#include "tokenizer.h"
#include "loss.h"
#include <stdio.h>

void train_network(EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer, Tokenizer* tkz, const char* text, int cycles, float lr)
{
    size_t tokens_count = 0;
    int* tokens = encode(tkz, text, &tokens_count);

    for (int it = 0; it < cycles; it++) {
        Matrix embeddings = forward_embedding(emb_layer, tokens, tokens_count);
        Matrix response = forward_linear(hidden_layer, &embeddings);
        ReLU(&response);
        Matrix logits = forward_linear(output_layer, &response);
        softmax(&logits);

        float loss = cross_entropy_loss(&logits, tokens + 1, tokens_count - 1);

        if (it % 20 == 0 || it == cycles - 1) {
            printf("%4d / %4d | loss: %f\n", it, cycles, loss);
        }

        Matrix gradient_logits = backward_softmax(&logits, tokens + 1, tokens_count - 1);
        Matrix gradient_response = backward_linear(output_layer, &response, &gradient_logits);
        backward_ReLU(&gradient_response, &response);
        Matrix gradient_embeddings = backward_linear(hidden_layer, &embeddings, &gradient_response);

        backward_embedding(emb_layer, &gradient_embeddings, tokens, tokens_count - 1, lr);

        matrix_update(&output_layer->weights, &output_layer->gradient_weights, lr);
        matrix_update(&output_layer->biases, &output_layer->gradient_biases, lr);
        matrix_update(&hidden_layer->weights, &hidden_layer->gradient_weights, lr);
        matrix_update(&hidden_layer->biases, &hidden_layer->gradient_biases, lr);

        free_matrix(&gradient_embeddings);
        free_matrix(&gradient_response);
        free_matrix(&gradient_logits);
        free_matrix(&logits);
        free_matrix(&response);
        free_matrix(&embeddings);
    }

    free(tokens);
    printf("training has be successfully!");
}
