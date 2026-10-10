#include "network.h"
#include "matrix.h"
#include "embedding.h"
#include "linear.h"


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