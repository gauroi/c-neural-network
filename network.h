#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "embedding.h"
#include "linear.h"


void save_network(const char* filename, EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer);
void load_network(const char* filename, EmbeddingLayer* emb_layer, LinearLayer* hidden_layer, LinearLayer* output_layer);


#endif