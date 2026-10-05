#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

typedef struct {
    char* vocab;
    int vocab_size;
} Tokenizer;

Tokenizer create_tokenizer(const char* vocab);
void free_tokenizer(Tokenizer* tkz);

int* encode(Tokenizer* tkz, const char* text, size_t* tokens_count);
char* decode(Tokenizer* tkz, const int* tokens, size_t tokens_count);

#endif