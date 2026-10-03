#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
    char* vocab;
    int vocab_size;
} Tokenizer;

Tokenizer create_tokenizer(const char* vocab);
void free_tokenizer(Tokenizer* tkz);

int* encode(Tokenizer* tkz, const char* text, int* arr_size);
char* decode(Tokenizer* tkz, const int* tokens, int arr_size);