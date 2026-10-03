#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tokenizer create_tokenizer(const char* vocab)
{
    Tokenizer tkz;
    tkz.vocab_size = strlen(vocab);
    tkz.vocab = malloc(tkz.vocab_size * sizeof(*vocab) + 1);
    if (tkz.vocab == NULL) {
        printf("malloc error!");
        exit(1);
    }
    strcpy(tkz.vocab, vocab);

    return tkz;
}


void free_tokenizer(Tokenizer* tkz)
{
    if (tkz->vocab != NULL) {
        free(tkz->vocab);
        tkz->vocab = NULL;
    }
}


int* encode(Tokenizer* tkz, const char* text, int* arr_size)
{
    *arr_size = strlen(text);
    int* tokens = malloc(*(arr_size) * sizeof(*tokens));

    for (int i = 0; i < *arr_size; i++) {
        char* ptr = strchr(tkz->vocab, text[i]);
        if (ptr == NULL) {
            tokens[i] = tkz->vocab_size;
            continue;
        }
        tokens[i] = (ptr - tkz->vocab);
    }

    return tokens;
}

char* decode(Tokenizer* tkz, const int* tokens, int tokens_count)
{
    char* response = malloc(tokens_count * sizeof(*response) + 1);
    if (response == NULL) {
        printf("malloc error!");
        exit(1);
    }

    for(int i = 0; i < tokens_count; i++) {
        if (tokens[i] == tkz->vocab_size) {
            response[i] = '#';
            continue;
        }
        response[i] = tkz->vocab[tokens[i]];
    }

    response[tokens_count] = '\0';
    return response;
}