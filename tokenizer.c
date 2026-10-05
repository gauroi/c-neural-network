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


int* encode(Tokenizer* tkz, const char* text, size_t* tokens_count)
{
    *tokens_count = strlen(text);
    int* tokens = malloc(*(tokens_count) * sizeof(*tokens));

    if (tokens == NULL) {
        printf("malloc error!");
        exit(1);
    }

    for (size_t i = 0; i < *tokens_count; i++) {
        char* ptr = strchr(tkz->vocab, text[i]);
        if (ptr == NULL) {
            char* ptrc = strchr(tkz->vocab, ' ');
            tokens[i] = (ptrc - tkz->vocab);
        }
        else {
            tokens[i] = (ptr - tkz->vocab);    
        }    
    }
    return tokens;
}

char* decode(Tokenizer* tkz, const int* tokens, size_t tokens_count)
{
    char* response = malloc(tokens_count * sizeof(*response) + 1);
    if (response == NULL) {
        printf("malloc error!");
        exit(1);
    }

    for(size_t i = 0; i < tokens_count; i++) {
        if (tokens[i] < 0 || tokens[i] >= tkz->vocab_size) {
            response[i] = '#';
        }
        else {
            response[i] = tkz->vocab[tokens[i]];
        }
    }

    response[tokens_count] = '\0';
    return response;
}