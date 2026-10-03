#include "tokenizer.h"
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