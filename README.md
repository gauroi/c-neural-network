# c-neural-network

Simple educational neural network written in pure C from scratch.
Features a character-level tokenizer and an embedding layer to process text data character by character.

## Project Structure

```text
├── matrix.h       # Matrix engine interface
├── matrix.c       # Memory allocation and binary weight I/O
├── tokenizer.h    # Tokenizer interface
├── tokenizer.c    # Character-level encoding and decoding
├── embedding.h    # Embedding layer interface
├── embedding.c    # Embedding layer logic and weights initialization
└── main.c         # Entry point and module testing
```

## Build

```bash
gcc main.c matrix.c tokenizer.c embedding.c -o ai_model
```
