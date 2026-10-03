# c-neural-network

Simple educational neural network written in pure C from scratch.

## Project Structure

```text
├── matrix.h       # Matrix engine interface
├── matrix.c       # Memory allocation and binary weight I/O
├── tokenizer.h    # Tokenizer interface
├── tokenizer.c    # Character-level encoding and decoding
└── main.c         # Entry point and module testing
```

## Build

```bash
gcc main.c matrix.c tokenizer.c -o ai_model
```
