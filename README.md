# c-neural-network

Simple educational neural network written in pure C from scratch.
Features a character-level tokenizer, an embedding layer, and a hidden linear layer with ReLU activation to process text data character by character.

## Project Structure

```text
├── matrix.h       # Matrix engine interface
├── matrix.c       # Memory allocation, binary weight I/O, and ReLU activation
├── tokenizer.h    # Tokenizer interface
├── tokenizer.c    # Character-level encoding and decoding
├── embedding.h    # Embedding layer interface
├── embedding.c    # Embedding layer logic and weights initialization
├── linear.h       # Linear layer interface
├── linear.c       # Fully connected layer logic and bias addition
└── main.c         # Entry point and module testing
```

## Build

```bash
gcc main.c matrix.c tokenizer.c embedding.c linear.c -o ai_model
```

## Usage

```bash
./ai_model
```

