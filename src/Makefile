# Compiler and flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
TARGET = symnmf

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): symnmf.c
	$(CC) $(CFLAGS) -o $(TARGET) symnmf.c -lm

# Clean up generated files
clean:
	rm -f $(TARGET)

