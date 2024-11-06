CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
TARGET = symnmf

all: $(TARGET)

# Rule to build the target executable, linking both symnmf.c and symnmfhelpers.c
$(TARGET): symnmf.c symnmfhelpers.c
	$(CC) $(CFLAGS) -o $(TARGET) symnmf.c symnmfhelpers.c -lm

# Clean up generated files
clean:
	rm -f $(TARGET)

