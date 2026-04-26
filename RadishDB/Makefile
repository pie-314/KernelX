# Compiler and flags
CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -g

# Project
TARGET  = radishdb
BUILD   = build

# Source files (for binary)
SRC = \
	src/main.c \
	src/hashtable.c \
	src/persistence.c \
	src/aof.c \
	src/expires.c \
	src/utils.c \
	src/engine.c \
	src/repl.c \
	src/result.c \
	src/server.c

# Library source files (core functionality)
LIB_SRC = \
	src/hashtable.c \
	src/persistence.c \
	src/aof.c \
	src/expires.c \
	src/utils.c \
	src/engine.c \
	src/result.c \
	src/radish_ffi.c

# Convert src/foo.c -> build/foo.o
OBJ = $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))
LIB_OBJ = $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))

# Default target
all: $(TARGET) libradish.a

# Link binary
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET)

# Build static library
libradish.a: $(LIB_OBJ)
	ar rcs libradish.a $(LIB_OBJ)

# Compile rule
$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# Ensure build dir exists
$(BUILD):
	mkdir -p $(BUILD)

# Clean
clean:
	rm -rf $(BUILD) $(TARGET) libradish.a

# ASAN build
asan:
	$(CC) -fsanitize=address -fno-omit-frame-pointer $(SRC) -o $(TARGET)

.PHONY: all clean asan

