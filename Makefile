CC ?= gcc
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -fPIC -Iinclude
LDFLAGS = -lopenblas -lm

SRCDIR = src
INCDIR = include
BUILDDIR = build

LIB_NAME = libpicoml.so
TEST_BIN = test_picoml

LIB_SRCS = $(SRCDIR)/matrix.c $(SRCDIR)/ops.c
LIB_OBJS = $(BUILDDIR)/matrix.o $(BUILDDIR)/ops.o
MAIN_OBJ = $(BUILDDIR)/main.o

.PHONY: all clean

all: $(LIB_NAME) $(TEST_BIN)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_NAME): $(LIB_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

$(TEST_BIN): $(MAIN_OBJ) $(LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR) $(LIB_NAME) $(TEST_BIN)
