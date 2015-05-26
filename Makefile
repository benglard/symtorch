LIBOPTS = -shared
CFLAGS = -fPIC -std=gnu99 -O3 -Wall -Werror -funroll-loops -ffast-math
CC = gcc

symtorch/libsymtorch.so : symtorch/symtorch.c
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	rm symtorch/libsymtorch.so