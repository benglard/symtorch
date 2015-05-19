LIBOPTS = -shared
CFLAGS = -fPIC -std=gnu99 -O3
CC = gcc

symtorch/libsymtorch.so : symtorch/symtorch.c
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	rm symtorch/libsymtorch.so