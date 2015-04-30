LIBOPTS = -shared
CFLAGS = -fPIC
CC = gcc

symtorch/libsymtorch.o : symtorch/symtorch.c
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	rm symtorch/libsymtorch.o