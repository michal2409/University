CC=gcc
CFLAGS=-c -Wall -Wextra -std=c11 -O2

all: main

main: main.o list.o tree.o
	$(CC) main.o list.o tree.o -o main

main.o: main.c tree.h
	$(CC) $(CFLAGS) main.c

tree.o: tree.c tree.h list.h
	$(CC) $(CFLAGS) tree.c

list.o: list.c list.h
	$(CC) $(CFLAGS) list.c

clean:
	rm -rf *.o main
