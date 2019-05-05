TARGET: netstore-server netstore-client

CC	= cc
CFLAGS	= -Wall -O2
LFLAGS	= -Wall

serwer.o klient.o err.o: err.h util.h

netstore-server: serwer.o err.o
	$(CC) $(LFLAGS) $^ -o $@

netstore-client: klient.o err.o
	$(CC) $(LFLAGS) $^ -o $@

.PHONY: clean TARGET
clean:
	rm -f netstore-server netstore-client *.o *~ *.bak
