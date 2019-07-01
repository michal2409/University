TARGETS = netstore-server netstore-client

CC = g++
CC_FLAGS = -Wall -Wextra -O2 -c -g -std=c++14
LF_FLAGS = -Wall -g
LDLIBS += -lboost_program_options -lboost_filesystem -lboost_system -pthread

all: $(TARGETS)

err.o: err.cpp
	$(CC) $(CC_FLAGS) $^ -o $@

netstore-server.o: netstore-server.cpp
	$(CC) $(CC_FLAGS) $^ -o $@

netstore-server: err.o netstore-server.o
	$(CC) $(LF_FLAGS) $^ $(LDLIBS) -o $@

netstore-client.o: netstore-client.cpp
	$(CC) $(CC_FLAGS) $^ -o $@

netstore-client: err.o netstore-client.o
	$(CC) $(LF_FLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean

clean:
	rm -f $(TARGETS) *.o *~ *.bak