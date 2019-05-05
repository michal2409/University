#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>

#include "err.h"
#include "util.h"

char *files = NULL;
char buffer[BUFFER_SIZE];
uint16_t files_num = 0;

void read_file_list(int sock, uint32_t files_length) {
  files = malloc((files_length + 1) * sizeof(char));
  uint32_t idx = 0;
  while (files_length > 0) {
    uint32_t bytes_read = min(files_length, BUFFER_SIZE);
    if (read_buffer(sock, buffer, bytes_read) <= 0)
      syserr("read_buffer error");
    for (int i = 0; i < bytes_read; ++i, ++idx) {
      if (buffer[i] == '|') {
        files[idx] = '\0';
        ++files_num;
      } else {
        files[idx] = buffer[i];
      }
    }
    files_length -= bytes_read;
  }
  files[idx] = '\0';
  if (idx > 0)
    ++files_num;
}

void print_list() {
  char *buf = files;
  for (int i = 0; i < files_num; ++i) {
    int len = strlen(buf);
    printf("%d.%s\n", i+1, buf);
    buf += len+1;
  }
}

char *get_file_name(int file_idx) {
  int i = 1, len;
  char *buf = files;
  while (i < file_idx) {
    len = strlen(buf);
    buf += len+1;
    ++i;
  }
  return buf;
}

uint32_t read_int_stdin() {
  int line_max = 12;
  char line[line_max];
  if (!fgets(line, line_max, stdin))
    syserr("fgets err");
  int n = strlen(line), i = 0;
  for (;i < n && isdigit(line[i]); ++i);
  if (i == 0 || line[i] != '\n')
    syserr("wrong intput");
  line[i] = '\0';
  uint64_t val = strtoull(line, NULL, 10);
  if (val > UINT_MAX) // overflow
    syserr("wrong intput");
  return val;
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3)
    fatal("Usage: %s <server-name-or-IP4> [<port>]\n", argv[0]);

  int sock, err;
  struct addrinfo addr_hints, *addr_result;
  char *port = (argc > 2) ? argv[2] : "6543", *ip = argv[1];

  // 'converting' host/port in string to struct addrinfo
  memset(&addr_hints, 0, sizeof(struct addrinfo));
  addr_hints.ai_family = AF_INET; // IPv4
  addr_hints.ai_socktype = SOCK_STREAM; // TCP
  addr_hints.ai_protocol = IPPROTO_TCP;
  err = getaddrinfo(ip, port, &addr_hints, &addr_result);

  if (err == EAI_SYSTEM)  // system error
    syserr("getaddrinfo: %s", gai_strerror(err));
  if (err != 0)  // other error (host not found, etc.)
    fatal("getaddrinfo: %s", gai_strerror(err));
  // initialize socket according to getaddrinfo results
  if ((sock = socket(addr_result->ai_family, addr_result->ai_socktype, addr_result->ai_protocol)) < 0)
    syserr("socket");
  // connect socket to the server
  if (connect(sock, addr_result->ai_addr, addr_result->ai_addrlen) < 0)
    syserr("connect");

  freeaddrinfo(addr_result);

  // sending request for file list
  if (!write_short(sock, REQ_FILE_LIST))
    syserr("write_short error");
  uint16_t action_code;
  uint32_t files_length;

  // reciving server ack
  if (read_short(sock, &action_code) < 0)
    syserr("read_short error");
  if (action_code != RESP_FILE_LIST)
    close(sock);

  // receiving length of all files
  if (read_long(sock, &files_length) < 0)
    syserr("read_long error");
  // filling matrix of files and printing to user
  read_file_list(sock, files_length);
  print_list();

  // reading data for file fragment request
  printf("Podaj numer pliku\n");
  uint32_t file_idx = read_int_stdin();
  if (file_idx < 1 || file_idx > files_num)
    syserr("selected wrong file number");

  printf("Podaj adres poczatku fragmentu\n");
  uint32_t start = read_int_stdin();

  printf("Podaj adres konca fragmentu\n");
  uint32_t end = read_int_stdin();
  if (start > end)
    syserr("start > end");

  // sending request for fragment
  if (!write_short(sock, REQ_FRAGMENT))
    syserr("write_short error");
  if (!write_long(sock, start))
    syserr("write_long error");
  if (!write_long(sock, end - start))
    syserr("write_long error");
  char *file_name = get_file_name(file_idx);
  uint16_t name_len = strlen(file_name);
  if (!write_short(sock, name_len))
    syserr("write_short error");
  if (!write_buffer(sock, file_name, name_len))
    syserr("write_buffer error");
  // receiving server response
  if (read_short(sock, &action_code) < 0)
    syserr("read_short error");

  switch (action_code) {
    case RESP_ERR: {
      uint32_t err_code;
      if (read_long(sock, &err_code) < 0)
        syserr("read_long error");
      switch (err_code) {
        case ERR_FILE_NAME:
          printf("Denial: wrong file name\n");
          break;
        case ERR_START_ADDRESS:
          printf("Denial: starting address greater than file size - 1\n");
          break;
        case ERR_ZERO_LEN:
          printf("Denial: zero length fragment\n");
          break;
      }
      break;
    }
    case RESP_FRAGMENT: {
      uint32_t frag_len;
      if (read_long(sock, &frag_len) < 0)
        syserr("read_long error");

      struct stat st = {0};
      if (stat("./tmp", &st) == -1)
        if (mkdir("tmp", S_IRWXU) < 0)
          syserr("Unable to create directory");

      char dir[300] = "./tmp/";
      strcat(dir, file_name);
      int fd = open(dir, O_RDWR | O_CREAT, S_IWRITE | S_IREAD);
      if (fd < 0)
        syserr("open file error");
      if (lseek(fd, start, SEEK_SET) < 0)
        syserr("lseek error");

      while (frag_len > 0) {
        uint32_t len = min(frag_len, BUFFER_SIZE);
        if (read_buffer(sock, buffer, len) <= 0)
          syserr("read_buffer error");
        if (write_buffer(fd, buffer, len) < 0)
          syserr("write_buffer error");
        frag_len -= len;
      }
      if (close(fd) < 0)
        syserr("close");
      break;
    }
  }

  if (close(sock) < 0)
    syserr("close");
  return 0;
}
