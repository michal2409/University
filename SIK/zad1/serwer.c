#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <dirent.h>
#include <stdint.h>
#include <sys/stat.h>

#include "err.h"
#include "util.h"

#define FILES_LEN_MAX    16842752
#define QUEUE_LENGTH     5
#define DEFAULT_PORT     6543

char files[FILES_LEN_MAX];
char buffer[BUFFER_SIZE];
char *directory;
int dir_len;

int send_file_list(int sock) {
  DIR *d;
  struct dirent *dir;

  if (!(d = opendir(directory)))
    return 0;

  int res = 0;
  while(1) {
    // filling array with file names separated by |
    uint32_t files_len = 0;
    while ((dir = readdir(d)) != NULL) {
      struct stat file_info;
      char path[dir_len + strlen(dir->d_name) + 2];
      strcpy(path, directory);
      path[dir_len] = '/';
      strcat(path, dir->d_name);
      lstat(path, &file_info);
      if (S_ISREG(file_info.st_mode)) {
        int n = strlen(dir->d_name);
        strcpy(files + files_len, dir->d_name);
        files[files_len + n] = '|';
        files_len += n+1;
      }
    }

    if (files_len > 0)
    	files_len--; // -1 because of not sending last |
    if (!write_short(sock, RESP_FILE_LIST))
      break;
    if (!write_long(sock, files_len))
      break;
    uint32_t wrote_bytes = 0;
    while (wrote_bytes < files_len) {
      uint32_t bytes_to_write = min(BUFFER_SIZE, files_len - wrote_bytes);
      if (!write_buffer(sock, files + wrote_bytes, bytes_to_write))
        break;
      wrote_bytes += bytes_to_write;
    }
    if (wrote_bytes >= files_len)
      res = 1;
    break;
  }
  closedir(d);
  return res;
}

int write_err(int sock, int err) {
  if (!write_short(sock, RESP_ERR))
    return 0;
  if (!write_long(sock, err))
    return 0;
  return 1;
}

int process_file(int sock) {
  uint32_t start, bytes_to_read;
  uint16_t name_len;

  if (read_long(sock, &start) < 0)
    return 0;
  if (read_long(sock, &bytes_to_read) < 0)
    return 0;
  if (read_short(sock, &name_len) < 0)
    return 0;

  char *path = buffer;
  strncpy(path, directory, dir_len);
  path[dir_len] = '/';
  path += dir_len + 1;

  if (read_buffer(sock, path, name_len) < 0)
    return 0;

  path[name_len] = '\0';

  if (bytes_to_read == 0) { // zero fragment len err
    write_err(sock, ERR_ZERO_LEN);
    return 0;
  }

  FILE* fp;
  if (!(fp = fopen(buffer, "r"))) {
    write_err(sock, ERR_FILE_NAME);
    return 0;
  }

  int res = 0;
  while(1) {
    if (fseek(fp, 0, SEEK_END) < 0)
      break;
    uint32_t file_size = ftell(fp);
    if (start > file_size - 1) {  // wrong starting addres
      write_err(sock, ERR_START_ADDRESS);
      break;
    }

    if (fseek(fp, start, SEEK_SET) < 0)
      break;

    if (file_size - start < bytes_to_read)
      bytes_to_read = file_size - start;

    if (!write_short(sock, RESP_FRAGMENT))
      break;
    if (!write_long(sock, bytes_to_read))
      break;

    while (bytes_to_read > 0) {
      uint32_t to_write = min(BUFFER_SIZE, bytes_to_read);
      uint32_t read_bytes = fread(buffer, sizeof(char), to_write, fp);
      if (!write_buffer(sock, buffer, read_bytes))
        break;
      bytes_to_read -= read_bytes;
    }

    if (bytes_to_read == 0)
      res = 1;
    break;
  }
  fclose(fp);
  return res;
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3)
    fatal("Usage: %s <directory> [port]\n", argv[0]);

  int sock, msg_sock, port = (argc > 2) ? atoi(argv[2]) : DEFAULT_PORT;
  struct sockaddr_in server_address, client_address;
  socklen_t client_address_len;
  directory = argv[1];
  dir_len = strlen(directory);

  // create IPv4 TCP socket
  if ((sock = socket(PF_INET, SOCK_STREAM, 0)) < 0)
    syserr("socket");

  server_address.sin_family = AF_INET;
  server_address.sin_addr.s_addr = htonl(INADDR_ANY);
  server_address.sin_port = htons(port);

  // bind the socket to a concrete address
  if (bind(sock, (struct sockaddr *) &server_address, sizeof(server_address)) < 0)
    syserr("bind");

  // switch to listening (passive open)
  if (listen(sock, QUEUE_LENGTH) < 0)
    syserr("listen");

  client_address_len = sizeof(client_address);

  uint16_t action_code;
  int res = 1;
  while(1) {
    // get client connection from the socket
    if ((msg_sock = accept(sock, (struct sockaddr *) &client_address, &client_address_len)) < 0)
      syserr("accept");

    while (1) {
      if(read_short(msg_sock, &action_code) <= 0)
        break;

      if (!res)
        continue;

      switch (action_code) {
        case REQ_FILE_LIST:
          res = send_file_list(msg_sock);
          break;
        case REQ_FRAGMENT:
          res = process_file(msg_sock);
          break;
      }
    }

    if (close(msg_sock) < 0)
      syserr("close");
  }

  return 0;
}
