#ifndef _UTIL_H_
#define _UTIL_H_

#include <endian.h>
#include <string>
#include <cstring>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <set>
#include <map>
#include <string>
#include <errno.h>
#include <iostream>
#include <stdlib.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <functional>
#include <sys/stat.h>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <fcntl.h>
#include <signal.h>

#define MAX_PACKET_SIZE 65507
#define MAX_DATA_SIMPL 65489
#define MAX_DATA_CMPLX 65481
#define TCP_BUFFER_SIZE  524288
#define CMD_SIZE 10
#define UINT_64_SIZE 8
#define QUEUE_LENGTH 5

#define ERR_NUM 0
#define HELLO_NUM 1
#define LIST_NUM 2
#define DEL_NUM 3
#define GET_NUM 4
#define ADD_NUM 5
#define EXIT_NUM 6

namespace po = boost::program_options;
namespace fs = boost::filesystem;

auto comp = [](const std::string& a, const std::string& b) { return a.compare(b) < 0; };

void serialize_string(char *str, char **ptr, int len) {
  memcpy(*ptr, str, len);
  *ptr += len;
}

void deserialize_string(char *str, char **ptr, int len) {
  memcpy(str, *ptr, len);
  *ptr += len;
}

void serialize_uint64(uint64_t t, char **ptr) {
  uint64_t net_t = htobe64(t);
  memcpy(*ptr, &net_t, sizeof(net_t));
  *ptr += sizeof(net_t);
}

void deserialize_uint64(uint64_t *t, char **ptr) {
  // *t = (uint64_t *)(*ptr);
  char *data = (char*)t;
  memcpy(data, *ptr, sizeof(*t));
  *ptr += sizeof(*t);
  *t = be64toh(*t);
}

int create_msg(char* buffer, char* cmd, uint64_t cmd_seq, uint64_t param, char *data, int data_len, bool is_cmplx) {
  char *ptr = buffer;
  char cmd_arr[CMD_SIZE+1];

  // Converting cmd to char[CMD_SIZE]
  memset(cmd_arr, 0, CMD_SIZE+1);
  strcpy(cmd_arr, cmd);

  // Serializing data to buffer
  serialize_string(cmd_arr, &ptr, CMD_SIZE);
  serialize_uint64(cmd_seq, &ptr);
  if (is_cmplx)
    serialize_uint64(param, &ptr);
  serialize_string(data, &ptr, data_len);
  // Returing number of wrote bytes
  return ptr - buffer;
}

int write_buffer(int fd, char *buffer, int bytes_to_write) {
  char *data = buffer;
  ssize_t left = bytes_to_write, wc;
  while (left > 0) {
    if ((wc = write(fd, data, left)) < 0)
      return -1;
    data += wc;
    left -= wc;
  }
  return 1;
}

int read_buffer(int fd, char *buffer, int bytes_to_read) {
  char *data = buffer;
  ssize_t left = bytes_to_read, rc, read_bytes = 0;

  while (left > 0) {
    if ((rc = read(fd, data, left)) < 0)
      return rc;
    if (rc == 0)
      return read_bytes;
    data += rc;
    left -= rc;
    read_bytes += rc;
  }
  return read_bytes;
}

int get_data(char* data, int msg_size, char* msg, bool is_cmplx) {
  int data_len = msg_size - CMD_SIZE - UINT_64_SIZE;
  if (is_cmplx)  // CMPLX msg has one uint64_t more
    data_len -= UINT_64_SIZE;
  deserialize_string(data, &msg, data_len);
  data[data_len] = '\0';
  return data_len;
}

uint64_t get_cmd_seq(char* buffer) {
  char *ptr = buffer+CMD_SIZE;
  uint64_t cmd_seq;
  deserialize_uint64(&cmd_seq, &ptr);
  return cmd_seq;
}

uint64_t get_param(char* buffer) {
  char *ptr = buffer+CMD_SIZE+UINT_64_SIZE;
  uint64_t param;
  deserialize_uint64(&param, &ptr);
  return param;
}

void get_cmd(char *buffer, char* cmd) {
  memset(cmd, 0, CMD_SIZE+1);
  memcpy(cmd, buffer, CMD_SIZE);
}

bool download_file(fs::path file_path, int msg_sock, std::string &msg) {
  bool ret_val = false;
  int write_status;
  while (1) {
    int fd = open(file_path.string().c_str(), O_RDWR | O_CREAT, S_IWRITE | S_IREAD), read_bytes;
    if (fd == -1) {
      msg = "open fail";
      break;
    }

    char tcp_buffer[TCP_BUFFER_SIZE];
    while ((read_bytes = read_buffer(msg_sock, tcp_buffer, TCP_BUFFER_SIZE)) > 0)
      if ((write_status = write_buffer(fd, tcp_buffer, read_bytes)) < 0)
        break;
    close(fd);
    if (read_bytes < 0) {
      msg = "read fail";
      break;
    }
    if (write_status < 0) {
      msg = "write fail";
      break;
    }
    ret_val = true;
    break;
  }
  if (!ret_val)
    fs::remove(file_path);
  return ret_val;
}

bool send_file(fs::path file_path, int msg_sock, std::string &msg) {
  bool ret_val = false;
  int write_status;
  while (1) {
  	FILE *fp;
    if (!(fp = fopen(file_path.string().c_str(), "r"))) {
      msg = "fopen fail";
      break;
    }
    int read_bytes;
    char tcp_buffer[TCP_BUFFER_SIZE];
    while ((read_bytes = fread(tcp_buffer, sizeof(char), TCP_BUFFER_SIZE, fp)) > 0)
      if ((write_status = write_buffer(msg_sock, tcp_buffer, read_bytes)) < 0)
        break;
    fclose(fp);
    if (read_bytes < 0) {
      msg = "read fail";
      break;
    }
    if (write_status < 0) {
      msg = "write fail";
      break;
    }
    ret_val = true;
    break;
  }
  return ret_val;
}

void print_err_msg(struct sockaddr_in addr, char* msg) {
  fprintf(stderr, "[PCKG ERROR] Skipping invalid package from {%s}:{%d}. %s",
          inet_ntoa(addr.sin_addr), ntohs(addr.sin_port), msg);
}

int send_msg(int sock, char *buffer, struct sockaddr_in addr, char* cmd,
             uint64_t cmd_seq, uint64_t param, char* data, int data_len, bool is_cmplx) {
  int size = create_msg(buffer, cmd, cmd_seq, param, data, data_len, is_cmplx);
  return sendto(sock, buffer, size, 0, (struct sockaddr*)&addr, sizeof(addr));
}

bool is_data_empty(char* buffer, int msg_size, bool is_cmplx, struct sockaddr_in addr) {
  char data[MAX_DATA_SIMPL+1];
  char *ptr = buffer+CMD_SIZE+UINT_64_SIZE;
  if (is_cmplx)
    ptr += UINT_64_SIZE;
  int data_len = get_data(data, msg_size, ptr, is_cmplx);
  if (data_len != 0) {
    print_err_msg(addr, (char*)"Data field not empty.");
    return false;
  }
  return true;
}

#endif