#ifndef _UTIL_H_
#define _UTIL_H_

#define ERR_FILE_NAME     1
#define ERR_START_ADDRESS 2
#define ERR_ZERO_LEN      3

#define REQ_FILE_LIST   1
#define REQ_FRAGMENT    2
#define RESP_ERR        2
#define RESP_FILE_LIST  1
#define RESP_FRAGMENT   3

#define BUFFER_SIZE     524288

uint32_t min(uint32_t x, uint32_t y) {
  return (x > y) ? y : x;
}

int write_buffer(int fd, char *buffer, uint32_t bytes_to_write) {
  char *data = buffer;
  uint32_t left = bytes_to_write, wc;
  while (left > 0) {
    if ((wc = write(fd, data, left)) < 0)
      return 0;
    data += wc;
    left -= wc;
  }
  return 1;
}

int read_buffer(int fd, char *buffer, uint32_t bytes_to_read) {
  char *data = buffer;
  uint32_t left = bytes_to_read, rc;

  while (left > 0) {
    if ((rc = read(fd, data, left)) <= 0)
      return rc;
    data += rc;
    left -= rc;
  }
  return 1;
}

int write_long(int fd, uint32_t num) {
  uint32_t conv = htonl(num);
  char *data = (char*)&conv;
  return write_buffer(fd, data, sizeof(conv));
}

int write_short(int fd, uint16_t num) {
  uint16_t conv = htons(num);
  char *data = (char*)&conv;
  return write_buffer(fd, data, sizeof(conv));
}

int read_long(int fd, uint32_t* value) {
  uint32_t ret;
  char *data = (char*)&ret;
  int res;
  if ((res = read_buffer(fd, data, sizeof(ret))) <= 0)
    return res;
  *value = ntohl(ret);
  return 1;
}

int read_short(int fd, uint16_t* value) {
  uint16_t ret;
  char *data = (char*)&ret;
  int res;
  if ((res = read_buffer(fd, data, sizeof(ret))) <= 0)
    return res;
  *value = ntohs(ret);
  return 1;
}

#endif
