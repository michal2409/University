#include <boost/algorithm/string.hpp>

#include "err.h"
#include "util.h"

#define TTL_VALUE 4

std::string remote_dotted_address;
std::string out_dir;
std::set<int> open_sock;
int remote_port;
int timeout;

std::multimap<std::string, std::string, decltype(comp)> filenames(comp);

int sock, flags=0;
struct sockaddr_in local_address, remote_address;
unsigned int remote_len = sizeof(remote_address);
char buffer[MAX_PACKET_SIZE];

void parse_cmd_line_args(int argc, char **argv) {
  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("g,g", po::value<string>()->required(), "MCAST_ADDR")
      ("p,p", po::value<int>()->required(), "CMD_PORT")
      ("o,o", po::value<string>()->required(), "OUT_FLDR")
      ("t,t", po::value<int>()->default_value(5), "TIMEOUT")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    remote_dotted_address = vm["g"].as<string>();
    remote_port = vm["p"].as<int>();
    if (remote_port < 0 || remote_port >= 1<<16)
      throw std::runtime_error("CMD_PORT out of range");
    out_dir = vm["o"].as<string>();
    timeout = vm["t"].as<int>();
    if (timeout > 300)
      throw std::runtime_error("TIMEOUT greater than 300");
    if (timeout <= 0)
      throw std::runtime_error("TIMEOUT less than or equal 0");
    if (!fs::exists(fs::path(out_dir)))
      throw std::runtime_error("OUT_FLDR doesn't exists.");
  }
  catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    exit(1);
  }
}

int send_simpl(struct sockaddr_in addr, char* cmd, uint64_t cmd_seq, char* data, int data_len) {
  return send_msg(sock, buffer, addr, cmd, cmd_seq, 0, data, data_len, false);
}

int send_simpl_sock(int sockt, struct sockaddr_in addr, char* cmd, uint64_t cmd_seq, char* data, int data_len) {
  return send_msg(sockt, buffer, addr, cmd, cmd_seq, 0, data, data_len, false);
}

int send_cmplx(struct sockaddr_in addr, char* cmd, uint64_t cmd_seq, uint64_t param, char* data, int data_len) {
  return send_msg(sock, buffer, addr, cmd, cmd_seq, param, data, data_len, true);
}

int send_cmplx_sock(int sockt, struct sockaddr_in addr, char* cmd, uint64_t cmd_seq, uint64_t param, char* data, int data_len) {
  return send_msg(sockt, buffer, addr, cmd, cmd_seq, param, data, data_len, true);
}

void setup_addr() {
  local_address.sin_family = AF_INET;
  local_address.sin_addr.s_addr = htonl(INADDR_ANY);
  local_address.sin_port = htons(0);

  remote_address.sin_family = AF_INET;
  remote_address.sin_port = htons(remote_port);
  if (inet_aton((char*)remote_dotted_address.c_str(), &remote_address.sin_addr) == 0)
    syserr("inet_aton");
}

void set_timeout(int sockt, int t) {
  struct timeval tv;
  tv.tv_sec = t;
  tv.tv_usec = 0;
  setsockopt(sockt, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv, sizeof(struct timeval));
}

int setup_udp_sock() {
  int optval, reuse;
  int sockt = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockt < 0)
    syserr("socket");
  reuse = 1;
  if (setsockopt(sockt, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
    syserr("setsockopt(SO_REUSEADDR) failed");
  optval = 1;
  if (setsockopt(sockt, SOL_SOCKET, SO_BROADCAST, (void*)&optval, sizeof(optval)) < 0)
    syserr("setsockopt broadcast");
  optval = TTL_VALUE;
  if (setsockopt(sockt, IPPROTO_IP, IP_MULTICAST_TTL, (void*)&optval, sizeof(optval)) < 0)
    syserr("setsockopt multicast ttl");
  if (bind(sockt, (struct sockaddr *)&local_address, sizeof(local_address)) < 0)
   syserr("bind");
  set_timeout(sockt, timeout);
  return sockt;
}

bool is_correct_response(char *buffer, uint64_t cmd_seq, char *expected_cmd, struct sockaddr_in serv_addr) {
  if (cmd_seq != get_cmd_seq(buffer)) {
    print_err_msg(serv_addr, (char*)"Server response cmd_seq mismatch.");
    return false;
  }
  char cmd[CMD_SIZE+1]; cmd[CMD_SIZE] = '\0';
  memcpy(cmd, buffer, CMD_SIZE);
  if (strcmp(cmd, expected_cmd) != 0) {
    print_err_msg(serv_addr, (char*)"Server response cmd mismatch.");
    return false;
  }
  return true;
}

void op_discover(int sockt, char *buf, uint64_t cmd_seq, bool upd_map,
                 std::multimap<uint64_t, std::string, std::greater<uint64_t>> &space_ip_map) {
  struct sockaddr_in serv_addr;
  unsigned int serv_addr_len = sizeof(serv_addr);
  int msg_size;

  if (send_simpl_sock(sockt, remote_address, (char*)"HELLO", cmd_seq, (char*)"", 0) < 0) {
    printf("Error while sending HELLO packet...Skipping\n");
    return;
  }

  time_t t_start = time(NULL), t_end;
  while ((t_end = time(NULL)) < t_start + timeout) {
    set_timeout(sockt, timeout-(t_end-t_start));
    while ((msg_size = recvfrom(sockt, buf, MAX_PACKET_SIZE, 0, (struct sockaddr*) &serv_addr, &serv_addr_len)) > 0) {
      if (!is_correct_response(buf, cmd_seq, (char*)"GOOD_DAY", serv_addr))
        break;

      uint64_t dspace = get_param(buf);
      char data[MAX_DATA_SIMPL+1];
      get_data(data, msg_size, buf+CMD_SIZE+2*UINT_64_SIZE, true);
      if (!upd_map)
        printf("Found %s (%s) with free space %lu\n", inet_ntoa(serv_addr.sin_addr), data, dspace);
      else {
        std::string ip(inet_ntoa(serv_addr.sin_addr));
        space_ip_map.insert(std::pair<uint64_t, std::string>(dspace, ip));
      }
    }
  }
  set_timeout(sockt, timeout);
}

void parse_filenames(char *data, std::string &ip) {
  char *start_ptr = data;

  for (char *ptr = data; *ptr != '\0'; ++ptr) {
    if (*ptr == '\n') {
      *ptr = '\0';
      std::string fname(start_ptr);
      filenames.insert(std::pair<std::string, std::string>(fname, ip));
      printf("%s (%s)\n", start_ptr, ip.c_str());
      start_ptr = ++ptr;
    }
  }
  std::string fname(start_ptr);
  filenames.insert(std::pair<std::string, std::string>(fname, ip));
  printf("%s (%s)\n", start_ptr, (char*)ip.c_str());
}

void op_search(uint64_t cmd_seq, char *pattern) {
  int msg_size;
  struct sockaddr_in serv_addr;
  unsigned int serv_addr_len = sizeof(serv_addr);

  if (send_simpl(remote_address, (char*)"LIST", cmd_seq, pattern, strlen(pattern)) < 0) {
    printf("Error while sending LIST packet...Skipping\n");
    return;
  }

  filenames.clear();
  time_t t_start = time(NULL), t_end;
  while ((t_end = time(NULL)) < t_start + timeout) {
    set_timeout(sock, timeout-(t_end-t_start));

    while ((msg_size=recvfrom(sock, buffer, sizeof(buffer), 0, (struct sockaddr*)&serv_addr, &serv_addr_len)) > 0) {
      if (!is_correct_response(buffer, cmd_seq, (char*)"MY_LIST", serv_addr))
        break;
      char data[MAX_DATA_SIMPL+1];
      get_data(data, msg_size, buffer+CMD_SIZE+UINT_64_SIZE, false);
      std::string ip(inet_ntoa(serv_addr.sin_addr));
      parse_filenames(data, ip);
    }
  }
  set_timeout(sock, timeout);
}

void op_remove(uint64_t cmd_seq, char *fname) {
  if (fname != NULL && fname[0] != '\0')
    send_simpl(remote_address, (char*)"DEL", cmd_seq, fname, strlen(fname));
  else
    printf("Remove error. File not specified.\n");
}

int connect_tcp(struct sockaddr_in addr) {
  int tcp_sock;
  if ((tcp_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    return -1;
  open_sock.insert(tcp_sock);
  if (connect(tcp_sock, (sockaddr *)&addr, sizeof(addr)) < 0)
    return -1;
  return tcp_sock;
}

struct sockaddr_in create_sockaddr(std::string ip, in_port_t port) {
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_aton(ip.c_str(), &addr.sin_addr) == 0)
    syserr("inet_aton");
  return addr;
}

void dwnld(fs::path file_path, struct sockaddr_in addr) {
  int tcp_sock;
  if ((tcp_sock = connect_tcp(addr)) < 0) {
    printf("TCP conection error...%s\n", strerror(errno));
    return;
  }
  std::string msg;
  if (download_file(file_path, tcp_sock, msg))
    printf("File %s downloaded (%s:%d)\n", file_path.filename().string().c_str(), inet_ntoa(addr.sin_addr), ntohs(addr.sin_port));
  else
    printf("File %s downloading failed (%s:%d) %s\n", file_path.filename().string().c_str(), inet_ntoa(addr.sin_addr), ntohs(addr.sin_port), msg.c_str());
  close(tcp_sock);
  open_sock.erase(tcp_sock);
}

void sendf(fs::path file_path, struct sockaddr_in addr) {
  int tcp_sock;
  if ((tcp_sock = connect_tcp(addr)) < 0) {
    printf("TCP conection error...%s\n", strerror(errno));
    return;
  }
  std::string msg;
  if (send_file(file_path, tcp_sock, msg))
    printf("File %s uploaded (%s:%d)\n", file_path.filename().string().c_str(), inet_ntoa(addr.sin_addr), ntohs(addr.sin_port));
  else
    printf("File %s uploading failed (%s:%d) %s\n", file_path.filename().string().c_str(), inet_ntoa(addr.sin_addr), ntohs(addr.sin_port), msg.c_str());
  shutdown(tcp_sock, SHUT_WR);
  char buf[10];
  while (read(tcp_sock, buf, sizeof(buf)) > 0);
  close(tcp_sock);
  open_sock.erase(tcp_sock);
}

void op_fetch(uint64_t cmd_seq, char *fname) {
  std::string fname_string(fname);
  auto it = filenames.find(fname_string);

  if (it == filenames.end()) {
    printf("File %s not found in available file list... Skipping fetch\n", fname);
    return;
  }

  int msg_size;
  struct sockaddr_in serv_addr = create_sockaddr(it->second, remote_port);
  unsigned int serv_addr_len = sizeof(serv_addr);

  send_simpl(serv_addr, (char*)"GET", cmd_seq, fname, strlen(fname));
  if ((msg_size = recvfrom(sock, buffer, sizeof(buffer), 0, (struct sockaddr*)&serv_addr, &serv_addr_len)) < 0) {
    printf("File {%s} downloading failed (:) %s", fname, "server hasn't responeded.");
    return;
  }

  if (!is_correct_response(buffer, cmd_seq, (char*)"CONNECT_ME", serv_addr))
    return;
  serv_addr.sin_port = htons(get_param(buffer));
  fs::path file_path = fs::absolute(fname, out_dir);
  std::thread(dwnld, file_path, serv_addr).detach();
}

void op_upload(uint64_t cmd_seq, std::string path) {
  fs::path file_path = fs::path(path);
  std::string fname = file_path.filename().string();
  if (!fs::exists(file_path)) {
    printf("File %s does not exists\n", fname.c_str());
    return;
  }

  bool can_add = false;
  char buf[MAX_PACKET_SIZE];
  struct sockaddr_in serv_addr;
  unsigned int serv_addr_len = sizeof(serv_addr);
  std::multimap<uint64_t, std::string, std::greater<uint64_t>> space_ip_map;

  int new_sock = setup_udp_sock();
  open_sock.insert(new_sock);

  op_discover(new_sock, buf, cmd_seq, true, space_ip_map);
  int msg_size;
  uint64_t file_size = fs::file_size(file_path);
  for (auto it=space_ip_map.begin(); it!=space_ip_map.end(); ++it) {
    serv_addr = create_sockaddr(it->second, remote_port);
    send_cmplx_sock(new_sock, serv_addr, (char*)"ADD", cmd_seq, file_size, (char*)fname.c_str(), fname.length());
    if ((msg_size = recvfrom(new_sock, buf, sizeof(buf), 0, (struct sockaddr*)&serv_addr, &serv_addr_len)) < 0)
      continue;
    char cmd[CMD_SIZE+1];
    get_cmd(buf, cmd);
    if (!strcmp(cmd, "NO_WAY")) {
      is_correct_response(buf, cmd_seq, (char*)"NO_WAY", serv_addr);
      continue;
    }
    if (!is_correct_response(buf, cmd_seq, (char*)"CAN_ADD", serv_addr))
      continue;
    if (!is_data_empty(buf, msg_size, true, serv_addr))
      continue;
    can_add = true;
    serv_addr.sin_port = htons(get_param(buf));
    break;
  }
  open_sock.erase(new_sock);
  if (!can_add) {
    printf("File %s too big \n", fname.c_str());
    return;
  }
  sendf(file_path, serv_addr);
}

void close_sockets() {
  for (auto it = open_sock.begin(); it != open_sock.end(); ++it)
    close(*it);
  open_sock.clear();
  close(sock);
}

void exit_handler(int s) {
  (void)s;
  close_sockets();
  exit(1);
}

void op_exit() {
  close_sockets();
  exit(EXIT_SUCCESS);
}

char *msg;
uint64_t cmd_seq = 0;

std::map<int, std::function<void()>> disp_table {
  {ERR_NUM,   []() { printf("CMD not supported\n");}},
  {HELLO_NUM, []() { std::multimap<uint64_t, std::string, std::greater<uint64_t>> smap;
                                             op_discover(sock, buffer, cmd_seq, false, smap);}},
  {LIST_NUM,  []() { op_search(cmd_seq, msg);}},
  {DEL_NUM,   []() { op_remove(cmd_seq, msg);}},
  {GET_NUM,   []() { op_fetch(cmd_seq, msg);}},
  {ADD_NUM,   []() { std::string path(msg);
                     std::thread(op_upload, cmd_seq, path).detach();}},
  {EXIT_NUM,  []() { op_exit();}}
};

std::map<std::string, int, decltype(comp)> cmd_to_int(comp);
void create_map() {
  std::vector<std::string> v1 = {"upload", "fetch", "remove", "search", "discover", "exit"};
  std::vector<int> v2 = {ADD_NUM, GET_NUM, DEL_NUM, LIST_NUM, HELLO_NUM, EXIT_NUM};
  for (int i = 0; i < (int)v1.size(); ++i)
    cmd_to_int[v1[i]] = v2[i];
}

int main (int argc, char **argv) {
  parse_cmd_line_args(argc, argv);
  setup_addr();
  sock = setup_udp_sock();
  create_map();
  signal(SIGINT, exit_handler);

  while(1) {
    std::string input, cmd, arg("");
    getline(std::cin, input);
    std::size_t first_space = input.find_first_of(' ');
    if (first_space != std::string::npos)
      arg = input.substr(first_space + 1);
    cmd = input.substr(0, first_space);

    boost::algorithm::to_lower(cmd);
    msg = (char*)arg.c_str();
    disp_table[cmd_to_int[cmd]]();
    ++cmd_seq;
  }
}
