#include "err.h"
#include "util.h"

std::set<string> filenames;
std::set<int> open_sock;
std::string multicast_dotted_address, directory;
char buffer[MAX_PACKET_SIZE];
uint64_t disk_space;
int timeout, sock, local_port, flags = 0;
struct sockaddr_in local_address, client_address;
unsigned int client_len = sizeof(client_address);
struct ip_mreq ip_mreq;

void read_filenames() {
  fs::path p(directory);
  if (!fs::exists(p) || !fs::is_directory(p))
    return;
  for (fs::directory_entry& f : fs::directory_iterator(p)) {
    if (!fs::is_regular_file(f.path()) || disk_space < fs::file_size(f.path()))
      continue;
    filenames.insert(f.path().filename().string());
    disk_space -= fs::file_size(f.path());
  }
}

void parse_cmd_line_args(int argc, char **argv) {
  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("g,g", po::value<string>()->required(), "MCAST_ADDR")
      ("p,p", po::value<int>()->required(), "CMD_PORT")
      ("f,f", po::value<string>()->required(), "SHRD_FLDR")
      ("b,b", po::value<uint64_t>()->default_value(52428800), "MAX_SPACE")
      ("t,t", po::value<int>()->default_value(5), "TIMEOUT")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    multicast_dotted_address = vm["g"].as<string>();
    local_port = vm["p"].as<int>();
    if (local_port < 0 || local_port >= 1<<16)
      throw std::runtime_error("CMD_PORT out of range");
    directory = vm["f"].as<string>();
    disk_space = vm["b"].as<uint64_t>();
    timeout = vm["t"].as<int>();
    if (timeout > 300)
      throw std::runtime_error("TIMEOUT greater than 300");
    if (timeout <= 0)
      throw std::runtime_error("TIMEOUT less than or equal 0");
    if (!fs::exists(fs::path(directory)))
      throw std::runtime_error("SHRD_FLDR doesn't exists.");
  }
  catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    exit(1);
  }
}

int send_simpl(char* cmd, uint64_t cmd_seq, char* data, int data_len) {
  return send_msg(sock, buffer, client_address, cmd, cmd_seq, 0, data, data_len, false);
}

int send_cmplx(char* cmd, uint64_t cmd_seq, uint64_t param, char* data, int data_len) {
  return send_msg(sock, buffer, client_address, cmd, cmd_seq, param, data, data_len, true);
}

void setup_udp_sock() {
  if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    syserr("socket");

  int reuse = 1;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
    syserr("setsockopt");

  ip_mreq.imr_interface.s_addr = htonl(INADDR_ANY);
  if (inet_aton((char*)multicast_dotted_address.c_str(), &ip_mreq.imr_multiaddr) == 0)
    syserr("inet_aton");
  if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (void*)&ip_mreq, sizeof(ip_mreq)) < 0)
    syserr("setsockopt");

  memset((char*)&local_address, 0, sizeof(local_address));
  local_address.sin_family = AF_INET;
  local_address.sin_addr.s_addr = htonl(INADDR_ANY);
  local_address.sin_port = htons(local_port);
  if (bind(sock, (struct sockaddr*)&local_address, sizeof(local_address)) < 0)
    syserr("bind");
}

int create_tcp_sock(uint64_t *tcp_port) {
  int tcp_sock;
  if ((tcp_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    return -1;

  struct sockaddr_in serv_tcp_adrr;
  socklen_t serv_tcp_adrr_len = sizeof(serv_tcp_adrr);

  memset((char *) &serv_tcp_adrr, 0, sizeof(serv_tcp_adrr));
  serv_tcp_adrr.sin_family = AF_INET;
  serv_tcp_adrr.sin_addr.s_addr = htonl(INADDR_ANY);
  serv_tcp_adrr.sin_port = htons(0);

  if ((bind(tcp_sock, (struct sockaddr *)&serv_tcp_adrr, serv_tcp_adrr_len)) < 0) {
    close(tcp_sock);
    return -1;
  }
  if (getsockname(tcp_sock, (struct sockaddr *)&serv_tcp_adrr, &serv_tcp_adrr_len) < 0) {
    close(tcp_sock);
    return -1;
  }
  if (listen(tcp_sock, QUEUE_LENGTH) < 0) {
    close(tcp_sock);
    return -1;
  }

  struct timeval tv;
  tv.tv_sec = timeout;
  tv.tv_usec = 0;
  setsockopt(tcp_sock, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv, sizeof(struct timeval));
  *tcp_port = (uint64_t)ntohs(serv_tcp_adrr.sin_port);
  return tcp_sock;
}

void answer_hello(uint64_t cmd_seq, uint64_t msg_size) {
  if (!is_data_empty(buffer, msg_size, false, client_address))
    return;
  send_cmplx((char*)"GOOD_DAY", cmd_seq, disk_space, (char*)multicast_dotted_address.c_str(), multicast_dotted_address.length());
}

void answer_list(uint64_t cmd_seq, int msg_size) {
  char data[MAX_DATA_SIMPL+1], fnames[MAX_DATA_SIMPL+1], *ptr = fnames;
  get_data(data, msg_size, buffer+CMD_SIZE+UINT_64_SIZE, false);
  int fnames_len = 0;

  for (auto f : filenames) {
    if (!strstr(f.c_str(), data))  // data pattern not matched with filename
      continue;

    int file_len = f.length();
    if (fnames_len + file_len > MAX_DATA_SIMPL) {  // Can't store next filename. Sending UDP packet.
      send_simpl((char*)"MY_LIST", cmd_seq, fnames, fnames_len-1);  // fnames_len-1 to trim last '\n'
      fnames_len = 0;
      ptr = fnames;
    }
    // Adding new filename ended with '\n'
    memcpy(ptr, f.c_str(), file_len);
    *(ptr+file_len) = '\n';
    fnames_len += file_len + 1;
    ptr += file_len + 1;
  }

  if (fnames_len > 0)
    send_simpl((char*)"MY_LIST", cmd_seq, fnames, fnames_len-1); // fnames_len-1 to trim last '\n'
}

void answer_del(int msg_size) {
  char data[MAX_DATA_SIMPL+1];
  get_data(data, msg_size, buffer+CMD_SIZE+UINT_64_SIZE, false);
  std::string s(data);
  filenames.erase(s);
  fs::path p = fs::absolute(data, directory);
  if (fs::exists(p)) {
    disk_space += fs::file_size(p);
    fs::remove(p);
  }
}

void sendf(int tcp_sock, fs::path file_path) {
  struct sockaddr_in cli_adrr;
  socklen_t cli_adrr_len = sizeof(cli_adrr);
  int msg_sock;
  if ((msg_sock = accept(tcp_sock, (struct sockaddr *)&cli_adrr, &cli_adrr_len)) < 0) {
    close(tcp_sock);
    return;
  }
  close(tcp_sock);
  std::string s;
  send_file(file_path, msg_sock, s);
  shutdown(msg_sock, SHUT_WR);
  char buf[10];
  while (read(msg_sock, buf, sizeof(buf)) > 0);
  close(msg_sock);
}

void dwnld(int tcp_sock, fs::path file_path, int file_size) {
  struct sockaddr_in cli_adrr;
  socklen_t cli_adrr_len = sizeof(cli_adrr);
  int msg_sock;
  if ((msg_sock = accept(tcp_sock, (struct sockaddr *)&cli_adrr, &cli_adrr_len)) < 0) {
    close(tcp_sock);
    return;
  }
  close(tcp_sock);
  std::string s;
  if (!download_file(file_path, msg_sock, s)) {
    filenames.erase(file_path.filename().string());
    disk_space += file_size;
  }
  close(msg_sock);
}

void answer_get(uint64_t cmd_seq, int msg_size) {
  char data[MAX_DATA_SIMPL+1];
  int data_len = get_data(data, msg_size, buffer+CMD_SIZE+UINT_64_SIZE, false);
  fs::path file_path = fs::absolute(data, directory);
  std::string fname_string(data);

  if (!fs::exists(file_path)) {
    print_err_msg(client_address, (char*)"File doesn't exists...skipping get.");
    return;
  }

  int tcp_sock;
  uint64_t tcp_port;
  if ((tcp_sock = create_tcp_sock(&tcp_port)) < 0) {
    printf("TCP connection error...%s\n", strerror(errno));
    return;
  }

  send_cmplx((char*)"CONNECT_ME", cmd_seq, tcp_port, data, data_len);
  std::thread(sendf, tcp_sock, file_path).detach();
}

void answer_add(uint64_t cmd_seq, int msg_size) {
  char data[MAX_DATA_CMPLX+1];
  int data_len = get_data(data, msg_size, buffer+CMD_SIZE+2*UINT_64_SIZE, true);
  uint64_t file_size = get_param(buffer);

  if (file_size > disk_space) {
    send_simpl((char*)"NO_WAY", cmd_seq, data, data_len);
    return;
  }

  std::string fname_string(data);
  if (filenames.find(fname_string) != filenames.end()) {
    send_simpl((char*)"NO_WAY", cmd_seq, data, data_len);
    return;
  }
  if (data[0] == '\0' || strchr(data, '\\')) {
    send_simpl((char*)"NO_WAY", cmd_seq, data, data_len);
    return;
  }
  int tcp_sock;
  uint64_t tcp_port;
  if ((tcp_sock = create_tcp_sock(&tcp_port)) < 0) {
    printf("TCP connection error...%s\n", strerror(errno));
    return;
  }

  disk_space -= file_size;
  filenames.insert(data);
  send_cmplx((char*)"CAN_ADD", cmd_seq, tcp_port, (char*)"", 0);
  fs::path file_path = fs::absolute(data, directory);
  std::thread(dwnld, tcp_sock, file_path, file_size).detach();
}

uint64_t cmd_seq;
int msg_size;

std::map<int, std::function<void()>> disp_table {
  {ERR_NUM,   []() { print_err_msg(client_address, (char*)"CMD not supported");}},
  {HELLO_NUM, []() { answer_hello(cmd_seq, msg_size);}},
  {LIST_NUM,  []() { answer_list(cmd_seq, msg_size);}},
  {DEL_NUM,   []() { answer_del(msg_size);}},
  {GET_NUM,   []() { answer_get(cmd_seq, msg_size);}},
  {ADD_NUM,   []() { answer_add(cmd_seq, msg_size);}}
};

std::map<string, int, decltype(comp)> cmd_to_int(comp);
void create_map() {
  std::vector<std::string> v1 = {"ADD", "GET", "DEL", "LIST", "HELLO"};
  std::vector<int> v2 = {ADD_NUM, GET_NUM, DEL_NUM, LIST_NUM, HELLO_NUM};
  // std::string s(CMD_SIZE+1, 0);

  for (int i = 0; i < (int)v1.size(); ++i) {
    // strcpy((char*)s.c_str(), v1[i].c_str()); // string s = v1[i]
    cmd_to_int[v1[i]] = v2[i];
  }
}

void exit_handler(int s) {
  (void)s;
  if (setsockopt(sock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (void*)&ip_mreq, sizeof ip_mreq) < 0)
    syserr("setsockopt");
  close(sock);
  exit(EXIT_SUCCESS);
}

int main (int argc, char **argv) {
  parse_cmd_line_args(argc, argv);
  read_filenames();
  setup_udp_sock();
  create_map();
  signal(SIGINT, exit_handler);

  while (1) {
    if ((msg_size=recvfrom(sock, buffer, sizeof(buffer), flags, (struct sockaddr*)&client_address, &client_len)) < 0) {
      printf("Recvfrom error...%s\n", strerror(errno));
      continue;
    }

    char cmd[CMD_SIZE+1];
    get_cmd(buffer, cmd);
    std::string cmd_str(cmd);
    cmd_seq = get_cmd_seq(buffer);
//    printf("%s\n", cmd);
    disp_table[cmd_to_int[cmd_str]]();
  }
}
