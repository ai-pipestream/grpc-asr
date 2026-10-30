#pragma once
#include "http.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

inline void http_fixture_check(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
inline bool send_bytes(int fd, const std::string& bytes) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    const auto sent = ::send(fd, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
    if (sent <= 0) return false;
    offset += sent;
  }
  return true;
}
struct Fixture {
  int listener = -1, client = -1;
  std::mutex mutex;
  std::atomic<bool> stop{false}, release{false};
  std::jthread thread;
  std::string url;
  explicit Fixture(std::function<void(int, Fixture&)> serve, uint16_t port = 0) {
    listener = ::socket(AF_INET, SOCK_STREAM, 0); http_fixture_check(listener >= 0, "fixture socket failed");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    http_fixture_check(::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "fixture bind failed");
    socklen_t size = sizeof(address);
    http_fixture_check(::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) == 0 && ::listen(listener, 1) == 0,
          "fixture listen failed");
    url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)) + "/feed?private-token=secret";
    thread = std::jthread([&, serve = std::move(serve)] {
      const int fd = ::accept(listener, nullptr, nullptr);
      if (fd < 0) return;
      { std::lock_guard lock(mutex); client = fd; }
      char request[4096];
      if (::recv(fd, request, sizeof(request), 0) > 0) serve(fd, *this);
      { std::lock_guard lock(mutex); ::close(fd); client = -1; }
    });
  }
  ~Fixture() {
    stop = true;
    { std::lock_guard lock(mutex); if (client >= 0) ::shutdown(client, SHUT_RDWR); }
    ::shutdown(listener, SHUT_RDWR); thread.join(); ::close(listener);
  }
  media::HttpPolicy policy(bool private_ok = true) const { return {{media::HttpInput::origin(url)}, private_ok}; }
};
